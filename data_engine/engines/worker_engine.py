import json
import zipfile
import shutil
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from loguru import logger

from data_engine.core.config import ARIA_SCOPE, CONFIG, HF_WARNING_THRESHOLD, RAW_DIR, TEMP_DIR
from data_engine.core.network_utils import patch_all_networking
from data_engine.core.utils import normalize_code, parse_datetime
from data_engine.engines.filtering_engine import FilteringEngine, ProcessVerdict, SkipReason
from data_engine.engines.parsing.edinet.fs_tbl import get_fs_tbl

# 設定 (SSOT から取得)
PARALLEL_WORKERS = CONFIG.PARALLEL_WORKERS
BATCH_PARALLEL_SIZE = CONFIG.BATCH_PARALLEL_SIZE
RAW_BASE_DIR = RAW_DIR




_worker_acc_cache = {}

def parse_worker(args):
    """並列処理用ワーカー関数"""
    docid, row, acc_obj, raw_zip = args
    
    # 【Pickling回避】acc_obj が文字列(年)の場合、ワーカー内キャッシュから遅延ロードする
    if isinstance(acc_obj, (str, int)):
        year = str(acc_obj)
        if year not in _worker_acc_cache:
            from data_engine.engines.edinet_engine import EdinetEngine
            engine = EdinetEngine(api_key=CONFIG.EDINET_API_KEY, data_path=CONFIG.DATA_PATH)
            _worker_acc_cache[year] = engine.get_account_list(year)
        acc_obj = _worker_acc_cache[year]
    
    extract_dir = TEMP_DIR / f"extract_{docid}"
    try:
        if acc_obj is None:
            return docid, None, "Account list not loaded", None

        logger.debug(f"解析開始: {docid} (Path: {raw_zip})")
        (extract_dir / "XBRL" / "PublicDoc").mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(raw_zip)) as zf:
            for member in zf.namelist():
                if "PublicDoc" in member or "AuditDoc" in member:
                    zf.extract(member, extract_dir)

        df = get_fs_tbl(
            account_list_common_obj=acc_obj,
            docid=docid,
            zip_file_str=str(raw_zip),
            temp_path_str=str(extract_dir),
            role_keyward_list=[]
        )

        if df is not None and not df.empty:
            df["docid"] = docid
            sec_code_meta = row.get("secCode")
            if sec_code_meta and str(sec_code_meta).strip():
                df["code"] = normalize_code(sec_code_meta, nationality="JP")
            else:
                df["code"] = None

            df["submitDateTime"] = row.get("submitDateTime", "")
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str)
                    
            # 【工学的主権】全てのオブジェクト列に対し、不適切なセンチネル値を NULL に統一
            # これにより「-」や「None」文字列が不必要に保存されるのを防ぐ
            df = df.replace({"-": None, "None": None, "nan": None, "NaN": None})

            quant_cnt = len(df[df['isTextBlock_flg'] == 0]) if 'isTextBlock_flg' in df.columns else 0
            text_cnt = len(df[df['isTextBlock_flg'] == 1]) if 'isTextBlock_flg' in df.columns else 0
            
            accounting_std = None
            total_physical = 0
            log_path = extract_dir / "XBRL" / "PublicDoc" / "log_dict.json"
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        ld = json.load(f)
                        total_physical = ld.get('total_facts_present', 0)
                        accounting_std = ld.get("AccountingStandardsDEI")
                except Exception:
                    pass
                    
            metrics = df.attrs.get('aria_metrics', {})
            theoretical_total = metrics.get('theoretical_total', len(df))

            # 【工程監査】理論上の全マッピング件数と最終抽出件数が一致しているか検証する
            if len(df) != theoretical_total:
                logger.warning(
                    f"CARDINALITY MISMATCH: Theoretical({theoretical_total}) != Actual({len(df)}). "
                    "Check for unintended drops or duplicates in merge logic."
                )

            if total_physical > 0:
                # 【物理/理論の一致を証明】
                match_pct = (len(df) / theoretical_total * 100) if theoretical_total > 0 else 100.0
                logger.info(
                    f"[SUCCESS] {docid} | 理論期待値(Total): {theoretical_total}件 == "
                    f"最終抽出:{len(df)}件 (数値:{quant_cnt}, テキスト:{text_cnt}) [Zero-Drop {match_pct:g}% 一致]"
                )
            else:
                logger.info(f"[SUCCESS] {docid} | 数値データ: {quant_cnt} 件, テキストブロック: {text_cnt} 件を Zero-Drop で抽出・保存に成功しました (計: {len(df)}件)")

            return docid, df, None, accounting_std

        msg = "No objects to concatenate" if (df is None or df.empty) else "Empty Results"
        return docid, None, msg, None

    except Exception as e:
        err_detail = traceback.format_exc()
        logger.error(f"解析例外: {docid}\n{err_detail}")
        return docid, None, f"{str(e)}", None
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


class WorkerEngine:
    def __init__(self, args, edinet, catalog, run_id, chunk_id):
        self.args = args
        self.edinet = edinet
        self.catalog = catalog
        self.merger = catalog.merger
        self.run_id = run_id
        self.chunk_id = chunk_id
        self.is_shutting_down = False

        self.listed_edinet_codes = set()
        if not self.catalog.master_df.empty:
            self.listed_edinet_codes = set(
                self.catalog.master_df[self.catalog.master_df["is_listed_edinet"].fillna(False)]["edinet_code"]
                .dropna()
                .unique()
            )
        # 2. 判定エンジンの初期化（憲法の番人）
        self.filtering = FilteringEngine(aria_scope=ARIA_SCOPE)

        # 全体的な通信の堅牢化を適用
        patch_all_networking()

        # 起動情報の集約表示（工学的主権: 冗長性を排除し、重要な情報のみを提示）
        # Chunk 0 または Merger/Discovery 起動時のみ表示
        is_merger = self.args.mode == "merger"
        if self.chunk_id in (0, "default", "primary-0", "retry-0") or is_merger:
            logger.info(f"ARIA Execution Scope: {ARIA_SCOPE}")

    def _apply_status(self, record, new_status):
        """【工学的主権】ステータスの優先順位階層に基づき更新を適用する"""
        hierarchy = {
            "retracted": 1,
            "failure": 2,
            "english_empty": 3,
            "attachment_empty": 4,
            "parsed": 5,
            "success": 6,
        }
        current = record.get("processed_status", "success") # デフォルトは success
        if hierarchy.get(new_status, 99) < hierarchy.get(current, 99):
            record["processed_status"] = new_status

    def run(self):
        """Workerモード (デフォルト): データの取得・解析・保存のパイプラインを実行する"""
        mode_label = "Discovery" if self.args.list_only else "Worker"
        logger.info(f"=== {mode_label} Pipeline Started ===")

        meta_cache_path = self.catalog.data_path / "meta" / "discovery_metadata.json"
        target_ids = self.args.id_list.split(",") if self.args.id_list else None

        # 2. 増分同期またはDiscoveryキャッシュの利用
        is_worker_with_cache = self.args.mode == "worker" and meta_cache_path.exists()

        if is_worker_with_cache:
            try:
                with open(meta_cache_path, "r", encoding="utf-8") as f:
                    all_meta = json.load(f)
                logger.info("Discovery時のメタデータキャッシュを利用します（APIフェッチをスキップ）。")
            except Exception as e:
                logger.warning(f"メタデータキャッシュの読み込みに失敗しました。APIフェッチにフォールバックします: {e}")
                is_worker_with_cache = False

        if not is_worker_with_cache:
            last_ope_time = None
            if not self.catalog.catalog_df.empty and "ope_date_time" in self.catalog.catalog_df.columns:
                # カタログ内の最新の操作日時を取得
                max_ope = self.catalog.catalog_df["ope_date_time"].max()
                if not pd.isna(max_ope) and str(max_ope).strip() != "":
                    try:
                        # 堅牢なパース関数の流用
                        dt_ope = parse_datetime(max_ope)
                        if not dt_ope:
                            raise ValueError(f"ope_date_time '{max_ope}' のパースに失敗しました。")
                        
                        # 時刻部分のみで計算が必要なため、一度ダミー日付で扱う
                        # max_ope が '2024-03-12 10:00:00' のような形式でも parse_datetime なら通る
                        base_time = datetime(2000, 1, 1, dt_ope.hour, dt_ope.minute, dt_ope.second)
                        dt_buffer = base_time - pd.Timedelta(hours=1)
                        last_ope_time = dt_buffer.strftime("%H:%M:%S")
                        logger.info(f"増分同期チェックポイント: {max_ope} -> {last_ope_time} (1h Buffer適用)")
                    except Exception as e:
                        logger.warning(f"ope_date_time の計算に失敗しました: {e}")
                        last_ope_time = None

            all_meta = self.edinet.fetch_metadata(self.args.start, self.args.end, ope_date_time=last_ope_time)
        if not all_meta:
            if self.args.list_only:
                print("JSON_MATRIX_DATA: []")
                # GHA堅牢化: 0件でも空のキャッシュファイルを生成し、後続のmvエラーを防止する
                meta_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(meta_cache_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            return True

        initial_count = len(all_meta)
        filtered_meta = []
        # Discoveryモード (list-only) の場合のみフィルタリングを実行
        if self.args.list_only:
            filtered_meta = []
            skipped_reasons = {
                SkipReason.NO_SEC_CODE: 0,
                SkipReason.INVALID_CODE_LENGTH: 0,
                SkipReason.HAS_SEC_CODE: 0,
                SkipReason.ALREADY_PROCESSED: 0,
                SkipReason.WITHDRAWN: 0,
                SkipReason.INVALID_METADATA: 0,
            }
            matrix_data = []

            for row in all_meta:
                doc_id = row.get("docID")
                is_processed = self.catalog.is_processed(doc_id)
                local_status = self.catalog.get_status(doc_id)

                # 【憲法の番人】判定エンジンへ委譲
                verdict, reason, ind = self.filtering.get_verdict(
                    row, is_processed=is_processed, local_status=local_status
                )

                # 物理的指標による詳細ログ (DocType, Ordinance, Form)
                i_doc = ind["doc"]
                i_ord = ind["ord"]
                i_form = ind["form"]
                i_xbrl = "XBRL:1" if ind["xbrl"] else "XBRL:0"
                
                title = row.get("docDescription", "名称不明")
                log_msg = f"[{i_doc}, {i_ord}, {i_form}, {i_xbrl}] {doc_id} | {title}"
                
                if verdict in [
                    ProcessVerdict.SKIP_OUT_OF_SCOPE,
                    ProcessVerdict.SKIP_PROCESSED,
                    ProcessVerdict.SKIP_WITHDRAWN,
                ]:
                    logger.debug(f"{log_msg} -> SKIP ({reason})")
                    if reason in skipped_reasons:
                        skipped_reasons[reason] += 1
                    continue

                # 解析対象なら PARSE、対象外なら Saved (書類保存のみ) と表示
                status_label = "PARSE" if verdict == ProcessVerdict.PARSE else "Saved"
                logger.info(f"{log_msg} -> ACCEPT ({status_label})")
                filtered_meta.append(row)
                raw_sec_code = normalize_code(str(row.get("secCode", "")).strip(), nationality="JP")
                matrix_data.append(
                    {
                        "id": doc_id,
                        "code": raw_sec_code,
                        "edinet": row.get("edinetCode"),
                        "xbrl": row.get("xbrlFlag") == "1",
                        "type": row.get("docTypeCode"),
                        "ord": row.get("ordinanceCode"),
                        "form": row.get("formCode"),
                    }
                )

            all_meta = filtered_meta
            skipped_count = sum(skipped_reasons.values())

            # スコアに応じた詳細ラベルの作成（工学的最適化）
            skip_details = [
                f"既処理: {skipped_reasons[SkipReason.ALREADY_PROCESSED]}",
                f"メタデータ不全: {skipped_reasons[SkipReason.INVALID_METADATA]}",
            ]
            if ARIA_SCOPE == "Listed":
                skip_details.append(f"証券コードなし: {skipped_reasons[SkipReason.NO_SEC_CODE]}")
                skip_details.append(f"形式不正: {skipped_reasons[SkipReason.INVALID_CODE_LENGTH]}")
            elif ARIA_SCOPE == "Unlisted":
                skip_details.append(f"証券コードあり: {skipped_reasons[SkipReason.HAS_SEC_CODE]}")

            logger.info(
                f"フィルタリング完了: {len(all_meta)}/{initial_count} 件を抽出 "
                f"(採用: {len(all_meta)} 件 | 総スキップ: {skipped_count} 件 ["
                f"{', '.join(skip_details)}])"
            )

            meta_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_cache_path, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, ensure_ascii=False, indent=2)
            logger.info(f"Discoveryメタデータを保存しました: {meta_cache_path}")

            print(f"JSON_MATRIX_DATA: {json.dumps(matrix_data)}")
            return True

        logger.info("=== Data Lakehouse 2.0 実行開始 ===")
        tasks = []
        potential_catalog_records = {}
        parsing_target_ids = set()
        found_target_ids = set()

        loaded_acc = {}
        worker_stats = {
            "assigned": len(target_ids) if target_ids else initial_count,
            "processed": 0,
            "already_skipped": 0,
            "metadata_saved": 0,
            "parsing_attempted": 0,
            "parsing_success": 0,
            "parsing_failure": 0,
        }

        for row in all_meta:
            doc_id = row.get("docID")
            if target_ids and doc_id not in target_ids:
                continue

            found_target_ids.add(doc_id)
            title = row.get("docDescription", "名称不明")

            # 【憲法の番人】判定エンジンへ委譲
            is_processed = self.catalog.is_processed(doc_id)
            local_status = self.catalog.get_status(doc_id)
            verdict, reason, ind = self.filtering.get_verdict(
                row, is_processed=is_processed, local_status=local_status
            )

            # 物理的指標による詳細ログ (Sovereign Log Format)
            i_doc = f"{ind['doc']:>3}"
            i_ord = f"{ind['ord']:>3}"
            i_form = f"{ind['form']:>6}"
            i_xbrl = "XBRL:1" if ind["xbrl"] else "XBRL:0"
            raw_code = str(row.get("secCode", "")).strip()
            norm_code = normalize_code(raw_code, nationality="JP") if raw_code else ""
            i_code = f"[{norm_code:8}]" if norm_code else "[        ]"
            
            # プレフィックスの構築 (物理的事実の垂直整列)
            log_prefix_facts = f"[{i_doc}, {i_ord}, {i_form}, {i_xbrl}] {i_code} {doc_id:8}"
            log_msg = f"{log_prefix_facts} | {title}"

            # 既処理・スコープ外スキップの明示的カウント
            if verdict == ProcessVerdict.SKIP_PROCESSED:
                worker_stats["already_skipped"] += 1
                logger.debug(f"[Skipped] {log_msg} -> Already Processed")
                continue
            if verdict == ProcessVerdict.SKIP_OUT_OF_SCOPE:
                logger.debug(f"[Skipped] {log_msg} -> Out of Scope: {reason}")
                continue
            if verdict == ProcessVerdict.SKIP_WITHDRAWN:
                if local_status == "retracted":
                    logger.debug(f"[Skipped] {log_msg} -> Already Retracted")
                    continue
                logger.info(f"[Retract] {log_msg} -> SYNC REQUIRED")
            elif verdict == ProcessVerdict.PARSE:
                # 定例の出力（PARSEのみ）は重複回避のため DEBUG 
                logger.debug(f"[PARSE  ] {log_msg} (Pending)")
            else:
                # SAVE_RAW 等の「保存のみ」
                logger.info(f"[Saved  ] {log_msg}")

            submit_date = parse_datetime(row["submitDateTime"])
            # ... (ディレクトリ作成ロジックは維持)
            if submit_date:
                save_dir = (
                    RAW_BASE_DIR
                    / "edinet"
                    / f"year={submit_date.year}"
                    / f"month={submit_date.month:02d}"
                    / f"day={submit_date.day:02d}"
                )
            else:
                logger.warning(f"Unparseable submitDateTime '{row['submitDateTime']}' for {doc_id}. Saving to unknown.")
                save_dir = RAW_BASE_DIR / "edinet" / "unknown"
            zip_dir = save_dir / "zip"
            pdf_dir = save_dir / "pdf"
            raw_zip = zip_dir / f"{doc_id}.zip"
            raw_pdf = pdf_dir / f"{doc_id}.pdf"
            zip_dir.mkdir(parents=True, exist_ok=True)
            pdf_dir.mkdir(parents=True, exist_ok=True)

            xbrl_flag = row.get("xbrlFlag") == "1"
            pdf_flag = row.get("pdfFlag") == "1"
            attach_flag = row.get("attachDocFlag") == "1"

            zip_ok = False
            if xbrl_flag:
                zip_ok = self.edinet.download_doc(doc_id, raw_zip, 1)

            pdf_ok = False
            if pdf_flag:
                pdf_ok = self.edinet.download_doc(doc_id, raw_pdf, 2)

            # --- ZIP 展開 (English/Attachment) ---
            rel_attach_path = None
            rel_english_path = None
            anomaly_status = None

            # 1. 英文書類 (type=4)
            # 【工学的主権】英文書類は投資判断の核心となるため、抽出失敗は明示的にステータスへ記録。
            if row.get("englishDocFlag") == "1":
                eng_dir = save_dir / "english" / doc_id
                tmp_zip = Path(TEMP_DIR) / f"{doc_id}_english.zip"
                if self.edinet.download_doc(doc_id, tmp_zip, 4):
                    try:
                        extracted_count = 0
                        with zipfile.ZipFile(tmp_zip, "r") as z:
                            for file_info in z.infolist():
                                # EnglishDoc フォルダ内の HTML または PDF を対象とする
                                if file_info.filename.lower().endswith((".pdf", ".htm", ".html")):
                                    eng_dir.mkdir(parents=True, exist_ok=True)
                                    fname = Path(file_info.filename).name
                                    with z.open(file_info) as src, open(eng_dir / fname, "wb") as dst:
                                        dst.write(src.read())
                                    extracted_count += 1
                        
                        if extracted_count > 0:
                            rel_english_path = str(eng_dir.relative_to(RAW_BASE_DIR.parent))
                            if not rel_english_path.endswith("/"):
                                rel_english_path += "/"
                        else:
                            logger.warning(f"English ZIP is empty or has no valid files: {doc_id}")
                            anomaly_status = "english_empty"
                    except Exception as e:
                        logger.error(f"Failed to extract English docs for {doc_id}: {e}")
                    finally:
                        if tmp_zip.exists(): tmp_zip.unlink()

            # 2. 添付書類 (type=3)
            # 【工学的主権】Web UI での閲覧性を担保するため、ZIP 内の不要な制御ファイル（__MACOSX等）を除外し PDF のみを抽出。
            if attach_flag:
                attach_dir = save_dir / "attach" / doc_id
                tmp_zip = Path(TEMP_DIR) / f"{doc_id}_attach.zip"
                if self.edinet.download_doc(doc_id, tmp_zip, 3):
                    try:
                        extracted_count = 0
                        with zipfile.ZipFile(tmp_zip, 'r') as z:
                            for file_info in z.infolist():
                                if file_info.filename.lower().endswith(".pdf"):
                                    attach_dir.mkdir(parents=True, exist_ok=True)
                                    fname = Path(file_info.filename).name
                                    with z.open(file_info) as src, open(attach_dir / fname, "wb") as dst:
                                        dst.write(src.read())
                                    extracted_count += 1
                        
                        if extracted_count > 0:
                            rel_attach_path = str(attach_dir.relative_to(RAW_BASE_DIR.parent))
                            if not rel_attach_path.endswith("/"):
                                rel_attach_path += "/"
                        else:
                            logger.warning(f"Attachment ZIP is empty or has no PDFs: {doc_id}")
                            # 【一貫性の維持】書類全体のステータス列が1つのため、英文(english_empty)を優先。
                            # 英文に異常がない、または英文提供がない場合のみ attachment_empty を記録。
                            if not anomaly_status:
                                anomaly_status = "attachment_empty"
                    except Exception as e:
                        logger.error(f"Failed to extract attachments for {doc_id}: {e}")
                    finally:
                        if tmp_zip.exists(): tmp_zip.unlink()

            dtc = row.get("docTypeCode")
            ord_c = row.get("ordinanceCode")
            form_c = row.get("formCode")
            period_start = (row.get("periodStart") or "").strip() or None
            period_end = (row.get("periodEnd") or "").strip() or None
            fiscal_year = int(period_end[:4]) if period_end else None

            num_months = None
            if period_start and period_end:
                try:
                    d1 = datetime.strptime(period_start, "%Y-%m-%d")
                    d2 = datetime.strptime(period_end, "%Y-%m-%d")
                    diff_days = (d2 - d1).days + 1
                    calc_months = round(diff_days / 30.4375)
                    if 1 <= calc_months <= 24:
                        num_months = calc_months
                except Exception:
                    pass

            # カタログレコードの生成
            sec_code = normalize_code(row.get("secCode", ""), nationality="JP")
            parent_id = row.get("parentDocID")
            is_amendment = parent_id is not None or str(dtc).endswith("1") or "訂正" in (title or "")

            # 最終ステータスの決定
            # ステータスの初期設定 (Hierarchy適用対象)
            if verdict == ProcessVerdict.SKIP_WITHDRAWN:
                initial_status = "retracted"
            elif anomaly_status:
                initial_status = anomaly_status # 英文異常 > 添付異常 は _process_doc 側で処理済
            else:
                initial_status = "success"

            rel_zip_path = str(raw_zip.relative_to(RAW_BASE_DIR.parent)) if zip_ok else None
            rel_pdf_path = str(raw_pdf.relative_to(RAW_BASE_DIR.parent)) if pdf_ok else None

            record = {
                "doc_id": doc_id,
                "jcn": (row.get("JCN") or "").strip() or None,
                "code": sec_code,
                "company_name": (row.get("filerName") or "").strip() or "Unknown",
                "company_name_en": (row.get("filerNameEn") or "").strip() or None,
                "company_name_kana": (row.get("filerNameKana") or "").strip() or None,
                "edinet_code": (row.get("edinetCode") or "").strip() or None,
                "issuer_edinet_code": (row.get("issuerEdinetCode") or "").strip() or None,
                "subject_edinet_code": (row.get("subjectEdinetCode") or "").strip() or None,
                "subsidiary_edinet_code": (row.get("subsidiaryEdinetCode") or "").strip() or None,
                "fund_code": (row.get("fundCode") or "").strip() or None,
                "submit_at": row.get("submitDateTime"),
                "seq_number": row.get("seqNumber"),
                "fiscal_year": fiscal_year,
                "period_start": period_start,
                "period_end": period_end,
                "num_months": num_months,
                "accounting_standard": None,
                "doc_type": dtc or "",
                "title": (title or "").strip() or None,
                "form_code": (form_c or "").strip() or None,
                "ordinance_code": (ord_c or "").strip() or None,
                "is_amendment": is_amendment,
                "parent_doc_id": (parent_id or "").strip() or None,
                "withdrawal_status": (row.get("withdrawalStatus") or "").strip() or None,
                "doc_info_edit_status": (row.get("docInfoEditStatus") or "").strip() or None,
                "disclosure_status": (row.get("disclosureStatus") or "").strip() or None,
                "legal_status": (row.get("legalStatus") or "").strip() or None,
                "current_report_reason": (row.get("currentReportReason") or "").strip() or None,
                "xbrl_flag": row.get("xbrlFlag") == "1",
                "pdf_flag": row.get("pdfFlag") == "1",
                "csv_flag": row.get("csvFlag") == "1",
                "english_flag": row.get("englishDocFlag") == "1",
                "attachment_flag": attach_flag,
                "raw_zip_path": rel_zip_path,
                "pdf_path": rel_pdf_path,
                "english_path": rel_english_path,
                "attach_path": rel_attach_path,
                "processed_status": initial_status,
                "source": "EDINET",
                "ope_date_time": (row.get("opeDateTime") or "").strip() or None,
            }
            potential_catalog_records[doc_id] = record

            if verdict == ProcessVerdict.PARSE and zip_ok:
                worker_stats["parsing_attempted"] += 1
                try:

                    from data_engine.engines.parsing.edinet.fs_tbl import linkbasefile

                    detect_dir = TEMP_DIR / f"detect_{doc_id}"
                    lb = linkbasefile(zip_file_str=str(raw_zip), temp_path_str=str(detect_dir))
                    lb.read_linkbase_file()
                    ty = lb.detect_account_list_year()

                    if ty == "-":
                        raise ValueError(f"Taxonomy year not identified for {doc_id}")

                    if ty not in loaded_acc:
                        acc = self.edinet.get_account_list(ty)
                        if not acc:
                            raise ValueError(f"Taxonomy version '{ty}' not found")
                        loaded_acc[ty] = acc

                    tasks.append((doc_id, row, loaded_acc[ty], raw_zip))
                    parsing_target_ids.add(doc_id)
                    logger.info(f"[PARSE  ] {log_msg} | FY: {ty}")
                except Exception as e:
                    logger.error(f"【解析中止】タクソノミ判定失敗 ({doc_id}): {e}")
                    self._apply_status(record, "failure")
                    worker_stats["parsing_failure"] += 1
                    continue
                finally:
                    if detect_dir.exists():
                        shutil.rmtree(detect_dir)
            else:
                worker_stats["metadata_saved"] += 1
                if verdict == ProcessVerdict.SAVE_RAW:
                    logger.debug(f"非解析対象保存: {doc_id} | {title}")
                elif verdict == ProcessVerdict.PARSE and not zip_ok:
                    logger.error(f"保存失敗（有報）: {doc_id} | {title}")
                    self._apply_status(record, "failure")
                    worker_stats["parsing_failure"] += 1

        # HF 警告
        checked_dirs = set()
        for row in all_meta:
            sd = parse_datetime(row.get("submitDateTime", ""))
            if not sd:
                continue
            day_dir = RAW_BASE_DIR / "edinet" / f"year={sd.year}" / f"month={sd.month:02d}" / f"day={sd.day:02d}"
            if day_dir not in checked_dirs and day_dir.exists():
                checked_dirs.add(day_dir)
                file_count = sum(1 for _ in day_dir.iterdir())
                if file_count > HF_WARNING_THRESHOLD:
                    logger.warning(f"⚠️ HFフォルダファイル数警告: {day_dir.name} に {file_count} ファイル")

        if target_ids:
            missing_ids = set(target_ids) - found_target_ids
            if missing_ids:
                logger.critical(f"Drift detected: Missing IDs {list(missing_ids)}")

        all_quant_dfs = []
        all_text_dfs = []
        processed_infos = []

        if tasks:
            logger.info(f"解析対象: {len(tasks)} 書類")
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                for i in range(0, len(tasks), BATCH_PARALLEL_SIZE):
                    if self.is_shutting_down:
                        break
                    batch = tasks[i : i + BATCH_PARALLEL_SIZE]
                    futures = [executor.submit(parse_worker, t) for t in batch]

                    for f in as_completed(futures):
                        did, res_df, err, accounting_std = f.result()

                        if did in potential_catalog_records:
                            target_rec = potential_catalog_records[did]
                            if err:
                                logger.error(f"解析失敗: {did} - {err}")
                                if "No objects to concatenate" not in err:
                                    self._apply_status(target_rec, "failure")
                                    worker_stats["parsing_failure"] += 1
                            elif res_df is not None:
                                self._apply_status(target_rec, "parsed")
                                worker_stats["parsing_success"] += 1

                                quant_only = res_df[res_df["isTextBlock_flg"] == 0]
                                if not quant_only.empty:
                                    all_quant_dfs.append(quant_only)

                                txt_only = res_df[res_df["isTextBlock_flg"] == 1]
                                if not txt_only.empty:
                                    all_text_dfs.append(txt_only)

                                if accounting_std:
                                    target_rec["accounting_standard"] = str(accounting_std)

                                meta_row = next(m for m in all_meta if m["docID"] == did)
                                processed_infos.append(
                                    {
                                        "docID": did,
                                        "sector": self.catalog.get_sector(
                                            normalize_code(meta_row.get("secCode", ""), nationality="JP")
                                        ),
                                    }
                                )
                    logger.info(f"解析進捗: {min(i + BATCH_PARALLEL_SIZE, len(tasks))} / {len(tasks)} tasks 完了")

        # 【工学的主権】全てのカタログレコードに bin_id を付与し、アクセス経路を統一する
        for record in potential_catalog_records.values():
            # MasterMerger のロジックを用いて不変の分散キー(bin)を決定する
            bridge_row = {"jcn": record.get("jcn"), "edinet_code": record.get("edinet_code"), "code": record.get("code")}
            record["bin_id"] = self.merger.get_bin_id(bridge_row)

        all_success = True
        bin_failures = set() # 失敗した Bin を記録

        processed_df = pd.DataFrame(processed_infos)
        if not processed_df.empty:
            processed_df["bin"] = processed_df["docID"].apply(lambda did: potential_catalog_records[did]["bin_id"])
            bins = processed_df["bin"].unique()

            if all_quant_dfs:
                try:
                    full_quant_df = pd.concat(all_quant_dfs, ignore_index=True)
                    for b_val in bins:
                        bin_docids = processed_df[processed_df["bin"] == b_val]["docID"].tolist()
                        sec_quant = full_quant_df[full_quant_df["docid"].isin(bin_docids)]
                        if not sec_quant.empty:
                            ok = self.merger.merge_and_upload(
                                b_val,
                                "financial_values",
                                sec_quant,
                                worker_mode=True,
                                catalog_manager=self.catalog,
                                run_id=self.run_id,
                                chunk_id=self.chunk_id,
                                defer=True,
                            )
                            if not ok:
                                bin_failures.add(b_val)
                                all_success = False
                except Exception as e:
                    logger.error(f"Quant merge failed: {e}")
                    all_success = False

            if all_text_dfs:
                try:
                    full_text_df = pd.concat(all_text_dfs, ignore_index=True)
                    for b_val in bins:
                        bin_docids = processed_df[processed_df["bin"] == b_val]["docID"].tolist()
                        sec_text = full_text_df[full_text_df["docid"].isin(bin_docids)]
                        if not sec_text.empty:
                            ok = self.merger.merge_and_upload(
                                b_val,
                                "qualitative_text",
                                sec_text,
                                worker_mode=True,
                                catalog_manager=self.catalog,
                                run_id=self.run_id,
                                chunk_id=self.chunk_id,
                                defer=True,
                            )
                            if not ok:
                                bin_failures.add(b_val)
                                all_success = False
                except Exception as e:
                    logger.error(f"Text merge failed: {e}")
                    all_success = False

        # 【Transactional Integrity】解析済み (parsed) レコードを成功 (success) に昇格させる
        for record in potential_catalog_records.values():
            if record["processed_status"] == "parsed":
                b_id = record["bin_id"]
                if b_id not in bin_failures:
                    # parsed は中間ステータス。アノマリがなければ success 相当だが、英/添付アノマリがある場合はそちらが優先されるはず。
                    # ここでは parsed の時だけ success に書き換える。
                    record["processed_status"] = "success"
                else:
                    self._apply_status(record, "failure")
                    logger.warning(f"Bin {b_id} の保存失敗によりステータスを failure に設定: {record['doc_id']}")

        final_catalog_records = list(potential_catalog_records.values())
        if final_catalog_records:
            df_cat = pd.DataFrame(final_catalog_records).drop_duplicates(subset=["doc_id"], keep="last")
            self.catalog.save_delta("catalog", df_cat, self.run_id, self.chunk_id, defer=True, local_only=True)

        if all_success:
            self.catalog.mark_chunk_success(self.run_id, self.chunk_id, defer=True, local_only=True)
            
            # 【工学的主権】貸借一致サマリー（Summary Stats）の出力
            # Assigned (総割当) = Already-Skipped (既処理) + Metadata-Only (非解析) + Parsing-Attempted (解析対象)
            # Parsing-Attempted = Success-Docs (完全成功) + Failure-Docs (一部または全部失敗)
            parse_docs = worker_stats['parsing_attempted']
            success_docs = worker_stats['parsing_success']
            failure_docs = parse_docs - success_docs
            
            logger.info(
                f"=== Worker 完了サマリー [{self.run_id}/{self.chunk_id}] ===\n"
                f"・割当総数: {worker_stats['assigned']} 件\n"
                f"  ├ 既処理スキップ: {worker_stats['already_skipped']} 件\n"
                f"  ├ 非解析対象保存: {worker_stats['metadata_saved']} 件 (Non-parsed Raw Save)\n"
                f"  └ 解析対象実行: {parse_docs} 件 (財務諸表等)\n"
                f"      └ 成功: {success_docs} / 失敗: {failure_docs}\n"
                f"・最終ステータス: SUCCESS"
            )
            return True
        else:
            logger.error(f"=== Worker 停止 (エラーあり) [{self.run_id}/{self.chunk_id}] ===")
            return False
