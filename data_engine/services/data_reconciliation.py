"""
ARIA Data Reconciliation Engine
Model-Driven Automated Quality Assurance

旧 integrity_audit.py を完全に置き換え、以下の「4層・11項目」の自律検証を行う：
Layer 1: スキーマ照合（マスター vs Pydantic Model）
Layer 2: 物理ファイル照合（ZIP/PDFの存在と破損チェック）
Layer 3: 分析マスタ照合（Binアサインメント、doc_id重複、孤児レコード）
Layer 4: APIカタログ照合（10年枠内のメタデータ不一致検証）
"""

import json
import os
import re
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from data_engine.catalog_manager import CatalogManager
from data_engine.core import utils
# ARIA モジュール
from data_engine.core.models import CatalogRecord, ListingEvent, StockMasterRecord


class DataReconciliationEngine:
    def __init__(self, hf_repo: str, hf_token: str, data_path: Path, repair: bool = False):
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.data_path = data_path
        self.repair = repair

        # 内部状態として CatalogManager を持つが、スコープは 'All' にして全量監査を行う
        # 修復モード (repair=True) の時のみマスターの動的更新を許可し、監査モードでは不要なAPIコールと更新をスキップする
        self.cm = CatalogManager(hf_repo, hf_token, data_path, scope="All", sync_master=False)

        # 検証エラーの集計
        self.anomalies = {
            "Layer1_Schema": [],
            "Layer2_Physical": [],
            "Layer2_Metadata": [],
            "Layer3_Analytical": [],
            "Layer4_Catalog": [],
            "Layer5_Indexing": [],
        }
        # 修復アクションの記録
        self.repairs = {
            "Layer1_Schema": [],
            "Layer2_Metadata": [],
            "Layer3_Analytical": [],
            "Layer5_Indexing": [],
        }

    def _report_anomaly(self, layer: str, message: str, doc_id: str = None, details: Any = None):
        anomaly = {"message": message}
        if doc_id:
            anomaly["doc_id"] = doc_id
        if details:
            anomaly["details"] = details

        self.anomalies[layer].append(anomaly)
        logger.warning(f"[{layer}] Anomaly detected: {message}")

    # ==========================================
    # Layer 1: Schema Reconciliation
    # ==========================================
    def reconcile_schemas(self):
        """PydanticモデルとParquetカラムの完全一致を検証し、修復を試みる"""
        logger.info("--- [Layer 1] Schema Reconciliation ---")

        check_targets = [
            ("catalog", "documents_index.parquet", CatalogRecord),
            ("master", "stocks_master.parquet", StockMasterRecord),
            ("listing", "listing_history.parquet", ListingEvent),
        ]

        for key, filename, model_class in check_targets:
            try:
                # 破損チェックを兼ねてロード試行
                df = None
                try:
                    df = self.cm.hf.load_parquet(key)
                except Exception as e:
                    logger.error(f"Critical corruption in {filename}: {e}")
                    if self.repair:
                        df = self._attempt_file_rollback(key)
                    if df is None:
                        continue

                if df.empty:
                    logger.info(f"{filename} is empty. Skipping schema check.")
                    continue

                model_fields = set(model_class.model_fields.keys())
                parquet_columns = set(df.columns)

                missing_in_parquet = model_fields - parquet_columns
                extra_in_parquet = parquet_columns - model_fields

                if missing_in_parquet or extra_in_parquet:
                    self._report_anomaly(
                        "Layer1_Schema",
                        f"Schema mismatch in {filename}",
                        details={"missing": list(missing_in_parquet), "extra": list(extra_in_parquet)},
                    )

                    if self.repair:
                        logger.info(f"Repairing schema for {filename}...")
                        # 欠落列の補完 (NULL) と余剰列の削除、型の正規化
                        self.cm.hf.save_and_upload(key, df, defer=True)
                        logger.info(f"Schema normalization staged for {filename}.")

                else:
                    logger.info(f"✅ {filename} matches {model_class.__name__} perfectly.")
            except Exception as e:
                self._report_anomaly("Layer1_Schema", f"Failed to verify schema for {filename}: {e}")

    def _attempt_file_rollback(self, key: str) -> pd.DataFrame:
        """HF のコミット履歴を遡り、正常に開ける最新バージョンを復旧させる"""
        logger.info(f"Attempting file rollback repair for {key}...")
        history = self.cm.hf.get_file_history(key)

        for commit_hash in history[1:]:  # 最新(現在)は既に壊れているため 1 つ前から
            try:
                logger.debug(f"Testing revision: {commit_hash[:7]}")
                df = self.cm.hf.load_parquet(key, revision=commit_hash)
                if not df.empty:
                    logger.info(f"Successfully rescued healthy version from commit {commit_hash[:7]}.")
                    # 直ちに修復版としてステージング（Atomic書き戻しの起点）
                    self.cm.hf.save_and_upload(key, df, defer=True)
                    return df
            except Exception:
                continue

        logger.error(f"Failed to find any healthy version in history for {key}.")
        return None

    # ==========================================
    # Layer 2: Physical Asset Reconciliation
    # ==========================================
    def reconcile_physical_assets(self, sample_size: int = 50):
        """物理ファイル(ZIP/PDF)の存在と整合性を検証"""
        logger.info("--- [Layer 2] Physical Asset Reconciliation ---")

        catalog_df = self.cm.catalog_df
        if catalog_df.empty:
            logger.info("Catalog is empty. Skipping physical checks.")
            return

        try:
            # Hugging Face の raw ディレクトリ配下のファイルリストを取得
            files = self.cm.hf.api.list_repo_files(repo_id=self.hf_repo, repo_type="dataset")
            # サブディレクトリ (zip/ pdf/) を含む全ファイルを対象にする
            raw_files = set([f for f in files if f.startswith("raw/edinet/")])

            # 存在すべきファイルの導出
            expected_zips = {}
            expected_pdfs = {}

            for _, row in catalog_df.iterrows():
                doc_id = row["doc_id"]
                submit_date_str = row.get("submit_at")
                if not submit_date_str:
                    continue

                # 【工学的主権】APIの意図(フラグ)に基づき期待値を算出。NaNトラップをpd.isnaで回避。
                # xbrl_flag/pdf_flag が None の場合は、既存レコードとの互換性のためパスの有無で判定
                should_have_zip = row.get("xbrl_flag")
                if should_have_zip is True:
                    expected_zips[doc_id] = utils.get_edinet_repo_path(doc_id, submit_date_str, suffix="zip")
                elif should_have_zip is None and not pd.isna(row.get("raw_zip_path")):
                    expected_zips[doc_id] = utils.get_edinet_repo_path(doc_id, submit_date_str, suffix="zip")

                should_have_pdf = row.get("pdf_flag")
                if should_have_pdf is True:
                    expected_pdfs[doc_id] = utils.get_edinet_repo_path(doc_id, submit_date_str, suffix="pdf")
                elif should_have_pdf is None and not pd.isna(row.get("pdf_path")):
                    expected_pdfs[doc_id] = utils.get_edinet_repo_path(doc_id, submit_date_str, suffix="pdf")

            # 存在確認 (Existence check)
            missing_zips = [doc_id for doc_id, path in expected_zips.items() if path not in raw_files]
            missing_pdfs = [doc_id for doc_id, path in expected_pdfs.items() if path not in raw_files]

            if missing_zips:
                self._report_anomaly(
                    "Layer2_Physical",
                    f"{len(missing_zips)} expected ZIP files are missing from HF storage.",
                    details=missing_zips,
                )
                if self.repair:
                    logger.info("Resetting status for docs with missing ZIPs to trigger downstream purge...")
                    for d_id in missing_zips:
                        self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == d_id, "processed_status"] = "pending"
                        self.repairs["Layer2_Metadata"].append(
                            {"doc_id": d_id, "action": "status_reset_due_to_missing_zip"}
                        )

            if missing_pdfs:
                self._report_anomaly(
                    "Layer2_Physical",
                    f"{len(missing_pdfs)} expected PDF files are missing from HF storage.",
                    details=missing_pdfs,
                )
                if self.repair:
                    logger.info("Resetting status for docs with missing PDFs to trigger downstream purge...")
                    for d_id in missing_pdfs:
                        self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == d_id, "processed_status"] = "pending"
                        self.repairs["Layer2_Metadata"].append(
                            {"doc_id": d_id, "action": "status_reset_due_to_missing_pdf"}
                        )

            # 【工学的主権】メタデータ不整合(Ghost Attributes)の検知
            ghost_zips = [
                doc_id
                for doc_id, row in catalog_df.iterrows()
                if row.get("xbrl_flag") is False and not pd.isna(row.get("raw_zip_path"))
            ]
            ghost_pdfs = [
                doc_id
                for doc_id, row in catalog_df.iterrows()
                if row.get("pdf_flag") is False and not pd.isna(row.get("pdf_path"))
            ]

            if ghost_zips:
                msg = (
                    f"Metadata conflict detected: {len(ghost_zips)} Ghost ZIP paths found "
                    "(API says no ZIP, but catalog has path)."
                )
                self._report_anomaly("Layer2_Metadata", msg, details=ghost_zips)
            if ghost_pdfs:
                msg = (
                    f"Metadata conflict detected: {len(ghost_pdfs)} Ghost PDF paths found "
                    "(API says no PDF, but catalog has path)."
                )
                self._report_anomaly("Layer2_Metadata", msg, details=ghost_pdfs)

            # 修復モード：ゴースト属性の除去 (API定義を優先しカタログを浄化)
            if self.repair and (ghost_zips or ghost_pdfs):
                logger.info("Repairing ghost attributes (resetting invalid paths to NULL and resetting status)...")
                for d_id in ghost_zips:
                    self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == d_id, "raw_zip_path"] = None
                    # 【工学的主権】ステータスをリセットすることで下流の Layer 3 での自動削除を誘発
                    self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == d_id, "processed_status"] = "invalid"
                    self.repairs["Layer2_Metadata"].append(
                        {"doc_id": d_id, "action": "reset_ghost_zip_path_and_status"}
                    )
                for d_id in ghost_pdfs:
                    self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == d_id, "pdf_path"] = None
                    self.repairs["Layer2_Metadata"].append({"doc_id": d_id, "action": "reset_ghost_pdf_path"})

                if ghost_zips:
                    logger.success(f"Successfully removed {len(ghost_zips)} Ghost ZIP paths from catalog.")
                if ghost_pdfs:
                    logger.success(f"Successfully removed {len(ghost_pdfs)} Ghost PDF paths from catalog.")

            logger.info(
                f"Verified existence of {len(expected_zips) - len(missing_zips)} ZIPs "
                f"and {len(expected_pdfs) - len(missing_pdfs)} PDFs."
            )

            # ランダムサンプリングによる ZIP 破損チェック (Integrity check)
            if expected_zips:
                import random

                from huggingface_hub import hf_hub_download

                available_zips = [path for path in expected_zips.values() if path in raw_files]
                sample_paths = random.sample(available_zips, min(sample_size, len(available_zips)))

                if sample_paths:
                    logger.info(f"Running deep CRC integrity check on {len(sample_paths)} sampled ZIP files...")
                corrupted = []
                for p in sample_paths:
                    try:
                        local_p = hf_hub_download(
                            repo_id=self.hf_repo, filename=p, repo_type="dataset", token=self.hf_token
                        )
                        with zipfile.ZipFile(local_p) as z:
                            # testzip() は破損ファイルの最初の名前を返し、正常なら None を返す
                            bad_file = z.testzip()
                            if bad_file:
                                corrupted.append((p, f"Bad inner file: {bad_file}"))
                    except Exception as e:
                        corrupted.append((p, str(e)))

                if corrupted:
                    self._report_anomaly(
                        "Layer2_Physical", f"Found {len(corrupted)} corrupted ZIP files.", details=corrupted
                    )
                    if self.repair:
                        # 破損ファイルを API から再取得
                        c_pending = 0
                        c_unrecoverable = 0
                        for repo_path, _ in corrupted:
                            match = re.search(r"raw/edinet/([^/]+)/", repo_path)
                            if match:
                                doc_id = match.group(1)
                                # 破損ファイルも再ダウンロードが必要なため同様に保護チェック
                                from data_engine.executors.backfill_manager import get_dynamic_limit_date

                                limit_date = get_dynamic_limit_date()

                                submit_at_str = str(
                                    self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, "submit_at"].iloc[0]
                                )
                                if not submit_at_str or submit_at_str == "nan":
                                    continue
                                submit_date = datetime.strptime(submit_at_str.split(" ")[0], "%Y-%m-%d").date()

                                if submit_date >= limit_date:
                                    self.cm.catalog_df.loc[
                                        self.cm.catalog_df["doc_id"] == doc_id, "processed_status"
                                    ] = "pending"
                                    c_pending += 1
                                else:
                                    self.cm.catalog_df.loc[
                                        self.cm.catalog_df["doc_id"] == doc_id, "processed_status"
                                    ] = "unrecoverable"
                                    c_unrecoverable += 1

                        if c_pending > 0:
                            logger.info(f"Marked {c_pending} corrupted files as 'pending' for Harvester.")
                        if c_unrecoverable > 0:
                            logger.warning(
                                f"Marked {c_unrecoverable} corrupted files as 'unrecoverable' "
                                f"(older than API limit: {limit_date})."
                            )
                elif sample_paths:
                    logger.info(f"✅ Deep CRC check passed for all {len(sample_paths)} sampled ZIP files.")

                if self.repair and (missing_zips or missing_pdfs):
                    # 不足ファイルはAPIを直接叩かず、ワーカー(edinet_harvester)に取得させるため pending にリセットする
                    # ただし、API保持期間（10年）を超過している場合は、永久ループ(404)を防ぐため unrecoverable とする
                    from data_engine.executors.backfill_manager import get_dynamic_limit_date

                    limit_date = get_dynamic_limit_date()
                    m_pending = 0
                    m_unrecoverable = 0

                    for doc_id in set(missing_zips + missing_pdfs):
                        submit_at_str = str(
                            self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, "submit_at"].iloc[0]
                        )
                        if not submit_at_str or submit_at_str == "nan":
                            continue

                        submit_date = datetime.strptime(submit_at_str.split(" ")[0], "%Y-%m-%d").date()

                        if submit_date >= limit_date:
                            self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, "processed_status"] = (
                                "pending"
                            )
                            m_pending += 1
                        else:
                            self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, "processed_status"] = (
                                "unrecoverable"
                            )
                            m_unrecoverable += 1

                    if m_pending > 0:
                        logger.info(f"Status reset to 'pending' for {m_pending} missing files.")
                    if m_unrecoverable > 0:
                        logger.warning(
                            f"Marked {m_unrecoverable} missing files as 'unrecoverable' "
                            f"(older than API limit: {limit_date})."
                        )

        except Exception as e:
            self._report_anomaly("Layer2_Physical", f"Physical reconciliation failed: {e}")

    # ==========================================
    # Layer 3: Analytical Data Reconciliation
    # ==========================================
    def reconcile_analytical_data(self):
        """分析用Parquetとカタログの整合性 (孤児・重複・Bin)"""
        logger.info("--- [Layer 3] Analytical Data Reconciliation (Target-reduced O(1) Deduction) ---")

        try:
            # 1. カタログから「各Binにあるべき書類」を数学的に事前計算（O(1) ターゲット・リダクション）
            def calculate_bin_id(r):
                e_code = str(r.get("edinet_code") or "")
                if e_code and len(e_code) >= 2 and e_code not in ["None", "nan"]:
                    return f"E{e_code[-2:]}"
                c_code = str(r.get("code") or "").split(":")[-1]
                if c_code and len(c_code) >= 2 and c_code not in ["None", "nan"]:
                    return f"P{c_code[-3:-1]}"
                jcn = str(r.get("jcn") or "")
                if jcn and len(jcn) >= 2 and jcn not in ["None", "nan"]:
                    return f"J{jcn[-2:]}"
                return "No"

            catalog_df = self.cm.catalog_df.copy()
            catalog_df["expected_bin"] = catalog_df.apply(calculate_bin_id, axis=1)
            # 正常（done）な書類だけを真の監査対象とする
            done_docs = catalog_df[catalog_df["processed_status"] == "success"]
            expected_docs_per_bin = done_docs.groupby("expected_bin")["doc_id"].apply(set).to_dict()

            # Binファイル群の取得
            files = self.cm.hf.api.list_repo_files(repo_id=self.hf_repo, repo_type="dataset")
            bin_files = [
                f
                for f in files
                if f.startswith("master/financial_values/bin=") or f.startswith("master/qualitative_text/bin=")
            ]

            if not bin_files:
                logger.info("No Master Data chunks (Bins) found.")
                return

            processed_bins = set()
            from huggingface_hub import hf_hub_download

            # 各 Bin ファイルごとの独立した O(1) 検証ループ
            for bf in bin_files:
                match = re.search(r"bin=([^/]+)/", bf)
                expected_bin = match.group(1) if match else "Unknown"
                processed_bins.add(expected_bin)

                expected_docs = expected_docs_per_bin.get(expected_bin, set())

                try:
                    df = None
                    try:
                        local_p = hf_hub_download(
                            repo_id=self.hf_repo, filename=bf, repo_type="dataset", token=self.hf_token
                        )
                        df = pd.read_parquet(local_p)
                    except Exception as e:
                        logger.error(f"Bin file {bf} is corrupted: {e}")
                        if self.repair:
                            # Bin ファイル単位のロールバック試行
                            history = self.cm.hf.get_file_history(bf)
                            for commit_hash in history[1:]:
                                try:
                                    temp_p = hf_hub_download(
                                        repo_id=self.hf_repo,
                                        filename=bf,
                                        repo_type="dataset",
                                        token=self.hf_token,
                                        revision=commit_hash,
                                    )
                                    df = pd.read_parquet(temp_p)
                                    if not df.empty:
                                        logger.info(f"Rescued {bf} from {commit_hash[:7]}")
                                        self.cm.hf.add_commit_operation(bf, Path(temp_p))
                                        break
                                except Exception:
                                    continue

                            if df is None:
                                logger.warning(f"Repair: Target-reducing regeneration for corrupted {bf}")

                        # 読めない場合（破損 ＆ ロールバック失敗）は、対象Binの書類だけをピンポイントで救済
                        if df is None:
                            if expected_docs:
                                self._report_anomaly(
                                    "Layer3_Analytical", f"Corrupted bin {bf}. {len(expected_docs)} docs missing."
                                )
                                if self.repair:
                                    # O(1) 消去法：対象Binの書類のみ pending にリセットする
                                    self.cm.catalog_df.loc[
                                        self.cm.catalog_df["doc_id"].isin(expected_docs), "processed_status"
                                    ] = "pending"
                                    logger.info(
                                        f"Catalog reset staged for {len(expected_docs)} docs "
                                        f"strictly in {expected_bin}."
                                    )
                            continue

                    modified = False

                    if "doc_id" in df.columns and not df.empty:
                        actual_docs = set(df["doc_id"].dropna().unique())
                    else:
                        actual_docs = set()

                    # 1. 重複チェック (ID + 値の完全一致のみ排除)
                    if "key" in df.columns and not df.empty:
                        dup_keys = ["doc_id", "key", "context_ref", "value"]
                        actual_keys = [k for k in dup_keys if k in df.columns]

                        dups = df[df.duplicated(subset=actual_keys, keep=False)]
                        if not dups.empty:
                            self._report_anomaly("Layer3_Analytical", f"Perfect duplicates in {bf}", details=len(dups))
                            if self.repair:
                                df = df.drop_duplicates(subset=actual_keys, keep="first")
                                self.repairs["Layer3_Analytical"].append(
                                    {"bin": bf, "action": "drop_duplicates", "count": len(dups)}
                                )
                                modified = True

                    # 2. 孤児・アサインメント不一致 (Binに存在するが、カタログ上でこのBinに属すべきではない書類)
                    unrecognized_docs = actual_docs - expected_docs
                    if unrecognized_docs:
                        self._report_anomaly(
                            "Layer3_Analytical", f"Unrecognized/Orphan docs in {bf}", details=len(unrecognized_docs)
                        )
                        if self.repair:
                            logger.info(f"Purging {len(unrecognized_docs)} unrecognized records from {bf}.")
                            df = df[~df["doc_id"].isin(unrecognized_docs)]
                            for d_id in unrecognized_docs:
                                self.repairs["Layer3_Analytical"].append(
                                    {"bin": bf, "doc_id": d_id, "action": "purge_orphan"}
                                )
                            modified = True

                    # 3. 欠落データチェック (カタログ上はこのBinに属すべきだが、健康なBin内に存在しない書類)
                    missing_in_bin = expected_docs - actual_docs
                    if missing_in_bin:
                        self._report_anomaly(
                            "Layer3_Analytical", f"Missing docs in healthy {bf}", details=len(missing_in_bin)
                        )
                        if self.repair:
                            from data_engine.executors.backfill_manager import get_dynamic_limit_date

                            limit_date = get_dynamic_limit_date()

                            for doc_id in missing_in_bin:
                                submit_at_str = str(
                                    self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, "submit_at"].iloc[0]
                                )
                                if not submit_at_str or submit_at_str == "nan":
                                    continue
                                submit_date = datetime.strptime(submit_at_str.split(" ")[0], "%Y-%m-%d").date()

                                if submit_date >= limit_date:
                                    self.cm.catalog_df.loc[
                                        self.cm.catalog_df["doc_id"] == doc_id, "processed_status"
                                    ] = "pending"
                                else:
                                    # Bin欠損＆再取得不可の完全なロスト状態
                                    self.cm.catalog_df.loc[
                                        self.cm.catalog_df["doc_id"] == doc_id, "processed_status"
                                    ] = "unrecoverable"

                            logger.info(
                                f"Catalog reset staged strictly for {len(missing_in_bin)} "
                                f"missing docs in {expected_bin}."
                            )

                    if self.repair and modified:
                        self.cm.hf.save_and_upload(bf, df, defer=True)

                except Exception as e:
                    self._report_anomaly("Layer3_Analytical", f"Failed to audit bin {bf}: {e}")

            # 4. 物理ファイル自体が存在しない Bin への究極のフェイルセーフ (Expected but completely missing bin file)
            # `financial_values` などが完全に消滅した場合、ファイルリストに挙がらないため、カタログ主導で検知する
            for e_bin, e_docs in expected_docs_per_bin.items():
                if e_bin == "No" or not e_docs:
                    continue
                if e_bin not in processed_bins:
                    self._report_anomaly(
                        "Layer3_Analytical", f"Completely missing Bin file for {e_bin}. {len(e_docs)} docs lost."
                    )
                    if self.repair:
                        self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"].isin(e_docs), "processed_status"] = (
                            "pending"
                        )
                        logger.info(
                            f"Staged {len(e_docs)} docs for regeneration due to completely missing bin {e_bin}."
                        )

        except Exception as e:
            self._report_anomaly("Layer3_Analytical", f"Analytical reconciliation failed: {e}")

    # ==========================================
    # Layer 4: API Catalog Reconciliation
    # ==========================================
    def reconcile_api_catalog(self, days_to_check: int = 30):
        """直近N日間のAPI応答とカタログを照合"""
        logger.info(f"--- [Layer 4] API Catalog Reconciliation (Last {days_to_check} days) ---")

        catalog_df = self.cm.catalog_df
        if catalog_df.empty:
            logger.info("Catalog is empty.")
            return

        api_key = os.getenv("EDINET_API_KEY")
        if not api_key:
            logger.error("EDINET_API_KEY environment variable is missing for Layer 4.")
            return

        from data_engine.engines.edinet_engine import EdinetEngine

        edinet = EdinetEngine(api_key, self.data_path)

        start_date = (datetime.now() - timedelta(days=days_to_check)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            # APIから最新メタデータを取得
            api_meta = edinet.fetch_metadata(start_date, end_date)
            if not api_meta:
                logger.info("No API metadata found for comparison.")
                return

            api_dict = {row["docID"]: row for row in api_meta}
            df_subset = catalog_df[catalog_df["doc_id"].isin(api_dict.keys())]

            mismatches = []
            # 照合項目：API由来であって、ARIAが内部で生成・加工しない「生の属性」
            check_fields = [
                ("submit_at", "submitDateTime"),
                ("doc_type", "docTypeCode"),
                ("ordinance_code", "ordinanceCode"),
                ("form_code", "formCode"),
                ("withdrawal_status", "withdrawalStatus"),
            ]

            for _, row in df_subset.iterrows():
                doc_id = row["doc_id"]
                api_row = api_dict[doc_id]

                for local_f, api_f in check_fields:
                    local_val = str(row.get(local_f) or "").strip()
                    api_val = str(api_row.get(api_f) or "").strip()

                    if local_val != api_val:
                        # withdrawalStatus は None と "0" を等価扱いする
                        if local_f == "withdrawal_status" and (
                            local_val in ["", "0", "None"] and api_val in ["", "0", "None"]
                        ):
                            continue

                        mismatches.append(
                            {"doc_id": doc_id, "field": local_f, "local_value": local_val, "api_value": api_val}
                        )

            if mismatches:
                self._report_anomaly(
                    "Layer4_Catalog",
                    f"Detected {len(mismatches)} metadata drift(s) against FSA EDINET API.",
                    details=mismatches,
                )
                if self.repair:
                    logger.info("Synchronizing Catalog metadata with FSA fact...")
                    for m in mismatches:
                        doc_id = m["doc_id"]
                        field = m["field"]
                        api_val = m["api_value"]
                        self.cm.catalog_df.loc[self.cm.catalog_df["doc_id"] == doc_id, field] = api_val
                    logger.info("Metadata drift repair staged in RAM.")
            else:
                logger.info("✅ Catalog is perfectly synchronized with EDINET API.")

        except Exception as e:
            self._report_anomaly("Layer4_Catalog", f"API reconciliation failed: {e}")

    def reconcile_indexing(self):
        """[Layer 5] 指数履歴の不変性と整合性を検証"""
        logger.info("--- [Layer 5] Indexing Reconciliation ---")
        try:
            from data_engine.core.models import ARIA_SCHEMAS

            # モデル定義に準拠した正しいキー名 ("indices") を使用
            key = "indices"
            df = None
            try:
                df = self.cm.hf.load_parquet(key)
            except Exception as e:
                logger.error(f"Index History is corrupted: {e}")
                if self.repair:
                    df = self._attempt_file_rollback(key)

            if df is not None:
                # 最小限のカラム検証
                schema = ARIA_SCHEMAS.get(key)
                if schema and set(df.columns) != set(schema.names):
                    self._report_anomaly("Layer5_Indexing", "Schema drift in history.parquet")
                    if self.repair:
                        self.cm.hf.save_and_upload(key, df, defer=True)
                else:
                    logger.info("✅ Index History is healthy.")
        except Exception as e:
            self._report_anomaly("Layer5_Indexing", f"Indexing check failed: {e}")

    def run_full_audit(self) -> Dict[str, Any]:
        """全レイヤーの監査を実行し結果レポートを生成"""
        logger.info("Starting ARIA Data Foundation Audit...")
        if self.repair:
            logger.info("REPAIR MODE: ACTIVE (Self-healing enabled)")

        # 工学的に正しい順序: API事実の確定 -> 目録の修復 -> 物理的照合 -> 分析整合性 -> 派生データ
        self.reconcile_api_catalog(days_to_check=30)
        self.reconcile_schemas()
        self.reconcile_physical_assets()
        self.reconcile_analytical_data()
        self.reconcile_indexing()

        if self.repair:
            # カタログ更新の反映
            self.cm.hf.save_and_upload("catalog", self.cm.catalog_df, clean_fn=self.cm._clean_dataframe, defer=True)
            # 大規模なコミットの送信
            if self.cm.hf.has_pending_operations:
                logger.info("Pushing self-healing repair commit to HF Hub...")
                self.cm.hf.push_commit("Self-healing repair by ARIA Integrity Audit")

        total_anomalies = sum(len(v) for v in self.anomalies.values())

        # 【工学的主権】より直感的な統計のため、重複を排除した「修正対象数」を算出
        unique_repairs = set()
        for layer_repairs in self.repairs.values():
            for r in layer_repairs:
                # 修正対象（doc_id または bin）を識別子としてカウント
                identifier = r.get("doc_id") or r.get("bin") or r.get("key")
                if identifier:
                    unique_repairs.add(identifier)

        total_repairs = len(unique_repairs) if unique_repairs else 0
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "REPAIRED"
            if self.repair and total_anomalies > 0
            else ("FAILED" if total_anomalies > 0 else "PASSED"),
            "total_anomalies": total_anomalies,
            "total_repairs": total_repairs,
            "details": self.anomalies,
            "repairs": self.repairs if self.repair else {},
        }

        if total_anomalies > 0:
            logger.error(f"Integrity Audit detected {total_anomalies} anomalies. See report for details.")
        else:
            logger.info("INTEGRITY AUDIT PASSED: Zero anomalies detected in the foundation.")

        return report


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()

    # Loguru は標準で詳細な出力を提供するため、logging.basicConfig は不要

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-report", action="store_true", help="Save the reconciliation report to JSON")
    parser.add_argument("--repair", action="store_true", help="Enable self-healing repair mode")
    args = parser.parse_args()

    hf_repo = os.getenv("HF_REPO")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_repo or not hf_token:
        logger.critical("HF_REPO and HF_TOKEN are required.")
        sys.exit(1)

    engine = DataReconciliationEngine(hf_repo, hf_token, Path("data"), repair=args.repair)
    report = engine.run_full_audit()

    if args.save_report:
        report_path = Path("logs/reconciliation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {report_path}")

    # GitHub Actions で異常を検知させるための Exit Code (Fail Fast)
    if report["status"] == "FAILED":
        sys.exit(2)


if __name__ == "__main__":
    main()
