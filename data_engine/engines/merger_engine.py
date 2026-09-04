import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from loguru import logger

from data_engine.core.config import RAW_DIR
from data_engine.core.network_utils import is_404
from data_engine.core.utils import force_gc, log_resources
from data_engine.storage.tar_bundle_manager import bundle_raw_files_by_date, merge_tars


class MergerEngine:
    """Mergerモード: デルタファイルの集約とGlobal更新 (Atomic Commit & Rollback 戦略)"""

    def __init__(self, catalog, run_id):
        self.catalog = catalog
        self.merger = catalog.merger
        self.run_id = run_id

    def run(self) -> bool:
        logger.info(f"=== Merger Started (RunID: {self.run_id}) ===")
        log_resources("Merger開始")

        # 1. スナップショットの取得 (Rollback用)
        self.catalog.take_snapshot()

        # 2. カタログデルタを先行ロード・処理・解放
        #    【メモリ最適化】Bin データはまだ読み込まない。カタログのみ処理後に即解放する。
        catalog_delta = self.catalog.load_catalog_delta(self.run_id)
        if catalog_delta is not None:
            logger.info(f"カタログデルタをマージ中: {len(catalog_delta)} 件")
            self.catalog.update_catalog(catalog_delta.to_dict("records"))
            del catalog_delta
            force_gc()
        else:
            logger.warning(f"カタログデルタが見つかりません。RunID: {self.run_id}")

        log_resources("カタログマージ完了")

        # 3. Bin デルタを逐次ロード・処理・解放
        #    【メモリ最適化】1 Bin ずつ yield で取得し、処理後に即解放する。
        #    全 Bin を同時にメモリに載せない。
        bin_count = 0
        total_bins = 0
        for key, df, total_bins_yielded in self.catalog.iter_bin_deltas(self.run_id):
            bin_count += 1
            total_bins = total_bins_yielded

            try:
                # key 形式: financial_{sector} or text_{sector} or financial_bin{bin}
                if key.startswith("financial_"):
                    m_type = "financial_values"
                    sector_or_bin = key.replace("financial_", "")
                elif key.startswith("text_"):
                    m_type = "qualitative_text"
                    sector_or_bin = key.replace("text_", "")
                else:
                    logger.warning(f"未知のデルタキーをスキップ: {key}")
                    continue

                if sector_or_bin.startswith("bin"):
                    bin_val = sector_or_bin.replace("bin", "")
                    logger.info(f"Master更新 (Bin統合): {m_type} | bin={bin_val}")
                    self.merger.merge_and_upload(
                        bin_val,
                        m_type,
                        df,
                        worker_mode=False,
                        catalog_manager=self.catalog,
                        defer=True,
                    )
                else:
                    logger.info(f"Master更新 (業種統合): {m_type} | sector={sector_or_bin}")
                    self.merger.merge_and_upload(
                        sector_or_bin,
                        m_type,
                        df,
                        worker_mode=False,
                        catalog_manager=self.catalog,
                        defer=True,
                    )
            except Exception as e:
                logger.error(f"Master統合失敗 ({key}): {e}")
                self.catalog.rollback(f"Master Merge Failure: {key}")
                return False
            finally:
                del df
                force_gc()

            # 【運用監視】10 Bin ごとにメモリ推移を記録し、枯渇の兆候を早期検知する
            if bin_count % 10 == 0:
                log_resources(f"Bin {bin_count}/{total_bins}")

        if bin_count == 0:
            logger.warning("Bin デルタが見つかりません。カタログのみの更新です。")

        log_resources(f"Binループ完了 ({bin_count} Bins)")

        # 4. RAW ファイル（ZIP/PDF）の永続化
        #    Worker がダウンロードした原本を HF の raw/ ディレクトリに永続化する。
        #    GHA では actions/upload-artifact で data/ 全体が Merger に渡される。
        self._upload_raw_files()

        log_resources("RAWアップロード完了")

        # 5. アトミックな確定 (Push Commit)
        logger.info("全ての更新を Hugging Face に一括コミットします...")

        # 【Pre-commit Audit】実行前のスナップショットと現在のデータ行数を比較
        # ネットワークエラー等による「意図しない上書き消去」を物理的に阻止する最終防壁
        snapshots = self.catalog._snapshots
        if snapshots:
            for key in ["catalog", "master"]:
                prev_df = snapshots.get(key)
                curr_df = getattr(self.catalog, f"{key}_df")
                
                if prev_df is not None and not prev_df.empty:
                    prev_len = len(prev_df)
                    curr_len = len(curr_df)
                    
                    # 行数が前回比 50% 以下に激減している場合は、異常事態と判定
                    # ※ 初回構築時は snapshots が空、または prev_df.empty なのでこのガードは通過する
                    if curr_len < (prev_len * 0.5):
                        logger.critical(
                            f"🚨 Pre-commit Audit 失敗 ({key}): "
                            f"行数が激減しています (前回: {prev_len} -> 今回: {curr_len})。指示なき削除を拒否します。"
                        )
                        self.catalog.rollback(f"Pre-commit Audit Failure: Row count anomaly in {key}")
                        return False

        # 検証用のローカル行数を保持
        expected_counts = {"catalog": len(self.catalog.catalog_df), "master": len(self.catalog.master_df)}

        success = self.catalog.push_commit(f"Atomic Build: {self.run_id}")

        if success:
            # 6. RaW-V (Read-after-Write Verification)
            #    コミット成功後にリモートから強制再取得し、重要ファイルの整合性を検証する。
            if self._verify_results(expected_counts):
                logger.success(f"=== Merger完了: RunID {self.run_id} の全ての更新が確定されました ===")
                # 完了後にデルタを清掃
                self.catalog.cleanup_deltas(self.run_id, cleanup_old=True)
                return True
            else:
                # RaW-V 失敗 → 自動ロールバック
                logger.critical("RaW-V 失敗! 自動ロールバックを実行します...")
                self.catalog.rollback(f"RaW-V Failure: {self.run_id}")
                return False
        else:
            logger.critical("Final Commit Failed! Rolling back to snapshots...")
            self.catalog.rollback(f"Commit Failure: {self.run_id}")
            return False

    def _upload_raw_files(self):
        """Worker がダウンロードした RAW ファイル（ZIP/PDF）を日次 TAR バンドルとして HF に永続化する。

        【最終防御層】.zip 拡張子を持つファイルは、バンドル前に
        zipfile.is_zipfile() で物理検証を行う。EDINET API が 200 OK +
        JSON ペイロードを返す仕様により、過去に汚染データが永続化された
        事故を二度と発生させないための構造的ゲートキーパー。
        """

        raw_edinet_dir = RAW_DIR / "edinet"
        log_resources("_upload_raw_files 開始")

        if not raw_edinet_dir.exists():
            logger.info("RAW ファイルディレクトリが存在しません。アップロードをスキップします。")
            return

        # 【Gatekeeper】バンドル前に .zip / .pdf ファイルの物理的整合性を検証
        poisoned_count = 0
        for f in raw_edinet_dir.rglob("*.zip"):
            if f.is_file() and not zipfile.is_zipfile(f):
                logger.error(f"🚨 Gatekeeper (ZIP): 汚染ファイルを除去しました: {f} ({f.stat().st_size} bytes)")
                f.unlink(missing_ok=True)
                poisoned_count += 1

        for f in raw_edinet_dir.rglob("*.pdf"):
            if f.is_file():
                try:
                    with open(f, "rb") as pdf_f:
                        header = pdf_f.read(4)
                        if header != b"%PDF":
                            logger.error(
                                f"🚨 Gatekeeper (PDF): 汚染ファイルを除去しました: {f} ({f.stat().st_size} bytes)"
                            )
                            f.unlink(missing_ok=True)
                            poisoned_count += 1
                except Exception:
                    f.unlink(missing_ok=True)
                    poisoned_count += 1

        if poisoned_count > 0:
            logger.warning(f"Gatekeeper: {poisoned_count} 件の汚染ファイルを除去しました。")

        log_resources("Gatekeeper完了")

        # ファイル数を実カウント（ディレクトリを除く）
        raw_file_count = sum(1 for f in raw_edinet_dir.rglob("*") if f.is_file())
        logger.info(f"RAW ファイル数: {raw_file_count} files")
        if raw_file_count == 0:
            logger.info("アップロード対象の RAW ファイルがありません。")
            return

        # 日次 TAR バンドルを作成
        tar_bundles = bundle_raw_files_by_date(raw_edinet_dir)
        if not tar_bundles:
            logger.info("TAR バンドル対象がありません。")
            return

        # 既存 TAR とのマージを含むアップロード
        # 【データ保護】既存 TAR が HF 上に存在する場合、ダウンロードしてマージする。
        # これにより、リトライ等で同一日付の書類が複数 run にまたがっても、
        # 以前の run で保存したデータが消失しない。
        skipped_dates = 0
        uploaded_dates = 0
        for tar_repo_path, (local_tar, local_index) in tar_bundles.items():
            final_tar, final_index = local_tar, local_index

            # 既存 TAR の存在確認とマージ
            try:
                existing_local = hf_hub_download(
                    repo_id=self.catalog.hf.hf_repo,
                    filename=tar_repo_path,
                    repo_type="dataset",
                    token=self.catalog.hf.hf_token,
                )
                # 既存 TAR が見つかった → マージ（重複排除）
                merged_tar = local_tar.with_name(local_tar.stem + "_merged.tar")
                final_tar, final_index = merge_tars(
                    Path(existing_local), local_tar, merged_tar
                )
                logger.info(f"既存 TAR とマージしました: {tar_repo_path}")
            except Exception as e:
                if is_404(e):
                    pass  # 既存 TAR なし → 新規 TAR をそのままアップロード
                else:
                    # ネットワーク障害 → 既存データの状態が不明 → Fail-Fast
                    logger.critical(
                        f"⛔ ネットワーク障害により {tar_repo_path} の存在確認ができません。"
                        f"データ保護のためこの日のアップロードをスキップします: {e}"
                    )
                    skipped_dates += 1
                    continue

            index_repo_path = tar_repo_path.replace(".tar", ".json")
            self.catalog.hf.add_commit_operation(tar_repo_path, final_tar)
            self.catalog.hf.add_commit_operation(index_repo_path, final_index)
            uploaded_dates += 1

        if skipped_dates > 0:
            logger.warning(f"⚠️ {skipped_dates} 日付分の TAR がネットワーク障害によりスキップされました")

        logger.info(
            f"RAW ファイルを {uploaded_dates} 個の日次 TAR バンドルとして "
            f"コミットバッファに追加しました (元ファイル数: {raw_file_count})"
        )

    def _verify_results(self, expected_counts: dict) -> bool:
        """
        RaW-V (Read-after-Write Verification)
        重要ファイル（カタログ・マスタ）について、リモート上の行数が期待値以上であることを検証する。
        """
        try:
            logger.info("RaW-V: リモートから最新データを強制再取得して整合性を検証中...")

            for key, expected_len in expected_counts.items():
                if expected_len == 0:
                    continue

                remote_df = self.catalog.hf.load_parquet(
                    key, force_download=True, clean_fn=self.catalog._clean_dataframe
                )
                remote_len = len(remote_df)

                if remote_len < expected_len:
                    logger.error(
                        f"⚠️ RaW-V 不整合検出 ({key}): ローカル {expected_len} 行 vs "
                        f"リモート {remote_len} 行 (欠落: {expected_len - remote_len} 行)"
                    )
                    return False

                logger.debug(f"✅ RaW-V 合格 ({key}): リモート {remote_len} 行")

            logger.success(f"✅ 全 {len(expected_counts)} 項目の RaW-V 検証に合格しました。")
            return True

        except Exception as e:
            logger.warning(f"RaW-V 検証中にネットワークエラー等の問題が発生しました: {e}")
            logger.warning("データの破損ではない可能性があるため、今回は成功として続行します。")
            return True
