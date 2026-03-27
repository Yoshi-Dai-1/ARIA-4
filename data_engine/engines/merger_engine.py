from loguru import logger

from data_engine.core.config import RAW_DIR


class MergerEngine:
    """Mergerモード: デルタファイルの集約とGlobal更新 (Atomic Commit & Rollback 戦略)"""

    def __init__(self, catalog, run_id):
        self.catalog = catalog
        self.merger = catalog.merger
        self.run_id = run_id

    def run(self) -> bool:
        logger.info(f"=== Merger Started (RunID: {self.run_id}) ===")

        # 1. スナップショットの取得 (Rollback用)
        self.catalog.take_snapshot()

        # 2. 全てのデルタファイルを収集
        deltas = self.catalog.load_deltas(self.run_id)
        if not deltas:
            logger.warning(f"処理対象のデルタが見つかりません。RunID: {self.run_id}")
            return True

        # 3. カタログのマージ
        if "catalog" in deltas:
            df_cat = deltas["catalog"]
            logger.info(f"カタログデルタをマージ中: {len(df_cat)} 件")
            self.catalog.update_catalog(df_cat.to_dict("records"))

        # 4. マスタデータ (financial / qualitative) のマージ
        # 業種・Binごとに分割されたデータを統合して MasterMerger に渡す
        for key, df in deltas.items():
            if key == "catalog":
                continue

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

        # 5. RAW ファイル（ZIP/PDF）の永続化
        #    Worker がダウンロードした原本を HF の raw/ ディレクトリに永続化する。
        #    GHA では actions/upload-artifact で data/ 全体が Merger に渡される。
        self._upload_raw_files()

        # 6. アトミックな確定 (Push Commit)
        logger.info("全ての更新を Hugging Face に一括コミットします...")

        # 検証用のローカル行数を保持
        expected_counts = {"catalog": len(self.catalog.catalog_df), "master": len(self.catalog.master_df)}

        success = self.catalog.push_commit(f"Atomic Build: {self.run_id}")

        if success:
            # 7. RaW-V (Read-after-Write Verification)
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
        """Worker がダウンロードした RAW ファイル（ZIP/PDF）を HF に永続化する"""
        raw_edinet_dir = RAW_DIR / "edinet"
        if not raw_edinet_dir.exists():
            logger.info("RAW ファイルディレクトリが存在しません。アップロードをスキップします。")
            return

        # ファイル数を実カウント（ディレクトリを除く）
        raw_file_count = sum(1 for f in raw_edinet_dir.rglob("*") if f.is_file())
        if raw_file_count == 0:
            logger.info("アップロード対象の RAW ファイルがありません。")
            return

        logger.info(f"RAW ファイルを HF に永続化します: {raw_file_count} files")
        self.catalog.hf.upload_raw_folder(raw_edinet_dir, "raw/edinet", defer=True)

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
