"""
Catalog Manager (Facade) — ARIA のデータレイクハウス状態管理の中核。
I/O処理は HfStorage、デルタ管理は DeltaManager、複雑な名寄せは ReconciliationEngine に委譲し、
自身は状態（DF）の保持とオーケストレーションに専念する。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from data_engine.core.config import CONFIG
from data_engine.core.models import EdinetCodeRecord
from data_engine.engines.edinet_engine import EdinetEngine
from data_engine.engines.fsa_engine import FsaEngine
from data_engine.engines.market_engine import MarketDataEngine
from data_engine.engines.master_merger import MasterMerger
from data_engine.engines.reconciliation_engine import ReconciliationEngine
from data_engine.storage.delta_manager import DeltaManager
from data_engine.storage.hf_storage import HfStorage


class CatalogManager:
    def __init__(
        self,
        hf_repo: str = None,
        hf_token: str = None,
        data_path: Path = None,
        scope: str = None,
        edinet: bool = True,
        sync_master: bool = False,
        force_refresh: bool = False,
    ):
        # 1. SSHT (Single Source of Truth) からの読み込みとオーバーライド
        self.hf_repo = hf_repo or CONFIG.HF_REPO
        self.hf_token = hf_token or CONFIG.HF_TOKEN
        self.data_path = data_path or CONFIG.DATA_PATH
        self.scope = (scope or CONFIG.ARIA_SCOPE).capitalize()

        # 0. 環境変数のバリデーション (Fail-Fast / Warning)
        CONFIG.validate_env(production=True, edinet=edinet)

        if not self.hf_repo or not self.hf_token:
            logger.warning("HF_REPO または HF_TOKEN が設定されていません。Hugging Face 操作はスキップされます。")

        # 2. 物理パス定義 (SSOT)
        paths = {
            "catalog": "catalog/documents_index.parquet",
            "master": "meta/stocks_master.parquet",
            "listing": "meta/listing_history.parquet",
            "name": "meta/name_history.parquet",
            "jpx_definitions": "meta/jpx_definitions.parquet",
        }

        # 3. Foundation Layer (Storage & Merger)
        self.hf = HfStorage(self.hf_repo, self.hf_token, self.data_path, paths)
        self.delta = DeltaManager(self.hf, self.data_path, paths, clean_fn=self._clean_dataframe)
        self.merger = MasterMerger(self.hf_repo, self.hf_token, self.data_path)

        # 4. Logic Layer (Engines)
        self.reconciliation = ReconciliationEngine(self)
        if edinet:
            self.edinet = EdinetEngine(
                api_key=CONFIG.EDINET_API_KEY,
                data_path=self.data_path,
                taxonomy_urls=CONFIG.TAXONOMY_URLS,
            )
        else:
            self.edinet = None
            logger.info("EdinetEngine はスキップされました (Market-only mode)。")

        self.fsa = FsaEngine()
        self.market = MarketDataEngine(self.data_path)

        # 5. Runtime State
        self._snapshots = {}
        self.edinet_codes = {}
        self.aggregation_map = {}

        # 6. Data Load (Lazy load も検討可能だが、現状は整合性維持のため即時ロード)
        self.catalog_df = self.hf.load_parquet("catalog", clean_fn=self._clean_dataframe, force_download=force_refresh)
        self.master_df = self.hf.load_parquet("master", clean_fn=self._clean_dataframe, force_download=force_refresh)

        # 7. Optimized Lookups (O(1))
        # doc_id -> status の高速な引き当て用キャッシュ
        self._status_cache: Dict[str, str] = {}
        self._processed_set: set[str] = set()
        self._rebuild_lookup_caches()

        logger.debug(f"CatalogManager Initialized (Scope: {self.scope}, SyncMaster: {sync_master})")

        if sync_master:
            # 整合性チェックと最新スキーマへのアップグレード
            self._retrospective_cleanse()

            # 起動時に各ソースからマスタを同期
            self.edinet_codes, self.aggregation_map = self.sync_edinet_code_lists()
            
            # JPX 銘柄一覧（data_j.xls）の同期
            try:
                jpx_master = self.market.fetch_jpx_master()
            except Exception as e:
                logger.error(f"JPXマスタの取得に失敗しました: {e}")
                jpx_master = pd.DataFrame()

            # Unified Master Sync
            self.reconciliation.sync_master_from_sources(
                edinet_codes=self.edinet_codes,
                aggregation_map=self.aggregation_map,
                jpx_master=jpx_master
            )
            
            if self.hf.has_pending_operations:
                logger.info("マスタの更新（Unified Sync）を検知しました。直ちに Hugging Face に保存します。")
                self.hf.push_commit("Unified Master Sync from multiple sources")
        else:
            logger.debug("マスタ同期をスキップしました (sync_master=False)。HF上の既存データを使用します。")

    # ──────────────────────────────────────────────
    # 委譲 (Delegations)
    # ──────────────────────────────────────────────
    def save_delta(
        self,
        key: str,
        df: pd.DataFrame,
        run_id: str,
        chunk_id: str,
        custom_filename: str = None,
        defer: bool = False,
        local_only: bool = False,
    ) -> bool:
        return self.delta.save_delta(key, df, run_id, chunk_id, custom_filename, defer, local_only)

    def mark_chunk_success(self, run_id: str, chunk_id: str, defer: bool = False, local_only: bool = False) -> bool:
        return self.delta.mark_chunk_success(run_id, chunk_id, defer, local_only)

    def load_deltas(self, run_id: str) -> Dict[str, pd.DataFrame]:
        return self.delta.load_deltas(run_id)

    def cleanup_deltas(self, run_id: str, cleanup_old: bool = True):
        self.delta.cleanup_deltas(run_id, cleanup_old)

    def push_commit(self, message: str = "Batch update from ARIA") -> bool:
        return self.hf.push_commit(message)

    def add_commit_operation(self, repo_path: str, local_path):
        """コミットバッファに操作を追加（HfStorage に委譲）"""
        return self.hf.add_commit_operation(repo_path, local_path)

    def get_sector(self, code: str) -> str:
        if self.master_df.empty:
            logger.info("マスタファイルが空です。初期構築を行います。")
            self.edinet_codes, self.aggregation_map = self.sync_edinet_code_lists()
            if self.edinet_codes:
                self.reconciliation.sync_master_from_edinet_codes()
                self.hf.push_commit("Initial Master Build from EDINET")

        # 1. 完全一致で検索 (JP:12340 等)
        row = self.master_df[self.master_df["code"] == code]
        
        # 2. 完全一致で見つからず、プレフィックスがある場合は剥がして再試行 (Legacy対応)
        if row.empty and ":" in str(code):
            core_code = str(code).split(":", 1)[1]
            row = self.master_df[self.master_df["code"] == core_code]
            
        if not row.empty:
            col_name = "sector_jpx_33" if "sector_jpx_33" in self.master_df.columns else "sector"
            val = row.iloc[0].get(col_name)
            return str(val) if val is not None else None
        return None

    def update_stocks_master(self, incoming_data: pd.DataFrame):
        return self.reconciliation.update_stocks_master(incoming_data)

    def sync_edinet_code_lists(self) -> Tuple[Dict[str, EdinetCodeRecord], Dict[str, str]]:
        """最新の EDINET コードリストと集約一覧を同期し、マッピングオブジェクトを返す"""
        return self.fsa.sync_edinet_code_lists()

    # ──────────────────────────────────────────────
    # State Management & Validation
    # ──────────────────────────────────────────────
    def _clean_dataframe(self, key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        try:
            if key == "catalog":
                from data_engine.core.models import CatalogRecord

                df["is_amendment"] = df["is_amendment"].astype(bool) if "is_amendment" in df.columns else False
                records = []
                for _, row in df.iterrows():
                    d = {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
                    records.append(CatalogRecord(**d).model_dump())
                return pd.DataFrame(records)

            elif key == "master":
                from data_engine.core.models import StockMasterRecord

                records = []
                for _, row in df.iterrows():
                    d = {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
                    records.append(StockMasterRecord(**d).model_dump())
                return pd.DataFrame(records)
            else:
                # 未知のキー（financial_values 等）はバリデーションせずそのまま返す
                return df
        except Exception as e:
            logger.warning(f"データクレンジングエラー ({key}): {e} - フォールバックとして元のDFを返します。")

        return df

    def _retrospective_cleanse(self):
        logger.info("データ構造の健全性確認を開始します (Retrospective Cleanse)...")
        updates_needed = False

        try:
            old_catalog_len = len(self.catalog_df)
            new_catalog = self._clean_dataframe("catalog", self.catalog_df.copy())
            if not new_catalog.equals(self.catalog_df):
                logger.warning(f"CatalogSchemaの不一致を検知。自動修正します。({old_catalog_len}件)")
                self.catalog_df = new_catalog
                self.hf.save_and_upload("catalog", self.catalog_df, defer=True)
                updates_needed = True

            old_master_len = len(self.master_df)
            new_master = self._clean_dataframe("master", self.master_df.copy())
            if not new_master.equals(self.master_df):
                logger.warning(f"StockMasterSchemaの不一致を検知。自動修正します。({old_master_len}件)")
                self.master_df = new_master
                self.hf.save_and_upload("master", self.master_df, defer=True)
                updates_needed = True

            if updates_needed:
                logger.success("✅ データ構造の自動修正予約が完了しました。")

        except Exception as e:
            logger.error(f"Retrospective Cleanse に失敗しました: {e}")

    def take_snapshot(self):
        self._snapshots = {
            "catalog": self.catalog_df.copy(),
            "master": self.master_df.copy(),
            "listing": self.hf.load_parquet("listing").copy(),
            "name": self.hf.load_parquet("name").copy(),
        }
        logger.info("Global 状態のスナップショットを取得しました (安全性確保)")

    def rollback(self, message: str = "RaW-V Failure: Automated Recovery Rollback"):
        if not self._snapshots:
            logger.error("❌ スナップショットが存在しないため、ロールバックできません。")
            return False

        logger.warning(f"⛔ ロールバックを開始します: {message}")
        self.hf.clear_operations()

        for key, df in self._snapshots.items():
            self.hf.save_and_upload(key, df, clean_fn=self._clean_dataframe, defer=True)

        success = self.hf.push_commit(f"ROLLBACK: {message}")
        if success:
            logger.success("✅ ロールバック・コミットが完了しました。整合性は復旧されました。")
            self.catalog_df = self._snapshots["catalog"]
            self.master_df = self._snapshots["master"]
        else:
            logger.critical("❌ ロールバック自体に失敗しました！")
        return success

    # ──────────────────────────────────────────────
    # Catalog / History Methods
    # ──────────────────────────────────────────────
    def _rebuild_lookup_caches(self):
        """カタログDFから高速ルックアップ用のキャッシュを再構築する"""
        if self.catalog_df.empty:
            self._status_cache = {}
            self._processed_set = set()
            return

        # カタログ内の全レコードのステータスを辞書化
        self._status_cache = dict(zip(self.catalog_df["doc_id"], self.catalog_df["processed_status"]))
        # 成功・取下げ済み・アノマリ（空書類等）のIDをSet化 (O(1)検索用)
        self._processed_set = {
            doc_id for doc_id, status in self._status_cache.items() 
            if status in ("success", "retracted", "english_empty", "attachment_empty")
        }

    def is_processed(self, doc_id: str) -> bool:
        """
        書類が「処理完了」かどうかを判定する (O(1) ルックアップ)。
        【工学的主権】success / retracted のみを処理済みとし、
        failure / pending は「再処理対象」として残す（自己修復の起点）。
        """
        return doc_id in self._processed_set

    def get_status(self, doc_id: str) -> str:
        """書類の現在の処理ステータスを取得する (O(1) ルックアップ)。"""
        return self._status_cache.get(doc_id, "unknown")

    def update_catalog(self, new_records: List[Dict]):
        if not new_records:
            return

        df_new = pd.DataFrame(new_records)
        df_new = self._clean_dataframe("catalog", df_new)

        if self.catalog_df.empty:
            self.catalog_df = df_new
        else:
            self.catalog_df = pd.concat([self.catalog_df, df_new], ignore_index=True)
        self.catalog_df.drop_duplicates(subset=["doc_id"], keep="last", inplace=True)

        self._rebuild_lookup_caches()  # キャッシュ更新
        self.hf.save_and_upload("catalog", self.catalog_df, clean_fn=self._clean_dataframe, defer=True)
        logger.info(f"カタログを更新・コミットバッファに追加しました (全 {len(self.catalog_df)} 件)")

        # 【工学的主権】マスタの last_submitted_at を自動更新する (完全同期)
        if not self.master_df.empty and "submit_at" in df_new.columns:
            master_updated = False
            # 今回登録された各コードの最新提出日時
            latest_submits = df_new.dropna(subset=["code", "submit_at"]).groupby("code")["submit_at"].max().to_dict()

            # プレフィックス耐性のあるルックアップ用辞書の構築
            # edinet_code を主キーとしつつ、code (新旧双方) でも引けるようにする
            master_dict = {}
            for _, row in self.master_df.iterrows():
                m_rec = row.to_dict()
                e_code = m_rec.get("edinet_code")
                m_code = m_rec.get("code")
                
                if e_code:
                    master_dict[e_code] = m_rec
                
                if pd.notna(m_code) and str(m_code).strip():
                    m_code_str = str(m_code)
                    master_dict[m_code_str] = m_rec
                    if ":" not in m_code_str:
                        master_dict[f"JP:{m_code_str}"] = m_rec
                    else:
                        core = m_code_str.split(":", 1)[1]
                        if core not in master_dict:
                            master_dict[core] = m_rec

            for code, submit_time in latest_submits.items():
                if not code or pd.isna(code) or str(code).strip() == "":
                    continue
                if code in master_dict:
                    m_rec = master_dict[code]
                    current_last = m_rec.get("last_submitted_at")
                    # 辞書順比較による最新判定 (フォーマットが整っている前提)
                    if not current_last or str(submit_time) > str(current_last):
                        m_rec["last_submitted_at"] = str(submit_time)
                        master_updated = True

            if master_updated:
                # 相互リンク (JP:12340 と 12340) による重複を排除して DataFrame化
                unique_records = []
                seen_ids = set()
                for rec in master_dict.values():
                    if id(rec) not in seen_ids:
                        unique_records.append(rec)
                        seen_ids.add(id(rec))
                
                new_master_df = pd.DataFrame(unique_records)
                self.master_df = self._clean_dataframe("master", new_master_df)
                self.hf.save_and_upload("master", self.master_df, clean_fn=self._clean_dataframe, defer=True)
                logger.info("書類の提出を検知し、マスタの last_submitted_at を更新しました。")

        # 【工学的主権】更新された銘柄の社名変更履歴を再構成
        unique_codes = df_new["code"].unique()
        for code in unique_codes:
            if not code or pd.isna(code):
                continue
            history_df = self.reconciliation.reconstruct_name_history(code)
            if not history_df.empty:
                self.update_name_history(history_df)

    def update_listing_history(self, new_events: pd.DataFrame):
        hist_df = self.hf.load_parquet("listing")
        m_df = pd.concat([hist_df, new_events], ignore_index=True)
        m_df.drop_duplicates(subset=["code", "type", "event_date"], keep="last", inplace=True)
        m_df.sort_values(["event_date", "code"], ascending=[False, True], inplace=False)
        self.hf.save_and_upload("listing", m_df, defer=True)

    def update_name_history(self, new_events: pd.DataFrame):
        """社名変更履歴を更新"""
        hist_df = self.hf.load_parquet("name")
        m_df = pd.concat([hist_df, new_events], ignore_index=True)

        # 漢字名ベースの重複排除
        m_df.drop_duplicates(subset=["code", "old_name", "new_name", "change_date"], keep="last", inplace=True)

        m_df.sort_values(["change_date", "code"], ascending=[False, True], inplace=True)
        self.hf.save_and_upload("name", m_df, defer=True)

    def get_listing_history(self) -> pd.DataFrame:
        return self.hf.load_parquet("listing")

    def get_name_history(self) -> pd.DataFrame:
        return self.hf.load_parquet("name")

