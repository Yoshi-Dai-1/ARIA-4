"""
Reconciliation Engine — ARIA の心臓部である名寄せ・生存判定・属性継承・スコープフィルタリングを担当するエンジン。

【工学的主権】リコンシリエーション・ロジックをモジュール化し、アイデンティティ解決とライフサイクル管理を分離。
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from loguru import logger

from data_engine.core.models import StockMasterRecord
from data_engine.core.utils import normalize_code
from data_engine.engines.reconciliation import IdentityResolver, LifecycleManager


class ReconciliationEngine:
    """
    名寄せ・属性解決エンジン (Orchestrator)
    
    ARIA 統治の 5 鉄則に基づき、EDINET（SSOT）と JPX（補完）を統合します。
    """
    
    SPECIAL_MARKET_KEYWORDS = ["ETF", "REIT", "PRO MARKET"]

    def __init__(self, catalog_manager):
        """
        Args:
            catalog_manager (CatalogManager): カタログ管理インスタンス
        """
        self.cm = catalog_manager
        self.resolver = IdentityResolver(catalog_manager)
        self.lifecycle = LifecycleManager(catalog_manager)
        logger.debug("ReconciliationEngine (Modular) を初期化しました。")

    def sync_master_from_sources(
        self, edinet_codes: dict, aggregation_map: dict, jpx_master: pd.DataFrame
    ) -> bool:
        """
        複数ソース（EDINETコードリスト、集約一覧、JPXマスタ）を統合してマスタを更新する。
        """
        # 1. EDINET コードリストを DataFrame 化 (IdentityResolver へ委譲)
        df_edinet = self.resolver.resolve_master_from_edinet(edinet_codes)

        # 2. 集約一覧 (ESE140190.csv) に基づく付け替え
        if aggregation_map:
            logger.info(f"集約一覧に基づきコードの付け替えを適用します: {len(aggregation_map)} 件対象")
            
            # ヘルパー: 逆方向名寄せ (継続コードに対し廃止コードを履歴付与)
            def apply_backward_agg(df, col_name="edinet_code"):
                if df.empty or col_name not in df.columns:
                    return df
                
                # aggregation_map: {old_code: new_code}
                # 逆引きマップ: {new_code: [old_code1, old_code2]} を作成
                from collections import defaultdict
                reverse_map = defaultdict(list)
                for old_c, new_c in aggregation_map.items():
                    reverse_map[new_c].append(old_c)
                    
                target_mask = df[col_name].isin(reverse_map.keys())
                if target_mask.any():
                    def append_former_backward(row):
                        new_c = row[col_name]
                        old_codes = reverse_map.get(new_c, [])
                        if not old_codes:
                            return row.get("former_edinet_codes")
                        
                        existing = row.get("former_edinet_codes")
                        # "None"、"nan"、空文字などのゴミを徹底排除し、カンマで正しく分割
                        if pd.isna(existing) or existing is None or str(existing).strip().lower() in ("", "none", "nan"):
                            existing_list = []
                        else:
                            existing_list = [x.strip() for x in str(existing).split(",") if x.strip()]
                            
                        # 新しい旧コードを履歴に追記 (重複回避・複数保持対応)
                        for oc in old_codes:
                            if oc not in existing_list:
                                existing_list.append(oc)
                                
                        return ",".join(existing_list)
                    
                    df.loc[target_mask, "former_edinet_codes"] = df[target_mask].apply(append_former_backward, axis=1)
                
                return df

            # 既存マスタと新規データの双方に適用
            if not self.cm.master_df.empty:
                self.cm.master_df = apply_backward_agg(self.cm.master_df)
            df_edinet = apply_backward_agg(df_edinet)

        # 3. JPX マスタを統合 (EDINET に存在しない ETF/REIT 等を補完)
        if not jpx_master.empty:
            logger.info(f"JPXマスタとEDINETコードリストを統合します: JPX={len(jpx_master)}件")
            jpx_master = jpx_master.assign(source_jpx=True)
            df_edinet = df_edinet.assign(source_jpx=False)
            incoming_data = pd.concat([df_edinet, jpx_master], ignore_index=True)
        else:
            incoming_data = df_edinet.assign(source_jpx=False)

        return self.update_stocks_master(incoming_data)


    def sync_master_from_edinet_codes(self, results: Optional[Dict[str, Any]] = None) -> bool:
        """後方互換性のためのラップメソッド"""
        if results is None:
            results = self.cm.edinet_codes

        # 銘柄一覧を統合ソースとして渡す (変換は委譲先で行われる)
        return self.sync_master_from_sources(results, {}, pd.DataFrame())

    def update_master_from_edinet_codes(self):
        """CatalogManager からの古い呼び出し形式への後方互換性"""
        return self.sync_master_from_edinet_codes()

    def update_stocks_master(self, incoming_data: pd.DataFrame):
        """
        マスタ更新 & 時系列リコンシリエーション (Sovereign Multi-Source Fusion)
        """
        if incoming_data.empty:
            return True

        def resolve_attr(group, col):
            vals = group[col].dropna()
            return vals.iloc[0] if not vals.empty else None

        # 1. 【Identity Bridging】証券コードと EDINET コードの架け橋
        incoming_data = self.resolver.bridge_fill(incoming_data)

        # 2. 【Disposal Rule】JPX 重複・不要レコードのフィルタリング (Ordinary Stock SSOT)
        incoming_data = self.resolver.apply_disposal_rule(incoming_data)

        # 3. 事前処理: プレフィックス正規化と親子紐付け
        processed_records: List[Dict[str, Any]] = []
        current_codes_in_run = set()
        for _, row in incoming_data.iterrows():
            rec: Dict[str, Any] = row.to_dict()
            rec = {k: v for k, v in rec.items() if not pd.isna(v) and v is not None}
            sec_code = rec.get("code")

            if sec_code:
                sec_code = normalize_code(sec_code, nationality="JP")
                rec["code"] = sec_code
                
                # 親子紐付け (LifecycleManager へ委譲)
                rec = self.lifecycle.setup_parent_code(rec)
            # 既存属性の承継 (キャッシュ活用)
            # 識別子の全量を記録 (消失判定用)
            for id_key in ["jcn", "edinet_code", "code"]:
                val = rec.get(id_key)
                if val:
                    current_codes_in_run.add(val)
            
            # 暫定的な identity_key の生成 (マスタ結合用)
            identity_key = rec.get("edinet_code") or sec_code
            if identity_key:
                rec["identity_key"] = identity_key

            if not self.cm.master_df.empty and sec_code:
                m_row = self.cm.master_df[self.cm.master_df["code"] == sec_code]
                if not m_row.empty:
                    m_rec = m_row.iloc[0].to_dict()
                    for k, v in m_rec.items():
                        if k not in rec or rec[k] is None:
                            rec[k] = v

            try:
                # StockMasterRecord による金型ガード
                master_rec = StockMasterRecord(**rec)
                d = master_rec.model_dump()
                # 統合ロジックに必要な内部フラグ (source_jpx) を維持
                d["source_jpx"] = rec.get("source_jpx", False)
                processed_records.append(d)
            except Exception as e:
                logger.error(f"銘柄バリデーション失敗 (code: {sec_code}): {e}")

        # 4. マージ処理
        incoming_df = pd.DataFrame(processed_records)
        current_m = self.cm.master_df.copy()
        all_states = pd.concat([current_m, incoming_df], ignore_index=True)

        # identity_key による物理的統合 (EDINET-First Multi-Key Identity Resolution)
        # 1. まず EDINET > Code > JCN の優先順位で解決を試みる
        all_states["identity_key"] = all_states["edinet_code"].fillna(all_states["code"]).fillna(all_states["jcn"])

        # 【堅牢化】識別子が一切存在しない不正レコードを物理的に排除 (Null Identity Key ガード)
        invalid_mask = all_states["identity_key"].isna()
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"識別子を持たない不正レコードが {invalid_count} 件検出されました。除外します。")
            all_states = all_states[~invalid_mask].copy()

        if all_states.empty:
            logger.warning("有効な銘柄レコードが存在しません。マスタ更新を中断します。")
            return True

        # 2. 複数の識別子が混在する場合、最も権威ある ID (JCN) に集約して統合
        def resolve_canonical_id(group):
            ids = group.dropna().unique()
            # 1. JCN (13桁の数字文字列) を最優先
            jcn_ids = [str(i) for i in ids if len(str(i)) == 13 and str(i).isdigit()]
            if jcn_ids:
                return jcn_ids[0]
            # 2. EDINETコード (Eから始まる) を次点
            edinet_ids = [str(i) for i in ids if str(i).startswith("E")]
            if edinet_ids:
                return edinet_ids[0]
            # 3. それ以外 (Security Code等)
            return str(ids[0]) if len(ids) > 0 else None

        # 【Transitive Identity Resolution】連鎖的な名寄せ
        # JCN, EDINETコード, 証券コードの連鎖を辿って、最も権威あるIDに収束させる
        # 各物理キーを順次 groupby の対象とし、そのグループ内で最も権威ある ID を全メンバーに波及 (Propagate) させる
        for key_col in ["jcn", "edinet_code", "code"]:
            mask = all_states[key_col].notna()
            if mask.any():
                all_states.loc[mask, "identity_key"] = (
                    all_states[mask].groupby(key_col, dropna=True)["identity_key"].transform(resolve_canonical_id)
                )

        # 最終パス: identity_key 自体でグループ化し、収束を確定させる
        all_states["identity_key"] = all_states.groupby("identity_key", dropna=False)["identity_key"].transform(resolve_canonical_id)
        
        # タイブレーカー用にソースの鮮度フラグを付与 (incoming=1, current=0)
        # これにより、日付が同じ（または無い）場合に新規データを優先する
        all_states["_priority"] = 0
        all_states.loc[all_states.index[len(current_m):], "_priority"] = 1
        jpx_defs: List[Dict[str, str]] = []
        best_records: List[Dict[str, Any]] = []

        # 【要点】dropna=True (デフォルト) にすることで、万一 identity_key が Null の行があっても無視される
        for _, group in all_states.groupby("identity_key", dropna=True):
            # 時系列ソート (最新優先、日付が同じなら新規優先)
            sorted_group = group.sort_values(
                ["last_submitted_at", "_priority"], 
                ascending=[False, False], 
                na_position="last"
            )
            latest_rec: Dict[str, Any] = sorted_group.iloc[0].copy()

            # 属性伝搬 (NULL 埋め)
            for attr in [
                "company_name", "sector_jpx_33", "sector_33_code", "sector_jpx_17",
                "sector_17_code", "market", "size_code", "size_category", "jcn",
                "edinet_code", "parent_code", "former_edinet_codes", "company_name_en",
                "company_name_kana", "submitter_type", "address",
                "industry_edinet", "industry_edinet_en", "capital", "settlement_date",
                "is_consolidated"
            ]:
                val = resolve_attr(sorted_group, attr)
                if val is not None:
                    latest_rec[attr] = val

            # Rule 4: is_active および is_listed_edinet の判定基準
            # is_listed_edinet: EDINET 上場区分が「上場」であるか
            # is_active: JPXに存在すれば基本的にTrueとし全件包摂するが、EDINETが「非上場」と明記している場合のみ、絶対にFalseとする (EDINET第一優先)
            # 【監査事実】StockMasterRecord のバリデーター (convert_to_bool_listed) により、
            # この時点で is_listed_edinet は bool 型 (True/False) に変換済み。
            # True = EDINET「上場」, False = EDINET「非上場」またはデフォルト
            listed_edinet_vals = sorted_group["is_listed_edinet"].dropna().tolist()
            is_listed_edinet = any(v is True for v in listed_edinet_vals)
            # 「非上場」の明示的判定: EDINETソースのレコードが存在し、かつ全てFalseの場合
            has_edinet_source = sorted_group["edinet_code"].notna().any()
            is_explicitly_unlisted_edinet = (
                has_edinet_source
                and len(listed_edinet_vals) > 0
                and not is_listed_edinet
            )
            
            from_jpx = sorted_group.get("source_jpx", pd.Series([False])).any()
            
            latest_rec["is_listed_edinet"] = is_listed_edinet
            
            if is_explicitly_unlisted_edinet:
                # EDINET「非上場」銘柄の場合、JPX保有であっても絶対に is_active=False とする
                latest_rec["is_active"] = False
            else:
                # それ以外（EDINET上場、あるいはEDINET未登録のETF等）は通常通り判定
                latest_rec["is_active"] = is_listed_edinet or from_jpx

            # JPX 定義の収集 (Dimension Table)
            self._collect_jpx_defs(latest_rec, jpx_defs)

            best_records.append(latest_rec)

        # 5. 【Tracking】消失銘柄の判定
        new_master_df = pd.DataFrame(best_records)
        
        # 【物理的整律】StockMasterRecord の定義に従ってカラム順序を固定
        master_cols = list(StockMasterRecord.model_fields.keys())
        # マスタに存在しない内部フラグ等は除外、必要なものだけ並べ替え
        existing_cols = [c for c in master_cols if c in new_master_df.columns]
        # 追加のカラム（内部フラグなど）があれば末尾へ
        remaining_cols = [c for c in new_master_df.columns if c not in master_cols]
        new_master_df = new_master_df[existing_cols + remaining_cols]

        new_master_df = self.lifecycle.track_disappearance(new_master_df, current_codes_in_run)

        # 6. 【Event Detection】上場・廃止イベントの一括検知
        # 消失判定（is_active変更）後の最終的なマスタ状態から変化を抽出
        listing_events: List[Dict[str, Any]] = []
        for _, row in new_master_df.iterrows():
            listing_events.extend(self.lifecycle.detect_listing_events(row.to_dict(), current_m))

        self.cm.master_df = new_master_df

        # 7. 【Redundancy Management】名称の厳格同期
        # マスタ内の名称を jpx_definitions の「正解」で上書きし、不一致を物理的に排除する
        if jpx_defs:
            df_defs = pd.DataFrame(jpx_defs).drop_duplicates(subset=["type", "code"])
            self.cm.master_df = self._resolve_with_definitions(self.cm.master_df, df_defs)

        # メタデータの保存 (Master, Listing History, JPX Definitions)
        self._save_metadata(jpx_defs, listing_events)

        logger.success(f"マスタ統合完了: {len(self.cm.master_df)} 銘柄を解決・更新しました。")

        return self.cm.hf.save_and_upload("master", self.cm.master_df, clean_fn=self.cm._clean_dataframe, defer=True)

    def _resolve_with_definitions(self, master_df: pd.DataFrame, defs_df: pd.DataFrame) -> pd.DataFrame:
        """マスタ内の業種名・規模区分名を定義テーブルの最新状態で強制的に同期する"""
        if defs_df.empty:
            return master_df
            
        # 33業種
        if "sector_33_code" in master_df.columns:
            s33: Dict[str, str] = defs_df[defs_df["type"] == "sector_33"].set_index("code")["name"].to_dict()
            master_df["sector_jpx_33"] = master_df["sector_33_code"].map(s33).fillna(
                master_df.get("sector_jpx_33", pd.Series(dtype="object"))
            )
        
        # 17業種
        if "sector_17_code" in master_df.columns:
            s17: Dict[str, str] = defs_df[defs_df["type"] == "sector_17"].set_index("code")["name"].to_dict()
            master_df["sector_jpx_17"] = (
                master_df["sector_17_code"]
                .map(s17)
                .fillna(master_df.get("sector_jpx_17", pd.Series(dtype="object")))
            )
        
        # 規模区分
        if "size_code" in master_df.columns:
            sz = defs_df[defs_df["type"] == "size"].set_index("code")["name"].to_dict()
            master_df["size_category"] = master_df["size_code"].map(sz).fillna(master_df.get("size_category", pd.Series(dtype="object")))
        
        return master_df

    def _collect_jpx_defs(self, rec, jpx_defs: list):
        """業種・規模名称のマッピング定義を収集（正規化のため）"""
        s33_c = rec.get("sector_33_code")
        if s33_c and not pd.isna(s33_c):
            jpx_defs.append({"type": "sector_33", "code": str(s33_c), "name": rec.get("sector_jpx_33")})
        
        s17_c = rec.get("sector_17_code")
        if s17_c and not pd.isna(s17_c):
            jpx_defs.append({"type": "sector_17", "code": str(s17_c), "name": rec.get("sector_jpx_17")})
        
        sz_c = rec.get("size_code")
        if sz_c and not pd.isna(sz_c):
            jpx_defs.append({"type": "size", "code": str(sz_c), "name": rec.get("size_category")})

    def _save_metadata(self, jpx_defs, listing_events):
        """マスタ付随メタデータの保存"""
        if jpx_defs:
            from datetime import date, timedelta
            today_str = date.today().isoformat()
            yesterday_str = (date.today() - timedelta(days=1)).isoformat()
            
            df_defs = pd.DataFrame(jpx_defs).drop_duplicates(subset=["type", "code"]).dropna(subset=["code"])
            
            if not df_defs.empty:
                try:
                    existing_defs = self.cm.hf.load_parquet("jpx_definitions", force_download=False)
                except Exception:
                    existing_defs = pd.DataFrame()
                
                if existing_defs.empty:
                    df_defs["valid_from"] = today_str
                    df_defs["valid_to"] = None
                    final_defs = df_defs
                else:
                    # 既存スキーママイグレーション (descriptionの削除とvalid_from/toの追加)
                    if "description" in existing_defs.columns:
                        existing_defs = existing_defs.drop(columns=["description"])
                    if "valid_from" not in existing_defs.columns:
                        existing_defs["valid_from"] = "2000-01-01"
                    if "valid_to" not in existing_defs.columns:
                        existing_defs["valid_to"] = None
                        
                    new_records = []
                    # 既存の最新定義 (valid_to is null) を取得
                    active_existing = existing_defs[existing_defs["valid_to"].isna()]
                    
                    for _, row in df_defs.iterrows():
                        t, c, n = row["type"], row["code"], row["name"]
                        mask = (active_existing["type"] == t) & (active_existing["code"] == c)
                        matched = active_existing[mask]
                        
                        if matched.empty:
                            new_records.append({"type": t, "code": c, "name": n, "valid_from": today_str, "valid_to": None})
                        elif matched.iloc[0]["name"] != n:
                            # SCD Type 2: 名称が変更されたため、既存レコードの valid_to を昨日(yesterday)で締め、今日から新レコードを追加する
                            # これにより valid_from と valid_to が同一日になるという日付重複の論理的破綻を防止し、事実を工学的に反映する
                            existing_defs.loc[(existing_defs["type"] == t) & (existing_defs["code"] == c) & (existing_defs["valid_to"].isna()), "valid_to"] = yesterday_str
                            new_records.append({"type": t, "code": c, "name": n, "valid_from": today_str, "valid_to": None})
                            
                    if new_records:
                        new_df = pd.DataFrame(new_records)
                        # pd.concat のため警告回避と一貫性の担保
                        final_defs = pd.concat([existing_defs, new_df], ignore_index=True)
                    else:
                        final_defs = existing_defs
                
                # JpxDefinitionRecord の順序に合わせる
                final_cols = ["type", "code", "name", "valid_from", "valid_to"]
                final_defs = final_defs[[c for c in final_cols if c in final_defs.columns]].sort_values(["type", "code", "valid_from"])    
                self.cm.hf.save_and_upload("jpx_definitions", final_defs, defer=True)

        if listing_events:
            events_df = pd.DataFrame(listing_events).drop_duplicates(subset=["code", "type"])
            self.cm.update_listing_history(events_df)

    def reconstruct_name_history(self, code: str) -> pd.DataFrame:
        """
        【Identity Sovereignty】カタログデータから特定の銘柄の社名変更履歴を決定論的に再構成。
        """
        if self.cm.catalog_df.empty:
            return pd.DataFrame()

        # 1. 該当コードの書類を抽出し、提出日時順にソート
        stock_docs = self.cm.catalog_df[self.cm.catalog_df["code"] == code].copy()
        if stock_docs.empty:
            return pd.DataFrame()

        stock_docs.sort_values("submit_at", ascending=True, inplace=True)

        # 2. 社名の遷移を検知
        history = []
        last_name = None
        for _, row in stock_docs.iterrows():
            curr_name = row.get("company_name")
            if not curr_name or pd.isna(curr_name):
                continue
            
            # 漢字名の正規化 (株), (有) 等の除去
            curr_name = self.normalize_company_name(curr_name)

            if last_name and curr_name != last_name:
                history.append({
                    "code": code,
                    "old_name": last_name,
                    "new_name": curr_name,
                    "change_date": str(row["submit_at"])[:10]
                })
            last_name = curr_name

        return pd.DataFrame(history)

    def normalize_company_name(self, name: str) -> str:
        """社名から法的形態の表記(株)などを除去し、純粋な商号を抽出"""
        if not name:
            return name
        for noise in ["株式会社", "有限会社", "合同会社", "（株）", "(株)", "（有）", "(有)"]:
            name = name.replace(noise, "")
        return name.strip()
