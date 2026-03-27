import datetime
import pandas as pd
from loguru import logger

class LifecycleManager:
    """銘柄の生存状態（Active/Disappeared）および親子関係・イベント検知を担当"""

    def __init__(self, catalog_manager):
        self.cm = catalog_manager

    def track_disappearance(self, new_master_df: pd.DataFrame, current_ids_in_run: set) -> pd.DataFrame:
        """【Tracking】消失銘柄の判定とフラグ設定"""
        if self.cm.master_df.empty:
            return new_master_df

        # identity_key の生成 (edinet_code優先) 
        # ※master_df は更新前、new_master_df は今回の統合結果
        if "identity_key" not in self.cm.master_df.columns:
            self.cm.master_df["identity_key"] = self.cm.master_df["edinet_code"].fillna(self.cm.master_df["code"]).fillna(self.cm.master_df["jcn"])
        
        if "identity_key" not in new_master_df.columns:
            new_master_df["identity_key"] = new_master_df["edinet_code"].fillna(new_master_df["code"]).fillna(new_master_df["jcn"])

        # 以前は存在したが、今回の入力（EDINET/JPX）のどちらにも現れなかった銘柄を特定
        disappeared_mask = (
            new_master_df["identity_key"].isin(self.cm.master_df["identity_key"]) & 
            ~new_master_df["identity_key"].isin(current_ids_in_run)
        )
        # 既に disappeared なものは維持、新規消失分をマーク
        new_master_df.loc[disappeared_mask, "is_disappeared"] = True
        new_master_df.loc[disappeared_mask, "is_active"] = False
        
        return new_master_df

    def detect_listing_events(self, latest_rec, current_master_df: pd.DataFrame) -> list:
        """上場・廃止イベントの検知 (is_active の変化に基づく)"""
        events = []
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        code = latest_rec.get("code")
        if not code or pd.isna(code):
            return events

        # is_active の現在値
        is_active_now = latest_rec.get("is_active", True)
        
        if current_master_df.empty:
            if is_active_now:
                events.append({"code": code, "type": "LISTING", "event_date": today})
            return events

        old_row = current_master_df[current_master_df["code"] == code]
        
        if not old_row.empty:
            was_active = old_row.iloc[0].get("is_active", True)
            if was_active and not is_active_now:
                events.append({"code": code, "type": "DELISTING", "event_date": today})
            elif not was_active and is_active_now:
                events.append({"code": code, "type": "LISTING", "event_date": today})
        elif is_active_now:
            # 新規発見
            events.append({"code": code, "type": "LISTING", "event_date": today})
            
        return events

    def setup_parent_code(self, rec: dict) -> dict:
        """【Parenting】優先株の親紐付け設定"""
        sec_code = rec.get("code")
        if sec_code and sec_code[-1] != "0":
            # 5桁目が0以外なら優先株とみなし、親コード(普通株)を推定
            rec["parent_code"] = f"JP:{sec_code.split(':', 1)[1][:4]}0"
        return rec
