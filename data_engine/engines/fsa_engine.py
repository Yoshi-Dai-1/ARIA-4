import io
import zipfile
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

from data_engine.engines.parsing.edinet.link_base_file_analyzer import account_list_common
from data_engine.core.models import EdinetCodeRecord
from data_engine.core.utils import normalize_code
from data_engine.core.network_utils import GLOBAL_ROBUST_SESSION


class FsaEngine:
    """
    金融庁（FSA）が提供する各種リスト（EDINETコードリスト、提出者集約一覧等）の
    取得、解析、および同期を専門に行うエンジン。

    【工学的主権】APIキー不要の公開URLを使用し、完全無料・自律的に動作する。
    """

    # --- 公開ダウンロードURL（認証不要） ---
    URL_JP = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
    URL_EN = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelisteng/Edinetcode.zip"
    URL_AGG = "https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/download/ESE140190.csv"

    def __init__(self, session: Optional[requests.Session] = None):
        # 堅牢な共通セッションを優先し、なければ GLOBAL を使用
        self.session = session or GLOBAL_ROBUST_SESSION
        logger.debug("FsaEngine を初期化しました。")

    def sync_edinet_code_lists(self) -> Tuple[Dict[str, EdinetCodeRecord], Dict[str, str]]:
        """
        金融庁から最新の EDINET コードリストと集約一覧を取得し、マッピングオブジェクトを生成する。
        【main準拠】APIキーを一切使用せず、公開URLから直接ZIPをダウンロードする。
        """
        results = {}
        agg_map = {}

        try:
            logger.info("EDINETコードリスト (和英) の同期を開始...")

            # --- 日本語版の取得と解析 ---
            res_jp = self.session.get(self.URL_JP, timeout=30)
            res_jp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(res_jp.content)) as z:
                csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
                df_jp = pd.read_csv(z.open(csv_file), encoding="cp932", skiprows=1)

            # --- 英語版の取得と解析 (業種翻訳の抽出用) ---
            res_en = self.session.get(self.URL_EN, timeout=30)
            res_en.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(res_en.content)) as z:
                csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
                df_en = pd.read_csv(z.open(csv_file), encoding="cp932", skiprows=1)

            # --- 集約一覧の取得と解析 ---
            try:
                res_agg = self.session.get(self.URL_AGG, timeout=15)
                res_agg.raise_for_status()
                df_agg = pd.read_csv(io.BytesIO(res_agg.content), encoding="cp932", skiprows=1)
                for _, agg_row in df_agg.iterrows():
                    old_c = str(agg_row.iloc[1]).strip()
                    new_c = str(agg_row.iloc[2]).strip()
                    if old_c and new_c and old_c != new_c:
                        agg_map[old_c] = new_c
                logger.info(f"EDINETコード集約一覧をロード: {len(agg_map)} 件の付け替えを特定")
            except Exception as ae:
                logger.warning(f"集約一覧の取得・解析に失敗しました (継続可能): {ae}")

            # --- 英文業種名のインデックス構築 ---
            en_industry_map = {}
            # カラム名を正規化して検索
            cols_en = {c.strip().lower(): c for c in df_en.columns}

            # Edinet Code カラムの特定
            en_code_col = next((v for k, v in cols_en.items() if "edinet code" in k or "ｅｄｉｎｅｔコード" in k), None)

            # Submitter's industry カラムの特定
            en_ind_col = next((v for k, v in cols_en.items() if "industry" in k or "提出者業種" in k), None)

            if en_code_col and en_ind_col:
                for _, row in df_en.iterrows():
                    e_code = str(row.get(en_code_col, "")).strip()
                    ind_en = str(row.get(en_ind_col, "")).strip()
                    if e_code and ind_en:
                        en_industry_map[e_code] = ind_en
            else:
                logger.warning(
                    f"英語版コードリストの必須カラムが見つかりません。探索結果: code={en_code_col}, "
                    f"industry={en_ind_col}"
                )

            # --- 名寄せ: EDINETコードをキーにマスタベースを構築 ---
            for _, row in df_jp.iterrows():
                e_code = str(row.get("ＥＤＩＮＥＴコード", "")).strip()
                if len(e_code) != 6:
                    continue

                sec_code = self._safe_int_str(row.get("証券コード"))
                # API側の "0" という異常な証券コードを排除
                if sec_code in ("0", "0000", "00000"):
                    sec_code = None
                
                # 【ARIA 強制正規化】EDINET取得時点で 5 桁化・プレフィックス付与を行う
                if sec_code:
                    sec_code = normalize_code(sec_code, nationality="JP")

                jcn = self._safe_int_str(row.get("提出者法人番号"))
                ind_en = en_industry_map.get(e_code)

                results[e_code] = EdinetCodeRecord(
                    edinet_code=e_code,
                    jcn=jcn,
                    submitter_type=row.get("提出者種別"),
                    is_listed_edinet=row.get("上場区分"),
                    is_consolidated=row.get("連結の有無"),
                    capital=self._safe_int_str(row.get("資本金")),
                    settlement_date=str(row.get("決算日") or "").strip() or None,
                    company_name=str(row.get("提出者名") or "").strip() or None,
                    company_name_en=str(row.get("提出者名（英字）") or "").strip() or None,
                    company_name_kana=str(row.get("提出者名（ヨミ）") or "").strip() or None,
                    address=str(row.get("所在地") or "").strip() or None,
                    industry_edinet=str(row.get("提出者業種") or "").strip() or None,
                    industry_edinet_en=ind_en,
                    code=sec_code,
                )

            logger.success(f"EDINETコードリスト同期完了: {len(results)} 件")

        except Exception as e:
            logger.error(f"EDINETコードリストの同期に失敗しました: {e}")

        return results, agg_map

    def _safe_int_str(self, val) -> Optional[str]:
        """数値を安全に文字列化（NaNや空文字はNone）"""
        if pd.isna(val) or val is None or str(val).strip() == "":
            return None
        try:
            # 浮動小数点経由で整数文字列に変換 (123.0 -> "123")
            return str(int(float(val)))
        except ValueError:
            return str(val).strip()
