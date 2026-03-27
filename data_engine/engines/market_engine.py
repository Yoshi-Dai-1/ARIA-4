import io
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from data_engine.core.network_utils import get_robust_session
from data_engine.core.utils import normalize_code

# normalize_code is now imported from utils


class IndexStrategy(ABC):
    """指数ごとのデータ取得ロジックの基底クラス"""

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        構成銘柄データを取得する
        Returns:
            pd.DataFrame: columns=["code", "weight"] (sector等は含まない)
        """
        pass


class NikkeiStrategy(IndexStrategy):
    """日経225 (Nikkei Source) および同形式のCSV用"""

    def __init__(self, url: str = None):
        # ユーザー指定のアーカイブ版URL (403を回避しやすい)
        default_url = "https://indexes.nikkei.co.jp/nkave/archives/file/nikkei_stock_average_weight_jp.csv"
        self.url = url if url else default_url
        self.index_name = "日経225"  # デフォルト
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }

    def fetch_data(self) -> pd.DataFrame:
        logger.info(f"{self.index_name}構成銘柄を取得中 (Archive CSV)...")
        session = get_robust_session()
        try:
            r = session.get(self.url, headers=self.headers)
            if r.status_code != 200:
                logger.error(f"{self.index_name}取得エラー: HTTP {r.status_code}")
                # HTTP 403 の場合は詳細なメッセージを出す
                if r.status_code == 403:
                    logger.error("日経新聞社サイトからアクセスが拒絶されました (403)。")
                r.raise_for_status()
        except Exception as e:
            logger.error(f"{self.index_name}リクエスト失敗: {e}")
            raise

        # Shift-JISでデコード
        try:
            # アーカイブCSV形式: 1行目がヘッダ。日付,銘柄名,コード,業種,ウエイト
            df = pd.read_csv(io.BytesIO(r.content), encoding="shift_jis")

            # 日経新聞のCSVは末尾に「データ取得元...」などの説明行が入ることがあるため、
            # コードが数値として解釈できる行のみを残す
            # また、カラム名に「ウエイト」と「ウエート」の表記揺れがあるため正規表現で対応
            code_col = next((c for c in df.columns if re.search(r"コード|Code", c, re.IGNORECASE)), None)
            weight_col = next(
                (c for c in df.columns if re.search(r"ウエ[イート]|ウェ[イート]|Weight", c, re.IGNORECASE)), None
            )

            if not code_col or not weight_col:
                raise ValueError(f"必須カラムが見つかりません。Columns: {df.columns}")

            # クリーニング
            df = df.dropna(subset=[code_col, weight_col])

            # コードを文字列化 (JPプレフィックス付与)
            df["code"] = df[code_col].astype(str).str.replace(".0", "", regex=False).str.strip().apply(
                lambda x: normalize_code(x, nationality="JP")
            )

            # ウエイトのパース
            def parse_weight(x):
                if isinstance(x, str):
                    clean = x.replace("%", "").strip()
                    if not clean:
                        return 0.0
                    return float(clean)
                return float(x)

            df["weight"] = df[weight_col].apply(parse_weight)

            logger.success(f"{self.index_name}データ取得成功: {len(df)} 件")
            return df[["code", "weight"]]

        except Exception as e:
            logger.error(f"{self.index_name}パース失敗: {e}")
            raise e


class TopixStrategy(IndexStrategy):
    """TOPIX (JPX Source)"""

    def __init__(self):
        self.url = "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv"

    def fetch_data(self) -> pd.DataFrame:
        logger.info("TOPIX構成銘柄を取得中...")
        session = get_robust_session()
        r = session.get(self.url)
        r.raise_for_status()

        try:
            # JPX CSV confirmed as Shift-JIS
            df = pd.read_csv(io.BytesIO(r.content), encoding="shift_jis")

            # 想定カラム: 日付,銘柄名,コード,業種,TOPIXに占める個別銘柄のウエイト,ニューインデックス区分
            # JPXの長大なヘッダや名称揺れにも正規表現で対応
            code_col = next((c for c in df.columns if re.search(r"コード|Code", c, re.IGNORECASE)), None)
            weight_col = next((c for c in df.columns if re.search(r"ウエ[イート]|Weight", c, re.IGNORECASE)), None)

            if not code_col or not weight_col:
                raise ValueError(f"TOPIX必須カラム欠落: {df.columns}")

            # フッター等の空行を削除
            df = df.dropna(subset=[code_col, weight_col])

            df = df[[code_col, weight_col]].rename(columns={code_col: "code", weight_col: "weight"})

            # 型変換 (JPプレフィックス付与)
            df["code"] = df["code"].astype(str).str.strip().apply(lambda x: normalize_code(x, nationality="JP"))

            def parse_weight(x):
                if isinstance(x, str):
                    clean = x.replace("%", "").strip()
                    if not clean:
                        return 0.0
                    return float(clean)
                return float(x)

            df["weight"] = df["weight"].apply(parse_weight)

            logger.success(f"TOPIXデータ取得成功: {len(df)} 件")
            return df[["code", "weight"]]

        except Exception as e:
            logger.error(f"TOPIXパース失敗: {e}")
            raise e


class MarketDataEngine:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.strategies: Dict[str, IndexStrategy] = {
            "Nikkei225": NikkeiStrategy(),
            "NikkeiHighDiv50": NikkeiStrategy(
                "https://indexes.nikkei.co.jp/nkave/archives/file/nikkei_high_dividend_yield_50_weight_jp.csv"
            ),
            "JPXNikkei400": NikkeiStrategy(
                "https://indexes.nikkei.co.jp/nkave/archives/file/jpx_nikkei_index_400_weight_jp.csv"
            ),
            "JPXNikkeiMidSmall": NikkeiStrategy(
                "https://indexes.nikkei.co.jp/nkave/archives/file/jpx_nikkei_mid_small_weight_jp.csv"
            ),
            "TOPIX": TopixStrategy(),
        }
        # 各戦略に表示用の指数名を設定
        self.strategies["NikkeiHighDiv50"].index_name = "日経高配当50"
        self.strategies["JPXNikkei400"].index_name = "JPX日経400"
        self.strategies["JPXNikkeiMidSmall"].index_name = "JPX日経中小型"

        from data_engine.core.config import CONFIG

        self.jpx_url = CONFIG.TSE_URL

    def fetch_jpx_master(self) -> pd.DataFrame:
        """JPXから最新の銘柄一覧を取得 (Retry付き)"""
        logger.info("JPX銘柄マスタを取得中...")
        session = get_robust_session()
        r = session.get(self.jpx_url, stream=True)
        r.raise_for_status()

        # 保存して読み込む (Excel形式のため)
        from data_engine.core.config import CONFIG

        CONFIG.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        xls_path = CONFIG.TEMP_DIR / "jpx_master.xls"
        with xls_path.open("wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        df = pd.read_excel(xls_path, dtype={"コード": str})
        # カラムマッピング (ARIAモデルの定義に合わせる)
        df = df.rename(
            columns={
                "コード": "code",
                "銘柄名": "company_name",
                "33業種コード": "sector_33_code",
                "33業種区分": "sector_jpx_33",
                "17業種コード": "sector_17_code",
                "17業種区分": "sector_jpx_17",
                "市場・商品区分": "market",
                "規模コード": "size_code",
                "規模区分": "size_category",
            }
        )
        # 数値カラムのゴミ（"-" 等）を処理
        for col in ["sector_33_code", "sector_17_code", "size_code"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace("-", None)

        df["code"] = df["code"].astype(str).str.strip().apply(lambda x: normalize_code(x, nationality="JP"))

        return df[
            [
                "code",
                "company_name",
                "sector_jpx_33",
                "sector_33_code",
                "sector_jpx_17",
                "sector_17_code",
                "market",
                "size_code",
                "size_category",
            ]
        ]

    def fetch_index_data(self, index_name: str) -> pd.DataFrame:
        """指数データの取得 (Retry付き)"""
        if index_name not in self.strategies:
            raise ValueError(f"未対応の指数: {index_name}")

        return self.strategies[index_name].fetch_data()

    def generate_index_diff(
        self, index_name: str, old_const: pd.DataFrame, new_const: pd.DataFrame, date_str: str
    ) -> pd.DataFrame:
        """指数イベント差分生成 (ADD, REMOVE, UPDATE)"""
        events = []

        # 辞書化 {code: weight}
        old_map = dict(zip(old_const["code"], old_const["weight"])) if not old_const.empty else {}
        new_map = dict(zip(new_const["code"], new_const["weight"])) if not new_const.empty else {}

        old_keys = set(old_map.keys())
        new_keys = set(new_map.keys())

        # ADD
        for code in new_keys - old_keys:
            events.append(
                {
                    "date": date_str,
                    "index_name": index_name,
                    "code": code,
                    "type": "ADD",
                    "old_value": None,
                    "new_value": new_map[code],
                }
            )

        # REMOVE
        for code in old_keys - new_keys:
            events.append(
                {
                    "date": date_str,
                    "index_name": index_name,
                    "code": code,
                    "type": "REMOVE",
                    "old_value": old_map[code],
                    "new_value": None,
                }
            )

        # UPDATE (共通部分でウエイト変化)
        for code in new_keys & old_keys:
            v_old = old_map[code]
            v_new = new_map[code]
            # 浮動小数点比較 (許容誤差 1e-6)
            if abs(v_old - v_new) > 1e-6:
                events.append(
                    {
                        "date": date_str,
                        "index_name": index_name,
                        "code": code,
                        "type": "UPDATE",
                        "old_value": v_old,
                        "new_value": v_new,
                    }
                )

        return pd.DataFrame(events, columns=["date", "index_name", "code", "type", "old_value", "new_value"])
