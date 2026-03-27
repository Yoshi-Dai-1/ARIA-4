from datetime import datetime
from typing import Optional

import pandas as pd


def normalize_code(code, nationality: str = None) -> Optional[str]:
    """
    証券コードを ARIA 規格に正規化した文字列として返す。
    - JP の場合: 4桁なら末尾0付与で5桁化し、"JP:" プレフィックスを付ける。
    - すでにプレフィックス (例: "JP:") が含まれる場合は二重付与を防止する。
    """
    if code is None or pd.isna(code):
        return None

    # 文字列化して空白除去
    c = str(code).strip()

    # 空文字列、"None"、"nan" の場合は None を返す (不完全なプレフィックス付与を防止)
    if not c or c.lower() in ["none", "nan"]:
        return None

    # すでにプレフィックスがあるかチェック
    if ":" in c:
        prefix, core_code = c.split(":", 1)
        current_nat = prefix.upper()
        c = core_code.strip()
    else:
        current_nat = nationality.upper() if nationality else None

    # Excel/Float 由来の ".0" を除去
    if c.endswith(".0"):
        c = c[:-2]

    # 日本株 (JP) の 5 桁化ルール
    if current_nat == "JP" and len(c) == 4:
        c = c + "0"

    # 最終的なプレフィックス結合
    if current_nat and c:
        return f"{current_nat}:{c}"

    return c or None


def get_edinet_repo_path(doc_id: str, submit_at: str, suffix: str = "zip") -> str:
    """
    EDINET書類のリポジトリ内パスを生成する (Partitioned Structure)
    Worker の保存構造に合わせ、ZIP は zip/ サブディレクトリ、PDF は pdf/ サブディレクトリに格納する。
    例: raw/edinet/year=2024/month=10/day=23/zip/S100BK7G.zip
    例: raw/edinet/year=2024/month=10/day=23/pdf/S100BK7G.pdf
    """
    if not submit_at or len(str(submit_at)) < 10:
        # 日付不明な場合はフォールバック (基本的には発生しない想定)
        return f"raw/edinet/unknown/{suffix}/{doc_id}.{suffix}"

    # 日付部分のみ抽出 (YYYY-MM-DD)
    d = str(submit_at)[:10]
    try:
        y, m, day = d.split("-")
        return f"raw/edinet/year={y}/month={m}/day={day}/{suffix}/{doc_id}.{suffix}"
    except Exception:
        return f"raw/edinet/unknown/{suffix}/{doc_id}.{suffix}"


def parse_datetime(dt_str: str):
    """EDINET の submitDateTime (YYYY-MM-DD HH:MM[:SS]) を堅牢にパースする"""
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        if len(dt_str) > 16:
            return datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
        return datetime.strptime(dt_str[:16], "%Y-%m-%d %H:%M")
    except Exception:
        return None
