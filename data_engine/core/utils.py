import json
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


class AriaJsonEncoder(json.JSONEncoder):
    """
    ARIA プロジェクト向けの標準 JSON エンコーダ。
    numpy 型、pandas 型、datetime 型を適切にネイティブ Python 型に変換してシリアル化する。
    """

    def default(self, obj: Any) -> Any:
        # 欠損値 (NaN/None/pd.NA) の処理を最優先
        if pd.isna(obj):
            return None

        # numpy 整数型の処理
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)

        # numpy 浮動小数点型の処理
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        # numpy 配列の処理
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # datetime / date 型の処理 (ISO 8601 形式)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        return super().default(obj)


def aria_json_dumps(obj: Any, **kwargs) -> str:
    """AriaJsonEncoder を使用してオブジェクトを JSON 文字列に変換する"""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, cls=AriaJsonEncoder, **kwargs)


def aria_json_dump(obj: Any, fp, **kwargs) -> None:
    """AriaJsonEncoder を使用してオブジェクトをファイルに JSON 出力する"""
    kwargs.setdefault("ensure_ascii", False)
    json.dump(obj, fp, cls=AriaJsonEncoder, **kwargs)


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


def get_safe_str(val: Any, default: str = "") -> str:
    """pandas.isna (NaN) を考慮して安全に文字列に変換する"""
    if val is None or pd.isna(val):
        return default
    return str(val)


def get_safe_int(val: Any, default: int = 0) -> int:
    """pandas.isna (NaN) を考慮して安全に整数に変換する"""
    if val is None or pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def log_resources(label: str):
    """【運用監視】メモリ・ディスクの使用状況を出力する (標準ライブラリのみ)

    GitHub Actions ランナー上でのメモリ枯渇を早期検知するための永続的な監視機能。
    Linux (GHA) では /proc/self/status から現在の VmRSS を取得し、
    macOS (開発機) では resource.getrusage の ru_maxrss にフォールバックする。
    """
    import resource
    import shutil
    import sys

    from loguru import logger

    try:
        # Linux: /proc/self/status から現在の VmRSS (Resident Set Size) を取得
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024  # KB → MB
                        break
                else:
                    raise FileNotFoundError
        except (FileNotFoundError, OSError):
            # macOS 等: ru_maxrss からピーク RSS を取得
            # Linux = KB, macOS = bytes
            divisor = 1024 if sys.platform == "linux" else (1024 * 1024)
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor

        disk = shutil.disk_usage("/")
        disk_free_gb = disk.free / (1024 ** 3)
        disk_used_pct = (disk.used / disk.total) * 100
        
        # 半角/全角を考慮した表示幅のパディング（全角=2, 半角=1 として計算）
        import unicodedata
        visual_width = sum(2 if unicodedata.east_asian_width(c) in ('F', 'W', 'A') else 1 for c in label)
        pad_length = max(0, 26 - visual_width)
        padded_label = label + " " * pad_length
        
        logger.info(
            f"📊 [{padded_label}] "
            f"Memory RSS: {rss_mb:4.0f} MB | "
            f"Disk: {disk_free_gb:5.1f} GB free ({disk_used_pct:4.1f}% used)"
        )
    except Exception as e:
        logger.warning(f"📊 [{label}] リソース診断失敗: {e}")


def force_gc():
    """GC を実行し、glibc の malloc に解放済みメモリの OS 返却を強制する。

    Python の gc.collect() はオブジェクトを解放するが、glibc の malloc は
    解放されたメモリを即座に OS に返却しない（アリーナプールに保持する）。
    malloc_trim(0) を呼び出すことで、空きアリーナを OS に返却させ、
    RSS（物理メモリ使用量）を確実に低下させる。
    macOS 等 glibc が存在しない環境では安全にスキップされる。
    """
    import gc

    gc.collect()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # macOS 等: glibc が存在しないためスキップ

