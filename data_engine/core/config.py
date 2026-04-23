import json
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from loguru import logger

# --- 物理パスの定義 ---
CORE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = CORE_DIR.parent.parent.resolve()
CONFIG_PATH = CORE_DIR / "aria_config.json"
TAXONOMY_PATH = CORE_DIR / "taxonomy_urls.json"


class AriaConfig:
    """
    ARIA の全設定を管理する SSOT (Single Source of Truth)。
    aria_config.json, taxonomy_urls.json, および環境変数を統合管理する。
    """

    def __init__(self):
        # 0. .env のロード (ローカル開発/実行時)
        load_dotenv(ROOT_DIR / ".env")

        # 1. 基本設定のロード
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing absolute configuration file: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._raw_config = json.load(f)

        # 2. タクソノミURLのロード
        self.TAXONOMY_URLS: Dict[str, str] = {}
        if TAXONOMY_PATH.exists():
            with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
                self.TAXONOMY_URLS = json.load(f)

        # 3. 各項目のバリデーションと正規化
        self.ARIA_SCOPE = self._validate_scope(self._raw_config.get("aria_scope"))
        self.XBRL_TARGET_DOC_TYPES = self._raw_config.get(
            "xbrl_target_doc_types", 
            ["120", "130", "140", "150", "160", "170", "180", "190"]
        )
        self.TSE_URL = self._raw_config.get(
            "tse_url",
            "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
        )
        self.HF_WARNING_THRESHOLD = self._raw_config.get("hf_folder_file_warning_threshold", 7000)

        # 4. FSA (金融庁) リソース URL
        fsa_base = "https://disclosure2.edinet-fsa.go.jp"
        self.FSA_CODE_LIST_URL = f"{fsa_base}/weee0020.zip"
        self.FSA_AGGREGATION_URL = f"{fsa_base}/weee0040.zip"
        self.FSA_EN_LIST_URL = f"{fsa_base}/weee0030.zip"

        # 5. 並列実行設定
        self.PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", 2))
        self.BATCH_PARALLEL_SIZE = 8

        # 6. 機密情報 (環境変数)
        self.EDINET_API_KEY = os.getenv("EDINET_API_KEY")
        self.HF_REPO = os.getenv("HF_REPO")
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        # 7. パス設定 (物理構造の SSOT)
        self.DATA_PATH = ROOT_DIR / "data"
        self.RAW_DIR = self.DATA_PATH / "raw"
        self.TEMP_DIR = self.DATA_PATH / "temp"

        # 8. ディレクトリの存在保証 (CI環境等のクリーンスタート対策)
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # 9. グローバルな実行環境の統制 (CI/静寂性)
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO").upper()

    def _validate_scope(self, scope: str) -> str:
        if not scope or not str(scope).strip():
            raise ValueError("aria_config.json に 'aria_scope' が定義されていません。")
        norm = str(scope).strip().capitalize()
        if norm not in ["Listed", "Unlisted", "All"]:
            raise ValueError(f"無効な ARIA_SCOPE です: {scope}")
        return norm

    def validate_env(self, production: bool = True, edinet: bool = True):
        """環境変数の存否をチェックし、不足があれば警告またはエラーを出す"""
        missing = []
        if edinet and not self.EDINET_API_KEY:
            missing.append("EDINET_API_KEY")
        if not self.HF_REPO:
            missing.append("HF_REPO")
        if not self.HF_TOKEN:
            missing.append("HF_TOKEN")

        if missing:
            msg = f"注意: 以下の環境変数が設定されていません: {', '.join(missing)}"
            if production:
                logger.warning(msg)
            else:
                logger.debug(msg)


# モジュールインポート時にインスタンス化して CONFIG として提供
CONFIG = AriaConfig()

# 後方互換性およびショートカット
ARIA_SCOPE = CONFIG.ARIA_SCOPE
TSE_URL = CONFIG.TSE_URL
HF_WARNING_THRESHOLD = CONFIG.HF_WARNING_THRESHOLD
DATA_PATH = CONFIG.DATA_PATH
RAW_DIR = CONFIG.RAW_DIR
TEMP_DIR = CONFIG.TEMP_DIR
