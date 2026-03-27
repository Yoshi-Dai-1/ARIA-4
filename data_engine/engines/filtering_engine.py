from enum import Enum
from typing import Any, Dict, Optional, Tuple

from data_engine.core.config import CONFIG
from data_engine.core.utils import normalize_code


class ProcessVerdict(str, Enum):
    """ARIA における書類処理の最終判定（司法判断）"""

    PARSE = "parse"  # フル解析対象（有報等）
    SAVE_RAW = "save_raw"  # 解析はしないが、書類（ZIP/PDF）自体は保存する
    SKIP_PROCESSED = "skip_already_processed"  # 既処理によるスキップ
    SKIP_OUT_OF_SCOPE = "skip_out_of_scope"  # スコープ外（Listed/Unlisted設定）によるスキップ
    SKIP_WITHDRAWN = "skip_withdrawn"  # 取下げ済みによるスキップ


class SkipReason(str, Enum):
    """フィルタリングでスキップされた詳細理由"""

    NONE = "none"
    WITHDRAWN = "withdrawn"
    ALREADY_PROCESSED = "already_processed"
    NO_SEC_CODE = "no_sec_code"  # 上場検索で証券コードなし
    INVALID_CODE_LENGTH = "invalid_code_length"  # 証券コードの形式不正
    HAS_SEC_CODE = "has_sec_code"  # 非上場検索で証券コードあり
    REACTION_REQUIRED = "retraction_sync_required"  # 取下げ同期が必要（スキップしない）
    INVALID_METADATA = "invalid_metadata"  # 提出日時や提出者名が欠落（不開示書類等）


class FilteringEngine:
    """
    ARIA の「憲法の番人」
    あらゆる書類に対し、ARIA_SCOPE と物理的事実に基づき一貫した処理判定を下す。
    将来の米国株（US:Ticker）や TDNet への拡張を考慮した設計。
    """

    def __init__(self, aria_scope: str = "All"):
        self.aria_scope = aria_scope
        # 【ARIA 解析対象の中央制御】 config.py に委譲


    def get_verdict(
        self,
        row: Dict[str, Any],
        is_processed: bool = False,
        local_status: Optional[str] = None,
    ) -> Tuple[ProcessVerdict, SkipReason, Dict[str, Any]]:
        """
        書類メタデータを物理的事実（Triple: Doc, Ord, Form）に基づき判定する。
        """
        # 0. メタデータの完全性チェック (幽霊レコードの排除)
        # ARIA 品質基準: 提出日時、提出者名、書類種別コードのいずれかが欠落している場合はノイズとみなす
        submit_at = row.get("submitDateTime")
        filer_name = row.get("filerName")
        doc_type = row.get("docTypeCode") or row.get("type")

        if not submit_at or not filer_name or not doc_type:
            # 物理的指標を最低限埋める (ログ用)
            indicators = {
                "doc": doc_type or "---",
                "ord": row.get("ordinanceCode") or "---",
                "form": row.get("formCode") or "---",
                "xbrl": False
            }
            return ProcessVerdict.SKIP_OUT_OF_SCOPE, SkipReason.INVALID_METADATA, indicators

        form_code = row.get("formCode") or row.get("form") or "---"
        ordinance = row.get("ordinanceCode") or row.get("ord") or "---"
        xbrl_flag = str(row.get("xbrlFlag") or row.get("xbrl") or "0")

        # 3つの物理的指標 (Facts for logging)
        indicators = {
            "doc": doc_type,
            "ord": ordinance,
            "form": form_code,
            "xbrl": xbrl_flag == "1"
        }

        # 1. 取下げチェック (物理的事実の優先)
        is_withdrawn = str(row.get("withdrawalStatus")) == "1"

        # 2. 既処理チェックと取下げ同期の特殊判定
        if is_processed:
            if is_withdrawn and local_status != "retracted":
                pass
            else:
                return ProcessVerdict.SKIP_PROCESSED, SkipReason.ALREADY_PROCESSED, indicators

        if is_withdrawn:
            return ProcessVerdict.SKIP_WITHDRAWN, SkipReason.WITHDRAWN, indicators

        # 3. 修正済み書類の排除 (Fact: docInfoEditStatus == 2 は古い版)
        if str(row.get("docInfoEditStatus")) == "2":
            return ProcessVerdict.SKIP_OUT_OF_SCOPE, SkipReason.ALREADY_PROCESSED, indicators

        # 4. スコープ判定
        if self.aria_scope == "Listed":
            norm_code = normalize_code(row.get("secCode"), nationality="JP")
            if not norm_code:
                return ProcessVerdict.SKIP_OUT_OF_SCOPE, SkipReason.NO_SEC_CODE, indicators
            core_code = norm_code.split(":", 1)[1] if ":" in norm_code else norm_code
            if len(core_code) < 4:
                return ProcessVerdict.SKIP_OUT_OF_SCOPE, SkipReason.INVALID_CODE_LENGTH, indicators
        elif self.aria_scope == "Unlisted":
            norm_code = normalize_code(row.get("secCode"), nationality="JP")
            if norm_code:
                core_code = norm_code.split(":", 1)[1] if ":" in norm_code else norm_code
                if len(core_code) >= 4:
                    return ProcessVerdict.SKIP_OUT_OF_SCOPE, SkipReason.HAS_SEC_CODE, indicators

        # 5. 解析対象判定 (Fact: DocType, XBRL)
        is_parsing_target = (
            doc_type in CONFIG.XBRL_TARGET_DOC_TYPES
            and indicators["xbrl"]
        )

        if is_parsing_target:
            return ProcessVerdict.PARSE, SkipReason.NONE, indicators

        # 解析非対象だが、書類(ZIP/PDF)とメタデータをレイクハウスへ保存する
        return ProcessVerdict.SAVE_RAW, SkipReason.NONE, indicators
