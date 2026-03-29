import math
from typing import Any, Optional, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_engine.core.utils import normalize_code


class EdinetDocument(BaseModel):
    """EDINET APIから取得される書類メタデータのバリデーションモデル (API v2 全フィールド網羅)"""

    seqNumber: int
    docID: str
    edinetCode: Optional[str] = None
    secCode: Optional[str] = None
    JCN: Optional[str] = None
    filerName: Optional[str] = None
    fundCode: Optional[str] = None
    ordinanceCode: Optional[str] = None
    formCode: Optional[str] = None
    docTypeCode: Optional[str] = None
    periodStart: Optional[str] = None
    periodEnd: Optional[str] = None
    submitDateTime: str
    docDescription: Optional[str] = None
    issuerEdinetCode: Optional[str] = None
    subjectEdinetCode: Optional[str] = None
    subsidiaryEdinetCode: Optional[str] = None
    currentReportReason: Optional[str] = None
    parentDocID: Optional[str] = None
    opeDateTime: Optional[str] = None
    withdrawalStatus: str = "0"
    docInfoEditStatus: str = "0"
    disclosureStatus: str = "0"
    xbrlFlag: str = "0"
    pdfFlag: str = "0"
    attachDocFlag: str = "0"
    englishDocFlag: str = "0"
    csvFlag: str = "0"
    legalStatus: str = "0"

    @field_validator("secCode", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")


class EdinetCodeRecord(BaseModel):
    """金融庁公表のEDINETコードリストレコード (13項目 + 英語版補完情報)"""

    edinet_code: str
    submitter_type: Optional[str] = None  # 提出者種別
    is_listed_edinet: Optional[str] = None  # 上場区分 (上場/非上場)
    is_consolidated: Optional[bool] = None  # 連結の有無 (True/False)
    capital: Optional[float] = None  # 資本金
    settlement_date: Optional[str] = None  # 決算日
    company_name: str  # 提出者名 (和文)
    company_name_en: Optional[str] = None  # 提出者名 (和文/英字)
    company_name_kana: Optional[str] = None  # 提出者名 (ヨミ)
    address: Optional[str] = None  # 所在地
    industry_edinet: Optional[str] = None  # 提出者業種 (和文)
    industry_edinet_en: Optional[str] = None  # 提出者業種 (英文/英語版リストより取得)
    code: Optional[str] = None  # 証券コード (5桁)
    jcn: Optional[str] = None  # 提出者法人番号 (JCN)

    @field_validator(
        "submitter_type",
        "is_listed_edinet",
        "is_consolidated",
        "settlement_date",
        "company_name_en",
        "company_name_kana",
        "address",
        "industry_edinet",
        "industry_edinet_en",
        "code",
        "jcn",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, v: Any) -> Any:
        """物理的事実に基づき、NaN/空欄/- を None に、有/無を bool に正規化する"""
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str):
            s_v = v.strip()
            if s_v.lower() in ["nan", "none", "", "-"]:
                return None
            if s_v == "有":
                return True
            if s_v == "無":
                return False
        return v

    @field_validator("code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")


class CatalogRecord(BaseModel):
    """統合ドキュメントカタログ (documents_index.parquet) のレコードモデル (39カラム構成)"""

    # 1. Identifiers (識別子・基本情報)
    doc_id: str
    bin_id: Optional[str] = None  # 物理パーティションID (分析用 Bin 分割キー)
    edinet_code: Optional[str] = None
    code: Optional[str] = None  # 証券コード (5桁)
    jcn: Optional[str] = None  # 法人番号 (Japan Corporate Number)
    company_name: str

    # 2. Timeline & Main Content (Web UI 最適化による前寄せ)
    submit_at: str
    seq_number: Optional[int] = None  # 同日提出書類の連番 (EDINETが保証する提出順序)
    title: Optional[str] = None
    doc_type: Optional[str] = None

    # 3. Supplemental Identifiers
    issuer_edinet_code: Optional[str] = None  # 発行者EDINETコード
    subject_edinet_code: Optional[str] = None  # 公開買付対象者EDINETコード
    subsidiary_edinet_code: Optional[str] = None  # 子会社EDINETコード (カンマ区切り)
    fund_code: Optional[str] = None  # ファンドコード (投資信託等)

    # 4. Domain/Fiscal (決算・期間属性)
    fiscal_year: Optional[int] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    num_months: Optional[int] = None
    accounting_standard: Optional[str] = None  # 会計基準 (J-GAAP, IFRS, etc.)

    # 5. Document Details (書類詳細特性)
    form_code: Optional[str] = None
    ordinance_code: Optional[str] = None
    is_amendment: bool = False
    parent_doc_id: Optional[str] = None  # 訂正対象の親書類ID
    withdrawal_status: Optional[str] = None  # 取下区分 (1:取下済)
    doc_info_edit_status: Optional[str] = None  # 財務局修正状態 (1:修正情報, 2:修正された書類)
    disclosure_status: Optional[str] = None  # 開示ステータス (1:OK, 2:修正 etc.)
    legal_status: Optional[str] = None  # 縦覧区分 (1:縦覧中, 2:延長期間中, 0:期間満了)
    current_report_reason: Optional[str] = None  # 臨時報告書の提出理由
    xbrl_flag: Optional[bool] = None  # EDINET XBRL(ZIP) 提供可否 (API xbrlFlag)
    pdf_flag: Optional[bool] = None  # EDINET PDF 提供可否 (API pdfFlag)
    csv_flag: Optional[bool] = None  # EDINET CSV 提供可否 (API csvFlag)
    english_flag: Optional[bool] = None  # EDINET 英文ファイル提供可否 (API englishDocFlag)
    attachment_flag: Optional[bool] = None  # EDINET 代替書面・添付文書提供可否 (API attachDocFlag)

    # 6. Infrastructure (システム管理情報)
    raw_zip_path: Optional[str] = None
    pdf_path: Optional[str] = None
    english_path: Optional[str] = None  # 英文資料(type=4)の展開先ディレクトリパス
    attach_path: Optional[str] = None  # 添付文書(type=3)の展開先ディレクトリパス
    processed_status: Optional[str] = "success"
    source: str = "EDINET"

    # 7. API V2 Lifecycle (増分同期・運用メタデータ)
    ope_date_time: Optional[str] = None  # 操作日時 (API V2 の核心項目)

    @field_validator("code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")

    @field_validator(
        "jcn",
        "edinet_code",
        "issuer_edinet_code",
        "subject_edinet_code",
        "subsidiary_edinet_code",
        "fund_code",
        "period_start",
        "period_end",
        "accounting_standard",
        "doc_type",
        "title",
        "form_code",
        "ordinance_code",
        "parent_doc_id",
        "withdrawal_status",
        "doc_info_edit_status",
        "disclosure_status",
        "legal_status",
        "current_report_reason",
        "raw_zip_path",
        "pdf_path",
        "english_path",
        "attach_path",
        "ope_date_time",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str):
            s_v = v.strip()
            if s_v.lower() in ["nan", "none", "", "-"]:
                return None
            return s_v
        return v


class StockMasterRecord(BaseModel):
    """
    ARIA 銘銘マスタレコード (Perfect Integrity)
    識別子、属性、業界、状態の論理的順序で構成。ウェブアプリでの利用を最適化。
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # --- 1. Essential Web Identity (Primary Keys) ---
    identity_key: str = Field(..., description="ARIA ユニーク識別子 (EDINET_CODE or CODE or JCN)")
    edinet_code: Optional[str] = Field(None, description="EDINETコード (EXXXXX)")
    code: Optional[str] = Field(None, description="証券コード (5桁正規化: JP:XXXX0)")
    jcn: Optional[str] = Field(None, description="法人番号 (13桁)")

    # --- 2. Identity Attributes (Searchable) ---
    company_name: str = Field(..., description="提出者名 (和文)")
    company_name_en: Optional[str] = Field(None, description="提出者名 (英文)")
    company_name_kana: Optional[str] = Field(None, description="提出者名 (ヨミ)")

    # --- 3. Lifecycle & Tracking (Operational Metadata) ---
    is_active: bool = Field(True, description="ARIA 収集・追跡対象フラグ(運用の真実)")
    is_disappeared: bool = Field(False, description="全ソース（EDINET/JPX）から消失した銘柄フラグ")
    is_listed_edinet: bool = Field(False, description="EDINET公式名簿 上場フラグ (法令の真実)")
    last_submitted_at: Optional[str] = Field(None, description="最終書類提出日時")

    # --- 4. Industry & Market Classification (Normalized) ---
    market: Optional[str] = Field(None, description="上場市場名")
    sector_jpx_33: Optional[str] = Field(None, description="JPX 33業種区分名")
    sector_33_code: Optional[str] = Field(None, description="JPX 33業種コード")
    sector_jpx_17: Optional[str] = Field(None, description="JPX 17業種区分名")
    sector_17_code: Optional[str] = Field(None, description="JPX 17業種コード")
    industry_edinet: Optional[str] = Field(None, description="EDINET業種区分 (和文)")
    industry_edinet_en: Optional[str] = Field(None, description="EDINET業種区分 (英文)")
    size_code: Optional[str] = Field(None, description="規模コード")
    size_category: Optional[str] = Field(None, description="規模区分名")

    # --- 5. Financial & Corporate Attributes ---
    parent_code: Optional[str] = Field(None, description="親銘柄コード (優先株などの場合)")
    former_edinet_codes: Optional[str] = Field(None, description="旧EDINETコード履歴 (カンマ区切り)")
    submitter_type: Optional[str] = Field(None, description="提出者種別")
    is_consolidated: Optional[bool] = Field(None, description="連結の有無 (True/False)")
    capital: Optional[float] = Field(None, description="資本金 (単位: 百万円)")
    settlement_date: Optional[str] = Field(None, description="決算期末")
    address: Optional[str] = Field(None, description="所在地")

    @field_validator(
        "identity_key",
        "edinet_code",
        "code",
        "jcn",
        "company_name_en",
        "company_name_kana",
        "submitter_type",
        "settlement_date",
        "address",
        "sector_jpx_33",
        "sector_33_code",
        "sector_jpx_17",
        "sector_17_code",
        "industry_edinet",
        "industry_edinet_en",
        "market",
        "size_code",
        "size_category",
        "last_submitted_at",
        "former_edinet_codes",
        "is_consolidated",
        "is_disappeared",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, v: Any) -> Any:
        """pandas の NaN や文字列 'nan' を物理的に排除し、工学的主権を保つ"""
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str):
            s_v = v.strip()
            # 連結フラグの正規化: "有" -> True, "無" -> False
            if s_v == "有":
                return True
            if s_v == "無":
                return False
            # 欠損値の正規化
            if s_v.lower() in ["nan", "none", "", "-"]:
                return None
            return s_v

        # 既に bool や数値などの場合はそのまま
        if isinstance(v, (bool, int, float)):
            return v

        return str(v).strip()

    # 証券コードを5桁に正規化 (SICC準拠)
    @field_validator("code", "parent_code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        return normalize_code(v, nationality="JP")

    @field_validator("is_listed_edinet", mode="before")
    @classmethod
    def convert_to_bool_listed(cls, v: Any) -> bool:
        """「上場/非上場」の文字列を物理的な bool へ変換する"""
        if isinstance(v, bool):
            return v
        s_v = str(v).strip()
        if s_v == "上場":
            return True
        if s_v == "非上場":
            return False
        return False  # デフォルトは非上場扱い（EDINETコードリストに載っていない場合）


class ListingEvent(BaseModel):
    """上場・廃止イベントの記録モデル"""

    code: str
    type: str  # LISTING, DELISTING
    event_date: str

    @field_validator("code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")


class NameEvent(BaseModel):
    """社名変更の記録モデル (漢字のみの追跡)"""

    code: str
    old_name: str
    new_name: str
    change_date: str

    @field_validator("code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")


class JpxDefinitionRecord(BaseModel):
    """JPX 業種・規模区分名等の定義マスタレコードモデル (Web API 正規化用)"""

    type: str = Field(..., description="区分種別 (sector_33, sector_17, size)")
    code: str = Field(..., description="区分コード")
    name: str = Field(..., description="区分名称 (和文)")
    valid_from: str = Field(..., description="適用開始日 (YYYY-MM-DD)")
    valid_to: Optional[str] = Field(None, description="適用終了日 (YYYY-MM-DD/現在有効ならNull)")


class IndexEvent(BaseModel):
    """指数構成銘柄の変更イベント記録モデル"""

    date: str
    index_name: str
    code: str
    type: str  # ADD, REMOVE, UPDATE
    old_value: Optional[float] = None
    new_value: Optional[float] = None

    @field_validator("code", mode="before")
    @classmethod
    def normalize_sec_code(cls, v: Optional[str]) -> Optional[str]:
        from data_engine.core.utils import normalize_code

        return normalize_code(v, nationality="JP")


class FinancialValueRecord(BaseModel):
    """財務数値データ (financial_values) のレコードモデル"""

    # 1. Identity (誰のデータか)
    docid: str
    # 2. Core Data (何の値か)
    key: Optional[str] = None
    label_jp: Optional[str] = None
    data_str: Optional[str] = None
    unit: Optional[str] = None
    # 3. Pivot & Filter Flags (比較・抽出軸)
    current_flg: Optional[int] = None
    non_consolidated_flg: Optional[int] = None
    prior_flg: Optional[int] = None
    # 4. Context (いつの状態か)
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant_date: Optional[str] = None
    end_date_pv: Optional[str] = None
    instant_date_pv: Optional[str] = None
    # 5. Taxonomy Details (詳細属性)
    role: Optional[str] = None
    label_en: Optional[str] = None
    label_jp_long: Optional[str] = None
    label_en_long: Optional[str] = None
    AccountingStandardsDEI: Optional[str] = None
    # 6. Technical Metadata (システム・XBRL用)
    isTextBlock_flg: int = 0
    abstract_flg: int = 0
    decimals: Optional[str] = None
    precision: Optional[str] = None
    context_ref: Optional[str] = None
    element_name: Optional[str] = None
    period_type: Optional[str] = None
    scenario: Optional[str] = None
    order: Optional[float] = None


class QualitativeTextRecord(BaseModel):
    """定性情報テキスト (qualitative_text) のレコードモデル"""

    # 1. Identity (誰のデータか)
    docid: str
    # 2. Core Data (何の値か)
    key: Optional[str] = None
    label_jp: Optional[str] = None
    data_str: Optional[str] = None
    unit: Optional[str] = None
    # 3. Pivot & Filter Flags (比較・抽出軸)
    current_flg: Optional[int] = None
    non_consolidated_flg: Optional[int] = None
    prior_flg: Optional[int] = None
    # 4. Context (いつの状態か)
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant_date: Optional[str] = None
    end_date_pv: Optional[str] = None
    instant_date_pv: Optional[str] = None
    # 5. Taxonomy Details (詳細属性)
    role: Optional[str] = None
    label_en: Optional[str] = None
    label_jp_long: Optional[str] = None
    label_en_long: Optional[str] = None
    AccountingStandardsDEI: Optional[str] = None
    # 6. Technical Metadata (システム・XBRL用)
    isTextBlock_flg: int = 1
    abstract_flg: int = 0
    decimals: Optional[str] = None
    precision: Optional[str] = None
    context_ref: Optional[str] = None
    element_name: Optional[str] = None
    period_type: Optional[str] = None
    scenario: Optional[str] = None
    order: Optional[float] = None


# =============================================================================
# PyArrow Schema 自動導出 (Phase 3: 金型アーキテクチャ)
# =============================================================================

# Python 型 → PyArrow 型の対応表
_PYTHON_TO_PYARROW = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
}


def pydantic_to_pyarrow(model_class) -> pa.Schema:
    """
    Pydantic モデルから PyArrow スキーマを自動導出する。
    models.py を変更すれば Parquet スキーマが自動追従する SSOT 設計。
    """
    fields = []
    for name, info in model_class.model_fields.items():
        py_type = info.annotation
        nullable = False

        # Pydantic v2: info.annotation に型情報が入っている
        # Optional[X] や Union[X, None] を判定
        origin = get_origin(py_type)
        if origin is Union:
            args = get_args(py_type)
            # NoneType (type(None)) が含まれているか確認
            none_type = type(None)
            if none_type in args:
                nullable = True
                # None 以外の実際の型を抽出
                real_types = [a for a in args if a is not none_type]
                if real_types:
                    py_type = real_types[0]
            else:
                py_type = args[0]

        # デフォルト値チェック (Nullable の補完)
        if hasattr(info, "default") and info.default is None:
            nullable = True
        elif not info.is_required():
            nullable = True

        pa_type = _PYTHON_TO_PYARROW.get(py_type, pa.string())
        fields.append(pa.field(name, pa_type, nullable=nullable))

    return pa.schema(fields)


# --- 事前構築済みスキーマ定数 (モジュールロード時に1回だけ導出) ---
SCHEMA_CATALOG = pydantic_to_pyarrow(CatalogRecord)
SCHEMA_MASTER = pydantic_to_pyarrow(StockMasterRecord)
SCHEMA_LISTING = pydantic_to_pyarrow(ListingEvent)
SCHEMA_NAME = pydantic_to_pyarrow(NameEvent)
SCHEMA_INDEX = pydantic_to_pyarrow(IndexEvent)
SCHEMA_FINANCIAL = pydantic_to_pyarrow(FinancialValueRecord)
SCHEMA_TEXT = pydantic_to_pyarrow(QualitativeTextRecord)

# キーベースのレジストリ (hf_storage / delta_manager が参照)
ARIA_SCHEMAS = {
    "catalog": SCHEMA_CATALOG,
    "master": SCHEMA_MASTER,
    "listing": SCHEMA_LISTING,
    "name": SCHEMA_NAME,
    "jpx_definitions": pydantic_to_pyarrow(JpxDefinitionRecord),
    "indices": SCHEMA_INDEX,
    "financial_values": SCHEMA_FINANCIAL,
    "qualitative_text": SCHEMA_TEXT,
}
