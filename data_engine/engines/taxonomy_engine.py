"""
ARIA Taxonomy Mapping Engine

【このファイルの役割（日常語での説明）】
金融庁が公式に配布している3つのExcelファイル（1f, 1e, 1g）の中身を
「そのまま全部」読み取り、ARIAが使える辞書（taxonomy_mapping.parquet）に変換する。

辞書は以下の用途で使われる：
1. 「NetSales と Revenue は同じ売上高だ」と特定する（aria_kpi_key による統合）
2. 「現金預金は、流動資産の下位にある」と知る（depth, abstract による階層構造）
3. 「この項目は金額だ／テキストだ」と判別する（type による分類）
4. 「この項目は借方か貸方か」を知る（balance による貸借区分）

【設計原則】
- 公式Excelの列を1つも捨てない（全列取得）
- taxonomy_urls.json（タクソノミZIP用）には一切触れない（完全に独立）
- 辞書用URLは taxonomy_mapping_urls.json で独立管理（予測への依存を排除）
- 既存の Parquet がある場合は上書きせずマージ（過去データの消失を防ぐ）
- 全年度分を自動マージし、業種変更や要素追加に対する取りこぼしをゼロにする
"""

import io
import json
import os
from typing import Dict, Optional

import pandas as pd
import requests

# --- パス設定 ---
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAXONOMY_MAPPING_URLS_PATH = os.path.join(
    _base_dir, "core", "taxonomy_mapping_urls.json"
)
META_DIR = os.path.join(_base_dir, "..", "master", "meta")
MAPPING_PARQUET_PATH = os.path.join(META_DIR, "taxonomy_mapping.parquet")

# --- ARIA Canonical KPI Mapping（橋渡しルール）---
# 業種やGAAPによって名前が違うが「同じ経済的意味」を持つ項目を、
# aria_* という共通の名前に統一するための辞書。
#
# 【設計原則】
# - aria_* はスクリーニング専用の「有限集合」であり、全要素を統一する目的ではない
# - ここに無い要素は aria_kpi_key = element_name のまま残り、銘柄詳細ページの表示に使われる
# - 冗長性の回避: 同じ経済的意味を持つ要素のみを1つの aria_* に統合する
#   会計上の定義が異なるもの（純資産 vs 株主資本）は別々の aria_* を維持する
CORE_KPI_MAPPING: Dict[str, str] = {
    # =============================================
    # 損益計算書（P/L）
    # =============================================
    # --- 売上高（トップライン）---
    # 企業の主たる営業活動による収益。業種や会計基準によって名称が大きく異なるため、
    # aria_net_sales という単一のキーに統合し、クロスGAAP・クロス業種スクリーニングを可能にする。
    "NetSales": "aria_net_sales",  # J-GAAP 一般商工業: 売上高
    "NetSalesOfCompletedConstructionContracts": "aria_net_sales",  # J-GAAP 建設業: 完成工事高
    "OrdinaryIncomeBNK": "aria_net_sales",  # J-GAAP 銀行: 経常収益（トップラインに相当）
    "OrdinaryIncomeINS": "aria_net_sales",  # J-GAAP 保険: 経常収益
    "OperatingRevenue": "aria_net_sales",  # J-GAAP 電気通信、鉄道等: 営業収益
    "OperatingRevenue1": "aria_net_sales",  # J-GAAP 海運: 営業収益
    "Revenue": "aria_net_sales",  # IFRS: 収益
    "Revenues": "aria_net_sales",  # US-GAAP: Revenues
    
    # --- 売上原価 ---
    # 売上高に対応する直接的な原価。
    "CostOfSales": "aria_cost_of_sales",  # J-GAAP 一般等 / IFRS: 売上原価
    "CostOfSalesOfCompletedConstructionContracts": "aria_cost_of_sales",  # J-GAAP 建設業: 完成工事原価
    
    # --- 売上総利益（粗利） ---
    # 売上高から売上原価を差し引いた利益。
    "GrossProfit": "aria_gross_profit",  # J-GAAP 一般等 / IFRS: 売上総利益
    "GrossProfitOnCompletedConstructionContracts": "aria_gross_profit",  # J-GAAP 建設業: 完成工事総利益
    
    # --- 販売費及び一般管理費（販管費） ---
    # 営業活動に間接的にかかる費用（人件費、広告宣伝費など）。
    "SellingGeneralAndAdministrativeExpenses": "aria_sga",  # J-GAAP: 販売費及び一般管理費
    
    # --- 営業利益 ---
    # 主たる営業活動から得られた利益。
    "OperatingIncome": "aria_operating_income",  # J-GAAP: 営業利益
    "OperatingProfitLoss": "aria_operating_income",  # IFRS: 営業利益（厳密にはIFRSで定義されない場合もあるが、企業が独自開示する要素）
    
    # --- 経常利益（J-GAAP特有概念） ---
    # 営業利益に営業外損益を加減した利益。IFRSおよびUS-GAAPには存在しない概念のため、
    # クロスGAAP比較には不向きだが、J-GAAP企業の単独分析や慣習的な指標として必須。
    "OrdinaryIncome": "aria_ordinary_income",  # J-GAAP: 経常利益
    
    # --- 営業外損益（J-GAAP特有概念） ---
    "NonOperatingIncome": "aria_non_operating_income",  # J-GAAP: 営業外収益（受取利息、配当金など）
    "NonOperatingExpenses": "aria_non_operating_expenses",  # J-GAAP: 営業外費用（支払利息など）
    
    # --- 特別損益（J-GAAP特有概念） ---
    # 突発的な事象による一時的な損益。IFRSでは特別項目という区分は禁止されている。
    "ExtraordinaryIncome": "aria_extraordinary_income",  # J-GAAP: 特別利益（固定資産売却益など）
    "ExtraordinaryLoss": "aria_extraordinary_loss",  # J-GAAP: 特別損失（減損損失など）
    
    # --- 税引前当期純利益 ---
    # 法人税等を差し引く前の利益。全会計基準で比較可能な基準利益。
    "IncomeBeforeIncomeTaxes": "aria_income_before_tax",  # J-GAAP 一般: 税引前当期純利益
    "ProfitLossBeforeTax": "aria_income_before_tax",  # IFRS: 税引前利益
    "IncomeBeforeIncomeTaxesBNK": "aria_income_before_tax",  # J-GAAP 銀行: 税引前当期純利益
    "IncomeBeforeIncomeTaxesINS": "aria_income_before_tax",  # J-GAAP 保険: 税引前当期純利益
    
    # --- 法人税等（P/L補完） ---
    # 所得に対する税金費用。実効税率（aria_computed_effective_tax_rate）の算出に必要。
    "IncomeTaxes": "aria_income_taxes",  # J-GAAP: 法人税、住民税及び事業税等（合計）
    "IncomeTaxExpenseIFRS": "aria_income_taxes",  # IFRS: 法人所得税費用
    
    # --- 金融収益・費用（IFRS特有） ---
    # IFRSにおける財務活動関連の損益。
    "FinanceIncomeIFRS": "aria_finance_income",  # IFRS: 金融収益
    "FinanceCostsIFRS": "aria_finance_costs",  # IFRS: 金融費用
    
    # --- 当期純利益 ---
    # 企業の最終的な利益（親会社株主に帰属する利益）。
    "ProfitLoss": "aria_net_income",  # J-GAAP / IFRS: 当期純利益
    "ProfitLossAttributableToOwnersOfParent": "aria_net_income",  # J-GAAP 連結: 親会社株主に帰属する当期純利益
    
    # --- 包括利益 ---
    # 当期純利益にその他の包括利益（為替換算調整勘定、有価証券評価差額など）を加えたもの。
    "ComprehensiveIncome": "aria_comprehensive_income",  # J-GAAP / IFRS: 包括利益
    
    # =============================================
    # 貸借対照表（B/S）
    # =============================================
    # --- 資産 ---
    "Assets": "aria_total_assets",  # J-GAAP / IFRS: 総資産
    "CurrentAssets": "aria_current_assets",  # J-GAAP / IFRS: 流動資産（1年以内に現金化予定の資産）
    "NoncurrentAssets": "aria_noncurrent_assets",  # J-GAAP / IFRS: 固定資産（非流動資産）
    
    # --- 負債 ---
    "Liabilities": "aria_total_liabilities",  # J-GAAP / IFRS: 負債合計
    "CurrentLiabilities": "aria_current_liabilities",  # J-GAAP / IFRS: 流動負債（1年以内に返済予定の負債）
    "NoncurrentLiabilities": "aria_noncurrent_liabilities",  # J-GAAP / IFRS: 固定負債（非流動負債）
    
    # --- 運転資本（Working Capital）構成要素 ---
    # 企業の短期的な資金繰り（キャッシュコンバージョンサイクル）を分析・計算するために必要な要素。
    "Inventories": "aria_inventories",  # J-GAAP: 棚卸資産（在庫）
    "InventoriesCAIFRS": "aria_inventories",  # IFRS: 棚卸資産
    "NotesAndAccountsReceivableTradeAndContractAssets": "aria_trade_receivables",  # J-GAAP: 受取手形、売掛金及び契約資産
    "AccountsReceivableTrade": "aria_trade_receivables",  # J-GAAP: 売掛金
    "NotesAndAccountsReceivableTrade": "aria_trade_receivables",  # J-GAAP: 受取手形及び売掛金
    "TradeAndOtherReceivablesCAIFRS": "aria_trade_receivables",  # IFRS: 営業債権及びその他の債権
    "NotesAndAccountsPayableTrade": "aria_trade_payables",  # J-GAAP: 支払手形及び買掛金
    "AccountsPayableTrade": "aria_trade_payables",  # J-GAAP: 買掛金
    "TradeAndOtherPayablesCLIFRS": "aria_trade_payables",  # IFRS: 営業債務及びその他の債務
    
    # --- 固定資産の内訳 ---
    "PropertyPlantAndEquipment": "aria_ppe",  # J-GAAP: 有形固定資産（土地、建物、機械など）
    "PropertyPlantAndEquipmentIFRS": "aria_ppe",  # IFRS: 有形固定資産
    "IntangibleAssets": "aria_intangible_assets",  # J-GAAP: 無形固定資産（ソフトウェアなど。のれんを含む場合あり）
    "IntangibleAssetsIFRS": "aria_intangible_assets",  # IFRS: 無形資産
    "Goodwill": "aria_goodwill",  # J-GAAP: のれん（買収時のプレミアム）
    "GoodwillIFRS": "aria_goodwill",  # IFRS: のれん
    "InvestmentSecurities": "aria_investment_securities",  # J-GAAP: 投資有価証券（持合株式など）
    
    # --- 純資産・資本 ---
    # 会計基準により「純資産（NetAssets）」と「帰属持分（Equity）」の概念が完全には一致しない。
    # aria_net_assets: 非支配持分(NCI)を【含む】全体の純資産。PBRの分母。
    # aria_owners_equity: 非支配持分(NCI)を【除く】親会社帰属分。ROEの分母。
    "NetAssets": "aria_net_assets",  # J-GAAP: 純資産合計（NCI含む）
    "Equity": "aria_net_assets",  # IFRS: 資本合計（NCI含む）
    "ShareholdersEquity": "aria_owners_equity",  # J-GAAP: 株主資本（NCIを除く。ただし自己株式は控除）
    "EquityAttributableToOwnersOfParent": "aria_owners_equity",  # IFRS: 親会社の所有者に帰属する持分（NCI除く）
    "CapitalStock": "aria_capital_stock",  # J-GAAP: 資本金
    "RetainedEarnings": "aria_retained_earnings",  # J-GAAP: 利益剰余金（内部留保）
    "RetainedEarningsIFRS": "aria_retained_earnings",  # IFRS: 利益剰余金
    "TreasuryStock": "aria_treasury_stock",  # J-GAAP: 自己株式（マイナス項目）
    "TreasuryShares": "aria_treasury_stock",  # IFRS: 自己株式（1g 要素リスト非掲載名称）
    "TreasurySharesIFRS": "aria_treasury_stock",  # IFRS: 自己株式（1g 要素リスト掲載名称）
    "NonControllingInterests": "aria_nci",  # J-GAAP: 非支配株主持分（旧:少数株主持分）
    "NonControllingInterestsIFRS": "aria_nci",  # IFRS: 非支配持分
    
    # --- 有利子負債の内訳（J-GAAP は合算要素が存在しないため個別取得） ---
    "ShortTermLoansPayable": "aria_short_term_loans",  # J-GAAP: 短期借入金
    "LongTermLoansPayable": "aria_long_term_loans",  # J-GAAP: 長期借入金
    "BondsPayable": "aria_bonds_payable",  # J-GAAP: 社債
    "CommercialPapersLiabilities": "aria_commercial_paper",  # J-GAAP: コマーシャル・ペーパー
    "ShortTermBondsPayable": "aria_short_term_bonds",  # J-GAAP: 短期社債
    
    # --- 有利子負債（IFRS は合算要素が存在）---
    "BondsAndBorrowingsLiabilitiesIFRS": "aria_borrowings_ifrs",  # IFRS: 社債及び借入金
    
    # =============================================
    # キャッシュフロー計算書（C/F）
    # =============================================
    "NetCashProvidedByUsedInOperatingActivities": "aria_cf_operating",  # J-GAAP: 営業活動によるキャッシュ・フロー
    "CashFlowsFromUsedInOperatingActivities": "aria_cf_operating",  # IFRS: 営業活動によるキャッシュ・フロー
    "NetCashProvidedByUsedInInvestmentActivities": "aria_cf_investing",  # J-GAAP: 投資活動によるキャッシュ・フロー
    "CashFlowsFromUsedInInvestingActivities": "aria_cf_investing",  # IFRS: 投資活動によるキャッシュ・フロー
    "NetCashProvidedByUsedInFinancingActivities": "aria_cf_financing",  # J-GAAP: 財務活動によるキャッシュ・フロー
    "CashFlowsFromUsedInFinancingActivities": "aria_cf_financing",  # IFRS: 財務活動によるキャッシュ・フロー
    "CashAndCashEquivalents": "aria_cash_equivalents",  # J-GAAP / IFRS: 現金及び現金同等物の残高（期末）
    "CashAndDeposits": "aria_cash_and_deposits",  # J-GAAP: 現金及び預金（B/S科目だが資金流動性の指標として利用）
    
    # =============================================
    # 1株あたり指標
    # =============================================
    "BasicEarningsLossPerShare": "aria_eps",  # J-GAAP / IFRS: 1株当たり当期純利益（EPS）
    "EarningsPerShare": "aria_eps",  # US-GAAP: Earnings Per Share
    "NetAssetsPerShare": "aria_bps",  # J-GAAP: 1株当たり純資産額（BPS）
    "BookValuePerShare": "aria_bps",  # US-GAAP: Book Value Per Share
    "DividendPaidPerShare": "aria_dividend_per_share",  # J-GAAP: 1株当たり配当額
    "DividendsPerShare": "aria_dividend_per_share",  # US-GAAP: Dividends Per Share
    "DividendsPaid": "aria_dividend_total",  # J-GAAP: 配当金支払額（総額）
    
    # =============================================
    # その他重要指標
    # =============================================
    "NumberOfEmployees": "aria_employees",  # 従業員数
    "TotalNumberOfIssuedShares": "aria_shares_issued",  # 発行済株式総数
    "IssuedSharesTotalNumberOfSharesOutstanding": "aria_shares_issued",  # 発行済株式総数（代替名）
    "ResearchAndDevelopmentExpenses": "aria_rnd_expenses",  # 研究開発費
    "DepreciationAndAmortization": "aria_depreciation",  # 一般的な減価償却費
    "DepreciationAndAmortizationSGA": "aria_depreciation",  # 販管費内の減価償却費
    "DepreciationOfPropertyPlantAndEquipment": "aria_depreciation",  # 有形固定資産の減価償却費
    "DepreciationAndAmortizationOpeCF": "aria_depreciation",  # C/F計算書上の減価償却費
    "PurchaseOfPropertyPlantAndEquipmentAndIntangibleAssets": "aria_capex",  # 有形及び無形固定資産の取得による支出（CAPEX）
    "PurchaseOfPropertyPlantAndEquipment": "aria_capex",  # 有形固定資産の取得による支出（CAPEX）
    "InterestExpenses": "aria_interest_expense",  # J-GAAP: 支払利息
    "InterestExpense": "aria_interest_expense",  # IFRS: 支払利息
    "InterestIncome": "aria_interest_income",  # 受取利息
    
    # =============================================
    # テキストブロック（qualitative_text 用）
    # =============================================
    "BusinessRisksTextBlock": "aria_text_business_risks",  # 事業等のリスク
    "ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock": "aria_text_mda",  # MD&A（経営者による分析）
    "DividendPolicyAndDividendsDeclaredTextBlock": "aria_text_dividend_policy",  # 配当政策
}

# --- 公式Excelの列名を、ARIA統一列名にマッピングする辞書 ---
# 3つのExcelはそれぞれ微妙に列名が異なる。この辞書で吸収する。
_COLUMN_ALIASES: Dict[str, list] = {
    # ARIA列名: [Excelで使われうる列名の候補リスト]
    "label_jp": ["標準ラベル（日本語）", "様式ツリー-標準ラベル（日本語）"],
    "label_jp_verbose": ["冗長ラベル（日本語）"],
    "label_en": ["標準ラベル（英語）"],
    "label_en_verbose": ["冗長ラベル（英語）"],
    "namespace": ["名前空間プレフィックス"],
    "element_name": ["要素名"],
    "xbrl_type": ["type"],
    "substitution_group": ["substitutionGroup"],
    "period_type": ["periodType"],
    "balance": ["balance"],
    "abstract": ["abstract"],
    "depth": ["depth"],
    "category_code": ["科目分類"],
    "reference_link": ["参照リンク"],
    "usage_label_jp": [
        "用途区分、財務諸表区分及び業種区分のラベル（日本語）",
        "用途別ラベル及び代替ラベル（日本語）",
    ],
    "usage_label_en": [
        "用途区分、財務諸表区分及び業種区分のラベル（英語）",
        "用途別ラベル及び代替ラベル（英語）",
    ],
    "documentation_jp": ["documentationラベル（日本語）"],
    "documentation_en": ["documentationラベル（英語）"],
}


class TaxonomyMappingEngine:
    """金融庁の公式Excelからtaxonomy_mapping.parquetを生成するエンジン。

    全年度のExcelを自動マージし、要素や業種の追加・変更（将来のシート名変更を含む）
    に対する取りこぼしをゼロにする。
    """

    def __init__(self, target_years: Optional[list[str]] = None):
        """初期化。

        Args:
            target_years: 処理対象の年度リスト。None の場合は設定ファイルの全年度を処理。
        """
        self.all_sources = self._load_mapping_urls()
        if target_years:
            self.target_years = [
                y for y in target_years if y in self.all_sources
            ]
        else:
            self.target_years = sorted(self.all_sources.keys())

    @staticmethod
    def _load_mapping_urls() -> Dict[str, Dict[str, str]]:
        """taxonomy_mapping_urls.json からExcel辞書用のURLを読み込む。

        taxonomy_urls.json（タクソノミZIP）とは完全に独立した設定ファイル。
        """
        with open(TAXONOMY_MAPPING_URLS_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("sources", {})

    def _download_excel(self, url: str) -> pd.ExcelFile:
        """URLからExcelをダウンロードし、メモリ上で開く。"""
        print(f"  ダウンロード中: {url}")
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        return pd.ExcelFile(io.BytesIO(res.content))

    def _resolve_column(self, df_columns: list, aria_name: str) -> Optional[str]:
        """Excelの列名ゆらぎを吸収し、ARIA統一名に変換する。見つからなければNone。"""
        candidates = _COLUMN_ALIASES.get(aria_name, [])
        for candidate in candidates:
            if candidate in df_columns:
                return candidate
        return None

    def _extract_all_columns(
        self, xl: pd.ExcelFile, source_category: str
    ) -> pd.DataFrame:
        """Excelの全シートから、公式提供の全列を取得する。

        Args:
            xl: 読み込み済みのExcelファイル
            source_category: 'J-GAAP', 'DEI_Qualitative', 'IFRS' などの分類名
        """
        # 説明ページは除外
        ignore_sheets = [
            "目次",
            "勘定科目リストについて",
            "タクソノミ要素リストについて",
            "IFRSタクソノミ要素リストについて",
            "国際会計基準タクソノミ要素リストについて",
        ]
        sheets = [s for s in xl.sheet_names if s not in ignore_sheets]

        all_rows = []
        for sheet in sheets:
            try:
                df = xl.parse(sheet, skiprows=1)
                if df.empty:
                    continue

                # 各ARIA統一列名に対応するExcel列を探す
                row_data = {}
                for aria_name in _COLUMN_ALIASES:
                    excel_col = self._resolve_column(list(df.columns), aria_name)
                    if excel_col:
                        row_data[aria_name] = df[excel_col]
                    else:
                        row_data[aria_name] = pd.Series(
                            [None] * len(df), dtype="object"
                        )

                extract_df = pd.DataFrame(row_data)

                # --- 1e 固有の欠陥修正 ---
                # 1e_ElementList では label_jp に相当する列が2つに分裂している：
                #   「様式ツリー-標準ラベル（日本語）」→ 見出しレベルの行に値が入る
                #   「詳細ツリー-標準ラベル（日本語）」→ 実項目の行に値が入る
                # どちらか片方にしか値が入らないため、両方を合体させて label_jp を構築する。
                detail_col = "詳細ツリー-標準ラベル（日本語）"
                if detail_col in df.columns:
                    # 様式ツリー側が NaN の行を、詳細ツリー側の値で埋める
                    extract_df["label_jp"] = extract_df["label_jp"].fillna(
                        df[detail_col]
                    )

                extract_df["source_category"] = source_category
                extract_df["industry_or_sheet"] = sheet
                all_rows.append(extract_df)

            except Exception as e:
                print(f"  シート '{sheet}' のパースをスキップ: {e}")


        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    def generate_mapping_dataframe(self) -> pd.DataFrame:
        """設定された全年度のExcelから全データを抽出し、1つのDataFrameに統合する。

        全年度をマージすることで、過去に存在した要素や業種が消えても
        辞書から消失しない（取りこぼしゼロ）。
        """
        print("=== ARIA Taxonomy Mapping 生成開始 ===")
        print(f"  対象年度: {self.target_years}")
        all_dfs = []

        for year in self.target_years:
            urls = self.all_sources[year]
            print(f"\n--- {year}年度 ---")

            # 各年度内の3ファイル処理
            file_configs = [
                ("1f", "J-GAAP"),
                ("1e", "DEI_Qualitative"),
                ("1g", "IFRS"),
            ]
            for file_key, source_cat in file_configs:
                url = urls.get(file_key)
                if not url:
                    continue
                try:
                    xl = self._download_excel(url)
                    df = self._extract_all_columns(xl, source_cat)
                    # 年度情報を付与（業種変更の追跡用）
                    df["taxonomy_year"] = year
                    print(
                        f"  {file_key} ({source_cat}): {len(df)} 行を抽出"
                    )
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  {file_key} ({year}) の処理に失敗: {e}")

        if not all_dfs:
            raise RuntimeError("すべてのExcelの処理に失敗しました。")

        final_df = pd.concat(all_dfs, ignore_index=True)

        # 欠損値の除去（namespace と element_name が無い行は無意味）
        final_df.dropna(subset=["namespace", "element_name"], inplace=True)

        # aria_kpi_key の付与（橋渡しルール適用。無い項目はそのままの要素名を採用）
        final_df["aria_kpi_key"] = final_df["element_name"].apply(
            lambda x: CORE_KPI_MAPPING.get(x, x)
        )

        # --- 橋渡しルールの完全性保証 ---
        # CORE_KPI_MAPPING に登録されているがExcelに存在しない要素を、
        # 独立した行として辞書に注入する。
        # これがないと、US-GAAP(Revenues)、IFRS(OperatingProfitLoss)、
        # 建設業(NetSalesOfCompletedConstructionContracts) 等がJOINで
        # aria_kpi_key = NULL となり、クロスGAAPスクリーニングが不可能になる。
        existing_elements = set(final_df["element_name"].unique())
        missing_entries = []
        for elem, aria_key in CORE_KPI_MAPPING.items():
            if elem not in existing_elements:
                missing_entries.append(
                    {
                        "element_name": elem,
                        "aria_kpi_key": aria_key,
                        "namespace": "aria_bridge",  # Excel非掲載の橋渡し行であることを明示
                        "source_category": "CORE_KPI_BRIDGE",
                        "industry_or_sheet": "cross_gaap",
                        "taxonomy_year": "bridge",
                    }
                )
        if missing_entries:
            bridge_df = pd.DataFrame(missing_entries)
            final_df = pd.concat([final_df, bridge_df], ignore_index=True)
            print(
                f"  橋渡し注入: CORE_KPI_MAPPING から {len(missing_entries)} 件の"
                f"Excel非掲載要素を辞書に追加"
            )

        # --- 型の正規化（Excelのシートによって bool/str が混在するため統一）---
        # abstract: True/False/'true'/'false' → bool型に統一
        final_df["abstract"] = (
            final_df["abstract"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "nan": None, "none": None})
        )
        # depth: float/int混在 → Int64（pandas nullable integer）に統一
        final_df["depth"] = pd.to_numeric(
            final_df["depth"], errors="coerce"
        ).astype("Int64")

        # --- 年度間で同一要素の重複を除去（最新年度を優先）---
        # 同じ (namespace, element_name, industry_or_sheet) が複数年度に存在する場合、
        # 最新年度（last）の情報を採用する。これにより業種名が変更された場合も
        # 旧名・新名の両方が辞書に残り、取りこぼしがゼロになる。
        final_df.sort_values("taxonomy_year", inplace=True)
        final_df.drop_duplicates(
            subset=["namespace", "element_name", "industry_or_sheet"],
            keep="last",
            inplace=True,
        )

        print(f"\n  統合完了: {len(self.target_years)}年度 → 全 {len(final_df)} 行")
        return final_df

    def upsert_to_parquet(self, new_df: pd.DataFrame):
        """既存のParquetを破壊せず、マージ（追記＋重複更新）する。"""
        os.makedirs(META_DIR, exist_ok=True)

        if os.path.exists(MAPPING_PARQUET_PATH):
            print("  既存 Parquet を検出。マージを実行...")
            existing_df = pd.read_parquet(MAPPING_PARQUET_PATH)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.sort_values("taxonomy_year", inplace=True)
            combined.drop_duplicates(
                subset=["namespace", "element_name", "industry_or_sheet"],
                keep="last",
                inplace=True,
            )
            combined.to_parquet(MAPPING_PARQUET_PATH, index=False)
            print(f"  マージ完了: 全 {len(combined)} 行")
        else:
            print(f"  新規 Parquet を作成: {MAPPING_PARQUET_PATH}")
            new_df.to_parquet(MAPPING_PARQUET_PATH, index=False)
            print(f"  作成完了: 全 {len(new_df)} 行")


if __name__ == "__main__":
    engine = TaxonomyMappingEngine()
    df = engine.generate_mapping_dataframe()
    engine.upsert_to_parquet(df)

