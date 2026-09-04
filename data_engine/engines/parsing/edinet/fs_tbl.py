"""
財務情報をまとめてpandas.DataFrameで出力するモジュール
"""


from __future__ import annotations
import xml.etree.ElementTree as ET
import re
from zipfile import ZipFile
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series



from datetime import datetime, timedelta, date
from pydantic import BaseModel, Field
from time import sleep
from typing import Literal
import json
from typing import Annotated
from pydantic.functional_validators import BeforeValidator

from loguru import logger
from .xbrl_parser_wrapper import *
from .link_base_file_analyzer import *
from .utils import *



# FsDataDf の整数カラム一覧（None補完時に型を維持するために使用）
_FSDATA_INT_COLS = frozenset([
    'isTextBlock_flg', 'abstract_flg',
    'current_flg', 'non_consolidated_flg', 'prior_flg',
])

class FsDataDf(pa.DataFrameModel):
    #ParentChildLink
    """
    'key': taxonomy like 'jpcrp_cor:NetSales'
    'data_str': data (string) like '1000000'
    'decimals': 3桁の表示
    'precision': ???
    'context_ref': # T:-3, M:-6, B:-9
    'element_name':
    'unit': # JPY
    'period_type':
    'isTextBlock_flg':
    'abstract_flg':
    'period_start': # durationの場合 当期末日, instantの場合 None
    'period_end': # durationの場合 当期末日, instantの場合 当期末日
    'instant_date': # durationの場合 None, instantの場合 当期末日
    'end_date_pv': # durationの場合 前期末日, instantの場合 None
    'instant_date_pv': # durationの場合 None, instantの場合 前期対象日
    'scenario':# シナリオ
    'role': #
    'label_jp':
    'label_jp_long':
    'label_en':
    'label_en_long':    
    'order':
    'child_key':
    'docid':
    """
    # 1. Identity (誰のデータか)
    docid: Series[str] = pa.Field(nullable=True)
    # 2. Core Data (何の値か)
    key: Series[str] = pa.Field(nullable=True)
    label_jp: Series[str] = pa.Field(nullable=True)
    data_str: Series[str] = pa.Field(nullable=True)
    unit: Series[str] = pa.Field(nullable=True)
    # 3. Pivot & Filter Flags (比較・抽出軸)
    current_flg: Series[int] = pa.Field(isin=[0,1],nullable=True)
    non_consolidated_flg: Series[int] = pa.Field(isin=[0,1],nullable=True)
    prior_flg: Series[int] = pa.Field(isin=[0,1],nullable=True)
    # 4. Context (いつの状態か)
    period_start: Series[str] = pa.Field(nullable=True)
    period_end: Series[str] = pa.Field(nullable=True)
    instant_date: Series[str] = pa.Field(nullable=True)
    end_date_pv: Series[str] = pa.Field(nullable=True)
    instant_date_pv: Series[str] = pa.Field(nullable=True)
    # 5. Taxonomy Details (詳細属性)
    role: Series[str] = pa.Field(nullable=True)
    label_en: Series[str] = pa.Field(nullable=True)
    label_jp_long: Series[str] = pa.Field(nullable=True)
    label_en_long: Series[str] = pa.Field(nullable=True)
    AccountingStandardsDEI: Series[str] = pa.Field(nullable=True)
    # 6. Technical Metadata (システム・XBRL用)
    isTextBlock_flg: Series[int] = pa.Field(isin=[0,1],nullable=True)
    abstract_flg: Series[int] = pa.Field(isin=[0,1],nullable=True)
    decimals: Series[str] = pa.Field(nullable=True)
    precision: Series[str] = pa.Field(nullable=True)
    context_ref: Series[str] = pa.Field(nullable=True)
    element_name: Series[str] = pa.Field(nullable=True)
    period_type: Series[str] = pa.Field(isin=['instant','duration'],nullable=True)
    scenario: Series[str] = pa.Field(nullable=True)
    order: Series[float] = pa.Field(nullable=True)

    class Config:
        coerce = True  # object -> int64 等の暗黙的型変換を許可（安全ネット）



def get_fs_tbl(account_list_common_obj,docid:str,zip_file_str:str,temp_path_str:str,role_keyward_list:list,doc_type:str='public')->FsDataDf:
    linkbasefile_obj = linkbasefile(
        zip_file_str=zip_file_str,
        temp_path_str=temp_path_str,
        doc_type=doc_type
        )
    linkbasefile_obj.read_linkbase_file()
    linkbasefile_obj.check()
    linkbasefile_obj.make_account_label(account_list_common_obj,role_keyward_list)
    xbrl_data_df,log_dict = get_xbrl_wrapper(
        docid=docid,
        zip_file=zip_file_str,
        temp_dir=Path(temp_path_str),
        out_path=Path(temp_path_str),
        update_flg=False
        )
    data_list = []
    matched_records_sum = 0
    # 【Zero-Drop Architecture】プレゼンテーション・リンクベースにマッチした全キーを追跡する
    matched_keys = set()

    for role in list(linkbasefile_obj.account_tbl_role_dict.keys()):
        key_in_the_role:pd.Series = linkbasefile_obj.account_tbl_role_dict[role].key
        data=pd.merge(
            xbrl_data_df.query("key in @key_in_the_role"),
            linkbasefile_obj.account_tbl_role_dict[role],
            on='key',
            how='left')
        data = data.assign(docid=docid,role=role)
        
        data = data.assign(
            non_consolidated_flg=data.context_ref.str.contains('NonConsolidated').astype(int),
            current_flg=data.context_ref.str.contains('CurrentYear').astype(int),
            prior_flg=data.context_ref.str.contains('Prior1Year').astype(int)
        )

        # 【Arelle直接ラベル補完】リンクベース由来ラベルがNULLの場合、
        # Arelle直接ラベル（タクソノミ参照チェーンから自動解決）で補完する。
        if 'arelle_label_jp' in data.columns:
            data['label_jp'] = data['label_jp'].fillna(data['arelle_label_jp'])
            data['label_en'] = data['label_en'].fillna(data['arelle_label_en'])
            data['label_jp_long'] = data['label_jp_long'].fillna(data['arelle_label_jp_long'])
            data['label_en_long'] = data['label_en_long'].fillna(data['arelle_label_en_long'])

        matched_keys.update(key_in_the_role.tolist())
        matched_records_sum += len(data)
        data_list.append(data)

    # 【Zero-Drop】Arelleが抽出した全ファクトのうち、プレゼンテーション・リンクベースに
    # マッチしなかった「孤立ファクト (Unlinked Facts)」を差集合として算出し、救出する。
    unlinked_records_count = 0
    if not xbrl_data_df.empty:
        all_arelle_keys = set(xbrl_data_df['key'].unique())
        unlinked_keys = all_arelle_keys - matched_keys

        if unlinked_keys:
            unlinked_df = xbrl_data_df[xbrl_data_df['key'].isin(unlinked_keys)].copy()
            # Unlinked Facts: Arelle直接ラベルを主要ソースとして使用
            unlinked_df = unlinked_df.assign(
                docid=docid,
                role=None,          # 物理的事実: プレゼンテーション・リンクベースに未登録
                label_jp=unlinked_df.get('arelle_label_jp'),      # Arelle直接ラベルを採用
                label_jp_long=unlinked_df.get('arelle_label_jp_long'),
                label_en=unlinked_df.get('arelle_label_en'),
                label_en_long=unlinked_df.get('arelle_label_en_long'),
                order=float('nan'),  # 物理的事実: 表示順序が定義されていない（float64 NaN）
                non_consolidated_flg=unlinked_df.context_ref.str.contains('NonConsolidated').astype(int),
                current_flg=unlinked_df.context_ref.str.contains('CurrentYear').astype(int),
                prior_flg=unlinked_df.context_ref.str.contains('Prior1Year').astype(int),
            )
            unlinked_records_count = len(unlinked_df)
            data_list.append(unlinked_df)
            # logger.info(f"Zero-Drop: {len(unlinked_keys)} 個の孤立キーから {unlinked_records_count} 件の Unlinked Facts を救出しました。")

    if not data_list:
        return pd.DataFrame() # 空のDFを返して後続で適切に処理
    
    # 最終結合: FsDataDf スキーマに存在するカラムのみを選択して返す
    target_cols = get_columns_df(FsDataDf)
    merged = pd.concat(data_list, ignore_index=True)
    
    # 【監査用メタデータ】展開後の理論件数を付加
    merged.attrs['aria_metrics'] = {
        'matched_records_sum': matched_records_sum,
        'unlinked_records_count': unlinked_records_count,
        'theoretical_total': matched_records_sum + unlinked_records_count
    }
    # スキーマに存在するがデータに存在しないカラムは None で補完する
    for col in target_cols:
        if col not in merged.columns:
            merged[col] = None

    # 【根本原因修正】pd.merge / pd.concat が整数カラムに NaN を注入すると、
    # pandas は int64 → float64 または object に自動アップキャストする。
    # Pandera の int64 バリデーションはこれを拒絶するため、
    # バリデーション直前に全フラグカラムを fillna(0).astype(int) で明示的にキャストする。
    # 意味論的にも正しい: フラグの欠損は「該当なし = 0」である。
    for col in _FSDATA_INT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # 【ラベル品質ゲート】Arelle直接ラベル注入後のラベル解決率を監査する。
    # IFRSタクソノミのダウンロード失敗等でラベルが劣化した場合を検出する。
    total = len(merged)
    if total > 0:
        label_jp_resolved = merged['label_jp'].notna().sum()
        label_rate = label_jp_resolved / total
        if label_rate < 0.5:
            logger.warning(
                f"⚠ Label quality degradation detected for docid={docid}: "
                f"label_jp resolved {label_jp_resolved}/{total} ({label_rate:.1%}). "
                f"Possible cause: IFRS taxonomy download failure or cache miss."
            )

    return FsDataDf(merged[target_cols])



class linkbasefile():
    def __init__(self,zip_file_str:str,temp_path_str:str,doc_type:str='public'):
        self.zip_file_str = zip_file_str
        self.temp_path_str = temp_path_str
        self.doc_type = doc_type
        self.log_dict = {}
    def read_linkbase_file(self):
        self.get_presentation_account_list_obj = get_presentation_account_list(
            zip_file_str=self.zip_file_str,
            temp_path_str=self.temp_path_str,
            doc_type=self.doc_type
        )
        self.parent_child_df = self.get_presentation_account_list_obj.export_parent_child_link_df()
        self.account_list = self.get_presentation_account_list_obj.export_account_list_df()
        self.log_dict = {**self.log_dict,**self.get_presentation_account_list_obj.export_log().model_dump()}
        
        self.get_calc_edge_list_obj = get_calc_edge_list(
            zip_file_str=self.zip_file_str,
            temp_path_str=self.temp_path_str
            )
        self.calc_edge_df = self.get_calc_edge_list_obj.export_parent_child_link_df()
        self.get_label_obj_jp = get_label(
            lang="Japanese",
            zip_file_str=self.zip_file_str,
            temp_path_str=self.temp_path_str,
            doc_type=self.doc_type
            )
        self.label_tbl_jp = self.get_label_obj_jp.export_label_tbl(label_to_taxonomi_dict=self.get_presentation_account_list_obj.export_label_to_taxonomi_dict())
        #self.log_dict={**self.log_dict,**self.get_label_obj_jp.export_log().model_dump()}
        self.get_label_obj_eng = get_label(
            lang="English",
            zip_file_str=self.zip_file_str,
            temp_path_str=self.temp_path_str,
            doc_type=self.doc_type
            )
        self.label_tbl_eng = self.get_label_obj_eng.export_label_tbl(label_to_taxonomi_dict=self.get_presentation_account_list_obj.export_label_to_taxonomi_dict())

    def check(self):
        p_key_set = set(self.parent_child_df.parent_key)
        c_key_set = set(self.parent_child_df.child_key)
        all_key_set = set(self.account_list.key)
        if len(p_key_set-all_key_set) != 0:
            logger.warning(f"parent key in arc-link that is not included in locator: \n{p_key_set-all_key_set}")
        if len(c_key_set-all_key_set) != 0:
            logger.warning(f"child key in arc-link that is not included in locator: \n{p_key_set-all_key_set}")
        if len(set(self.label_tbl_jp.key) - all_key_set) != 0:
            logger.warning(f"key in label that is not included in locator: \n{set(self.label_tbl_jp.key) - all_key_set}")

    def make_account_label(self,account_list_common_obj,role_list,role_label_list=[]):
        account_label_org = self.make_account_label_org()
        account_label_common = self.make_account_label_common(account_list_common_obj)
        account_label = pd.concat([account_label_org,account_label_common],axis=0)
        account_tbl = pd.merge(
            self.account_list[['key','role']],
            account_label[['label_jp','label_jp_long','label_en','label_en_long']],
            left_on='key',
            right_index=True,
            how='left')
        self.account_link_tracer_obj = account_link_tracer(self.parent_child_df)
        
        role_list_all = list(set(self.parent_child_df.role))
            
        if len(role_list)>0:
            # role_list が優先される
            role_list_f = []
            for role_t in role_list:
                role_list_f = role_list_f + [role_key for role_key in role_list_all if role_t in role_key]
        else:
            role_list_f = role_list_all
        account_tbl_role_dict = {}
        for role_text in role_list_f:
            role_suffix = role_text.split('/')[-1]
            account_tbl_of_the_role = account_tbl.query("role.str.contains(@role_suffix)").drop_duplicates()
            account_tbl_of_the_role = pd.merge(
                account_tbl_of_the_role,
                self.account_link_tracer_obj.get_child_order_recursive_list(
                    key_list=account_tbl_of_the_role.key.to_list(),
                    role=role_text
                )[['order','child_key']],
                left_on='key',
                right_on='child_key',
                how='left')
            account_tbl_of_the_role.order = account_tbl_of_the_role.order.fillna(1)
            account_tbl_of_the_role.sort_values('order')
            account_tbl_role_dict.update({role_suffix:account_tbl_of_the_role})
        self.account_tbl_role_dict = account_tbl_role_dict

    def make_account_label_org(self):
        df = self.label_tbl_jp.query("role == 'label'").set_index("key").rename(columns={"text": "label_jp"})
        df = df.join(
            [
                self.label_tbl_jp.query("role == 'verboseLabel'").set_index("key")[["text"]].rename(
                    columns={"text": "label_jp_long"}
                ),
                self.label_tbl_eng.query("role == 'label'").set_index("key")[["text"]].rename(
                    columns={"text": "label_en"}
                ),
                self.label_tbl_eng.query("role == 'verboseLabel'").set_index("key")[["text"]].rename(
                    columns={"text": "label_en_long"}
                ),
            ],
            how="left",
        )
        return df
    
    def make_summary_tbl(self):
        df = pd.DataFrame(index=list(set(self.account_list.key)))
        df = df.assign(
            is_parent=df.index.isin(self.parent_child_df.parent_key),
            is_child=df.index.isin(self.parent_child_df.child_key),
            is_calc_parent=df.index.isin(self.calc_edge_df.parent_key),
            is_calc_child=df.index.isin(self.calc_edge_df.child_key)
            )

    def detect_account_list_year(self):
        head_list = list(set(self.get_presentation_account_list_obj.export_account_list_df().schema_taxonomi_head))
        
        # 判定用マッピング (日付 -> 年代版)
        taxo_map = {
            "2025-11-01": "2026",
            "2024-11-01": "2025",
            "2023-12-01": "2024",
            "2022-11-01": "2023",
            "2021-11-01": "2022",
            "2020-11-01": "2021",
            "2020-03-31": "2020",
            "2019-11-01": "2020",
            "2019-03-31": "2019",
            "2019-02-28": "2019",
            "2018-03-31": "2018",
            "2018-02-28": "2018",
            "2017-03-31": "2017",
            "2017-02-28": "2017",
            "2016-03-31": "2016",
            "2016-02-29": "2016",
            "2015-03-31": "2015",
            "2014-03-31": "2014",
            "2013-08-31": "2014",
            # --- IFRS Namespaces ---
            "ifrs.org/taxonomy/2024-03-21": "2024",
            "ifrs.org/taxonomy/2023-03-23": "2023",
            "ifrs.org/taxonomy/2022-03-24": "2022",
            "ifrs.org/taxonomy/2021-03-24": "2021",
            "ifrs.org/taxonomy/2020-03-16": "2020",
            "ifrs.org/taxonomy/2019-03-27": "2019",
            "ifrs.org/taxonomy/2018-03-16": "2018",
            "ifrs.org/taxonomy/2017-03-09": "2017",
            "ifrs.org/taxonomy/2016-03-31": "2016",
            "ifrs.org/taxonomy/2015-03-11": "2015",
            "ifrs.org/taxonomy/2014-03-05": "2014",
        }

        # 優先順位: 開示府令用(jpcrp) > 財務諸表本表用(jppfs) > DEI(jpdei) > 特定目的会社用(jpsps) > IFRS
        # 毎年更新される可能性が高い順から探し、年代特定が可能な名前空間を採用する。
        targets = ["/taxonomy/jpcrp", "/taxonomy/jppfs", "/taxonomy/jpdei", "/taxonomy/jpsps", "ifrs.org", "/ifrs/"]
        
        for t in targets:
            # 優先度順に名前空間を走査
            candidate_heads = [h for h in head_list if t in h]
            for head in candidate_heads:
                for date_pattern, year_val in taxo_map.items():
                    if date_pattern in head:
                        self.account_list_year = year_val
                        return self.account_list_year
                        
        self.account_list_year = "-"
        return self.account_list_year
    
    def make_account_label_common(self,account_list_common_obj):
        self.detect_account_list_year()
        #self.account_list_common_obj = account_list_common_obj
        #label_to_taxonomi_dict=self.get_presentation_account_list_obj.export_label_to_taxonomi_dict()
        
        account_label_common = account_list_common_obj.get_assign_common_label()
        return account_label_common