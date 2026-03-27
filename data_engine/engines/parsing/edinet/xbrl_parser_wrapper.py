from __future__ import annotations
import json
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import pandas as pd
import pandera as pa
from arelle import Cntlr
from arelle.ModelValue import qname
from pandera.typing import Series
from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator

from .utils import get_columns_df
# %% #################################################################
#
#            schema
#
######################################################################

# Arelle ラベル解決用ロール定数
LABEL_ROLE = "http://www.xbrl.org/2003/role/label"
VERBOSE_LABEL_ROLE = "http://www.xbrl.org/2003/role/verboseLabel"

class xbrl_elm_schema(pa.DataFrameModel):
    """
        key:prefix+":"+element_name
        data_str
        context_ref
    """
    key: Series[str] = pa.Field(nullable=True)
    data_str: Series[str] = pa.Field(nullable=True)
    context_ref: Series[str] = pa.Field(nullable=True)
    decimals: Series[str] = pa.Field(nullable=True)# T:-3, M:-6, B:-9
    precision: Series[str] = pa.Field(nullable=True)
    element_name: Series[str] = pa.Field(nullable=True)
    unit: Series[str] = pa.Field(nullable=True)# 'JPY'
    period_type: Series[str] = pa.Field(isin=['instant','duration'],nullable=True) # 'instant','duration'
    isTextBlock_flg: Series[int] = pa.Field(isin=[0,1],nullable=True) # 0,1
    abstract_flg: Series[int] = pa.Field(isin=[0,1],nullable=True) # 0,1
    period_start: Series[str] = pa.Field(nullable=True)
    period_end: Series[str] = pa.Field(nullable=True)
    instant_date: Series[str] = pa.Field(nullable=True)
    # Arelle直接ラベル（タクソノミ参照チェーンから自動解決）
    arelle_label_jp: Series[str] = pa.Field(nullable=True)
    arelle_label_en: Series[str] = pa.Field(nullable=True)
    arelle_label_jp_long: Series[str] = pa.Field(nullable=True)
    arelle_label_en_long: Series[str] = pa.Field(nullable=True)

StrOrNone = Annotated[str, BeforeValidator(lambda x: x or "")]

class ArreleFact(BaseModel):

    key:StrOrNone
    data_str:StrOrNone
    decimals:StrOrNone
    precision:StrOrNone
    context_ref:StrOrNone
    element_name:StrOrNone
    unit:StrOrNone
    period_type:StrOrNone
    isTextBlock_flg:StrOrNone
    abstract_flg:StrOrNone
    period_start:StrOrNone
    period_end:StrOrNone
    instant_date:StrOrNone
    end_date_pv:StrOrNone
    instant_date_pv:StrOrNone
    scenario:StrOrNone
    # Arelle直接ラベル
    arelle_label_jp:StrOrNone
    arelle_label_en:StrOrNone
    arelle_label_jp_long:StrOrNone
    arelle_label_en_long:StrOrNone

def _safe_label(concept, lang: str, role: str) -> str | None:
    """Arelleのラベル解決。QNameフォールバック（偽ラベル）を検出して除外する。"""
    if concept is None:
        return None
    try:
        result = concept.label(lang=lang, preferredLabel=role)
        # Arelleはラベルが見つからない場合、QName文字列をフォールバックとして返す
        # これは事実ではないためNULLとして扱う
        if result == str(concept.qname):
            return None
        return result
    except Exception:
        return None

def get_fact_data(fact)->ArreleFact:
    concept = fact.concept
    qname_str = str(fact.qname)
    fact_data = {
        'key':qname_str,
        'data_str':fact.value,
        'decimals':fact.decimals,
        'precision':fact.precision,
        'context_ref':fact.contextID,
        'element_name':str(fact.qname.localName),
        'unit':fact.unitID,#(str) – unitRef attribute
        'period_type':concept.periodType,#'instant','duration'
        'isTextBlock_flg':int(concept.isTextBlock), # 0,1
        'abstract_flg':int(concept.abstract=='true'), # Note: concept.abstract is str not bool.
        # 【Arelle直接ラベル】タクソノミ参照チェーンから自動解決
        'arelle_label_jp': _safe_label(concept, 'ja', LABEL_ROLE),
        'arelle_label_en': _safe_label(concept, 'en', LABEL_ROLE),
        'arelle_label_jp_long': _safe_label(concept, 'ja', VERBOSE_LABEL_ROLE),
        'arelle_label_en_long': _safe_label(concept, 'en', VERBOSE_LABEL_ROLE),
    }
    if fact.context.startDatetime:
        fact_data['period_start'] = fact.context.startDatetime.strftime('%Y-%m-%d')
    else:
        fact_data['period_start'] = None
    if fact.context.endDatetime:
        fact_data['period_end'] = fact.context.endDatetime.strftime('%Y-%m-%d') # 1 day added???
    else:
        fact_data['period_end'] = None
    if fact.context.instantDatetime:
        fact_data['instant_date'] = fact.context.instantDatetime.strftime('%Y-%m-%d') # 1 day added???
    else:
        fact_data['instant_date'] = None

    
    fact_data['end_date_pv']=None
    fact_data['instant_date_pv']=None
    for item in fact.context.propertyView:
                if item:
                    if item[0] == 'endDate':
                        fact_data['end_date_pv'] = item[1]
                    elif item[0] == 'instant':
                        fact_data['instant_date_pv'] = item[1]
    scenario = []
    for (dimension, dim_value) in fact.context.scenDimValues.items():
        scenario.append({
            'ja': (
                dimension.label(preferredLabel=None, lang='ja', linkroleHint=None),
                dim_value.member.label(preferredLabel=None, lang='ja', linkroleHint=None)),
            'en': (
                dimension.label(preferredLabel=None, lang='en', linkroleHint=None),
                dim_value.member.label(preferredLabel=None, lang='en', linkroleHint=None)),
            'id': (
                dimension.id,
                dim_value.member.id),
        })
    if scenario:
            scenario_json = json.dumps(
                scenario, ensure_ascii=False, separators=(',', ':'))
    else:
        scenario_json = None

    fact_data['scenario'] = scenario_json
    return fact_data


def get_xbrl_dei_df(xbrl_filename:str,log_dict,temp_dir)->(xbrl_elm_schema,dict):
    if log_dict['arelle_log_fname'] is None:
        log_dict['arelle_log_fname'] = str(temp_dir / "arelle.log")

    ctrl = Cntlr.Cntlr(logFileName=str(log_dict['arelle_log_fname']))
    model_xbrl = ctrl.modelManager.load(xbrl_filename)
    localname="AccountingStandardsDEI"
    qname_prefix = "jpdei_cor"
    if qname_prefix in model_xbrl.prefixedNamespaces:
        ns = model_xbrl.prefixedNamespaces[qname_prefix]
        facts = model_xbrl.factsByQname[qname(ns, name=f"{qname_prefix}:{localname}")]
        fact_list = list(facts)
        if len(fact_list) > 0:
            log_dict[localname] = fact_list[0].value
        else:
            log_dict[localname] = None
    else:
        log_dict[localname] = None
        
    ctrl.close()
    return log_dict


def get_xbrl_df(xbrl_filename:str,log_dict,temp_dir)->(xbrl_elm_schema,dict):
    """
    arelle.ModelInstanceObject - Arelle
        https://arelle.readthedocs.io/en/2.18.0/apidocs/arelle/arelle.ModelInstanceObject.html#arelle.ModelInstanceObject.ModelFact
    """
    if log_dict['arelle_log_fname'] is None:
        log_dict['arelle_log_fname'] = str(temp_dir / "arelle.log")

    ctrl = Cntlr.Cntlr(logFileName=str(log_dict['arelle_log_fname']))
    model_xbrl = ctrl.modelManager.load(xbrl_filename)
    if len(model_xbrl.facts)==0:
        log_dict['xbrl_load_status']="failure"
        ctrl.close()
        return pd.DataFrame(columns=get_columns_df(xbrl_elm_schema)),log_dict
    else:
        log_dict['xbrl_load_status']="success"
        log_dict['total_facts_present'] = log_dict.get('total_facts_present', 0) + len(model_xbrl.facts)
        fact_dict_list = []
        for fact in model_xbrl.facts:
            fact_dict_list.append(get_fact_data(fact))
        # log
        ctrl.close()
        return pd.DataFrame(fact_dict_list).drop_duplicates(),log_dict

def get_xbrl_wrapper(docid,zip_file:str,temp_dir:Path,out_path:Path,update_flg=False,log_dict=None):
    if log_dict is None:
        log_dict = {"is_xbrl_file":None, "is_xsd_file":None, "arelle_log_fname":None,"status":None,"error_message":None}
    
    
    log_dict["already_parse_xbrl"] = False
    try:
        log_dict["already_parse_xbrl"] = False
        #data_dir_raw=PROJDIR / "data" / "1_raw"
        #zip_file = list(data_dir_raw.glob("data_pool_*/"+docid+".zip"))[0]
        with ZipFile(str(zip_file)) as zf:
            # すべての .xbrl ファイルを抽出（IFRS書類は複数インスタンスを持つ）
            fn=[item for item in zf.namelist() if item.endswith(".xbrl") and "PublicDoc" in item]
            if len(fn)>0:
                for f in fn:
                    zf.extract(f, out_path)
                log_dict["is_xbrl_file"] = True
            else:
                log_dict["is_xbrl_file"] = False
            # すべての .xsd ファイルを抽出
            fn=[item for item in zf.namelist() if item.endswith(".xsd") and "PublicDoc" in item]
            if len(fn)>0:
                for f in fn:
                    zf.extract(f, out_path)
                log_dict["is_xsd_file"] = True
            else:
                log_dict["is_xsd_file"] = False
            # すべての def.xml ファイルを抽出
            fn=[item for item in zf.namelist() if item.endswith("def.xml") and "PublicDoc" in item]
            if len(fn)>0:
                for f in fn:
                    zf.extract(f, out_path)
                log_dict["is_def_file"] = True
            else:
                log_dict["is_def_file"] = False
        xbrl_path=out_path / "XBRL" / "PublicDoc"

        # xbrl and xsd and def files must exist
        has_xbrl = len(list(xbrl_path.glob("*.xbrl"))) > 0
        has_xsd = len(list(xbrl_path.glob("*.xsd"))) > 0
        has_def = len(list(xbrl_path.glob("*def.xml"))) > 0

        if has_xbrl and has_xsd and has_def:
            all_xbrl_files = list(xbrl_path.glob("*.xbrl"))
            all_parsed_dfs = []
            
            for xbrl_f in all_xbrl_files:
                xbrl_filename = str(xbrl_f)
                (xbrl_path / "arelle.log").touch()
                
                # Parse facts
                df_part, log_dict = get_xbrl_df(xbrl_filename, log_dict, temp_dir)
                
                # Parse DEI metadata (only once or update if exist)
                log_dict = get_xbrl_dei_df(xbrl_filename, log_dict, temp_dir)
                if 'AccountingStandardsDEI' in log_dict and log_dict['AccountingStandardsDEI']:
                    df_part['AccountingStandardsDEI'] = log_dict['AccountingStandardsDEI']
                else:
                    df_part['AccountingStandardsDEI'] = None
                    
                all_parsed_dfs.append(df_part)
            
            xbrl_parsed = pd.concat(all_parsed_dfs, ignore_index=True) if all_parsed_dfs else pd.DataFrame(columns=get_columns_df(xbrl_elm_schema))
            log_dict["get_xbrl_status"] = "success"
            log_dict["get_xbrl_error_message"] = None

            out_filename=str(xbrl_path / "log_dict.json")
            with open(out_filename, mode="wt", encoding="utf-8") as f:
                json.dump(log_dict, f, ensure_ascii=False, indent=2)
            
            return xbrl_parsed,log_dict
        else:
            log_dict["get_xbrl_status"] = "failure"
            log_dict["get_xbrl_error_message"] = "No xbrl or xsd file"
            out_filename=str(xbrl_path / "log_dict.json")
            with open(out_filename, mode="wt", encoding="utf-8") as f:
                json.dump(log_dict, f, ensure_ascii=False, indent=2)
            return pd.DataFrame(columns=get_columns_df(xbrl_elm_schema)),log_dict
    except Exception as e:
        log_dict["get_xbrl_status"] = "failure"
        log_dict["get_xbrl_error_message"] = e
        out_filename=str(xbrl_path / "log_dict.json")
        with open(out_filename, mode="wt", encoding="utf-8") as f:
            json.dump(log_dict, f, ensure_ascii=False, indent=2)
            
        return pd.DataFrame(columns=get_columns_df(xbrl_elm_schema)),log_dict