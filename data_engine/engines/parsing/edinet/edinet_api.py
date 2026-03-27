"""

"""

# %% Requirements

from __future__ import annotations
import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Annotated, Any, Literal, Optional

import pandas as pd
import pandera as pa
import requests
from loguru import logger
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field, validate_call
from pydantic.functional_validators import BeforeValidator
from tqdm import tqdm



# Allowing None str type
StrOrNone = Annotated[str, BeforeValidator(lambda x: x or "")]

def get_columns(schema):
    return list(schema.model_json_schema()['properties'].keys())

class EdinetResponse(BaseModel):
    """書類一覧APIのレスポンススキーマ

    access_date: アクセス日
    seqNumber: 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
    docID: filename
    edinetCode: EDINETコード
    secCode: 証券コード
    JCN: 法人番号
    filerName: 提出者名
    fundCode: ファンドコード
    ordinanceCode: 政令コード
    formCode: 様式コード
    docTypeCode: 書類種別コード
    periodStart: 開始期間
    periodEnd: 終了期間
    submitDateTime: 書類提出日時 
    docDescription: EDINET の閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列
    issuerEdinetCode: 発行会社EDINETコード大量保有について発行会社のEDINETコード
    subjectEdinetCode: 公開買付けについて対象となるEDINETコード
    subsidiaryEdinetCode: 子会社のEDINETコードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    currentReportReason: 臨報提出事由、臨時報告書の提出事由が出力されます。複数存在する場合、","(カンマ)で結合した文字列が出力
    parentDocID: 親書類管理番号
    opeDateTime: 「2-1-6 財務局職員による書類情報修正」、「2-1-7 財務局職員による書類の不開示」、磁気ディスク提出及び紙面提出を行った日時が出力
    withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    disclosureStatus: 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は"3"、それ以外は"0"が出力
    xbrlFlag: 書類にXBRLがある場合は"1"それ以外0
    pdfFlag: 書類にPDFがある場合は"1"それ以外0
    attachDocFlag: 書類に代替書面・添付文書がある場合:1 それ以外:0
    englishDocFlag: 書類に英文ファイルがある場合1
    csvFlag: 書類にcsvがある場合1
    legalStatus: "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)
    
    参考: 11_EDINET_API仕様書（version 2）.pdfより
    """
    access_date: date = Field(..., title="access date", description="access date")
    seqNumber: int =Field(..., title="seq number", description="The number YYYY/MM/DD-senCumber, which is given to documents submitted on the same day in the order of submission time, becomes the submission order information.")
    docID: StrOrNone = Field("", title="document id", description="filename of document (docID.zip)")
    edinetCode: StrOrNone = Field("", title="EDINET code", description="EDINET code")
    secCode: StrOrNone = Field("", title="Securities Code", description="Securities Code")
    JCN: StrOrNone = Field("", title="corporate identity number", description="corporate identity number")
    filerName: StrOrNone = Field("", title="Name of submitter", description="Name of submitter")
    fundCode: StrOrNone = Field("", title="Fund code", description="Fund code")
    ordinanceCode: StrOrNone = Field("", title="government ordinance code", description="government ordinance code")
    formCode: StrOrNone = Field("", title="form code", description="form code")
    docTypeCode: StrOrNone = Field("", title="document type code", description="document type code")
    periodStart: StrOrNone = Field("", title="start period", description="start period(YYYY-MM-DD)")
    periodEnd: StrOrNone = Field("", title="end period", description="end period(YYYY-MM-DD)")
    submitDateTime: StrOrNone = Field("", title="submit date time", description="submit date time(YYYY-MM-DD HH:MM:SS)")
    docDescription: StrOrNone = Field("", title="document description", description="String displayed in the 'Documents submitted' field on the document search results screen of the EDINET browse site.")
    issuerEdinetCode: StrOrNone = Field("", title="issuer EDINET code", description="Issuer company EDINET code Large holding EDINET code of the issuing company")
    subjectEdinetCode: StrOrNone = Field("", title="subject EDINET code", description="Target EDINET code for the tender offer.")
    subsidiaryEdinetCode: StrOrNone = Field("", title="subsidiary EDINET code", description="The EDINET code of the subsidiary is output. If more than one exists (up to 10), the strings are output concatenated by ',' (comma).")
    currentReportReason: StrOrNone = Field("", title="current report reason", description="The reasons for submitting an extraordinary report and reasons for submitting an extraordinary report are output. If there is more than one, the strings are output as ',' (comma) concatenated strings.")
    parentDocID: StrOrNone = Field("", title="parent document management number", description="Parent document management number")
    opeDateTime: StrOrNone = Field("", title="operation date time", description="The date and time when the document was modified by the staff of the Financial Bureau, the document was not disclosed by the staff of the Financial Bureau, and the magnetic disk submission and paper submission were made.")
    withdrawalStatus: StrOrNone = Field("", title="withdrawal status", description="Withdrawal draft is '1', withdrawn documents are '2', and others are '0'.")
    docInfoEditStatus: StrOrNone = Field("", title="document information edit status", description="Information that the staff of the Financial Bureau modified the document is '1', the modified document is '2', and others are '0'.")
    disclosureStatus: StrOrNone = Field("", title="disclosure status", description="Information that the staff of the Financial Bureau started non-disclosure of the document is '1', the document that is not disclosed is '2', the information that the staff of the Financial Bureau released the non-disclosure of the document is '3', and others are '0'.")
    xbrlFlag: StrOrNone = Field("", title="XBRL flag", description="If the document has XBRL, it is '1', otherwise 0.")
    pdfFlag: StrOrNone = Field("", title="PDF flag", description="If the document has PDF, it is '1', otherwise 0.")
    attachDocFlag: StrOrNone = Field("", title="attach document flag", description="If the document has an alternative document/attachment, it is '1', otherwise 0.")
    englishDocFlag: StrOrNone = Field("", title="english document flag", description="If the document has an English file, it is '1'.")
    csvFlag: StrOrNone = Field("", title="csv flag", description="If the document has a csv, it is '1'.")
    legalStatus: StrOrNone = Field("", title="legal status", description="'1': being viewed '2': during the extended period (a document whose statutory viewing period has expired but can still be viewed.) '0': viewing period expired (a document whose viewing period has expired and cannot be viewed due to the expiration of the extended period or withdrawal. Note that this does not include non-disclosure.)")


class EdinetResponseList(BaseModel):
    """書類一覧APIのレスポンススキーマのリスト
    以下からなるedinet_response_schemaのリスト
        access_date: アクセス日
        seqNumber: 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
        docID: filename
        edinetCode: EDINETコード
        secCode: 証券コード
        JCN: 法人番号
        filerName: 提出者名
        fundCode: ファンドコード
        ordinanceCode: 政令コード
        formCode: 様式コード
        docTypeCode: 書類種別コード
        periodStart: 開始期間
        periodEnd: 終了期間
        submitDateTime: 書類提出日時 
        docDescription: EDINET の閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列
        issuerEdinetCode: 発行会社EDINETコード大量保有について発行会社のEDINETコード
        subjectEdinetCode: 公開買付けについて対象となるEDINETコード
        subsidiaryEdinetCode: 子会社のEDINETコードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
        currentReportReason: 臨報提出事由、臨時報告書の提出事由が出力されます。複数存在する場合、","(カンマ)で結合した文字列が出力
        parentDocID: 親書類管理番号
        opeDateTime: 「2-1-6 財務局職員による書類情報修正」、「2-1-7 財務局職員による書類の不開示」、磁気ディスク提出及び紙面提出を行った日時が出力
        withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
        docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
        disclosureStatus: 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は"3"、それ以外は"0"が出力
        xbrlFlag: 書類にXBRLがある場合は"1"それ以外0
        pdfFlag: 書類にPDFがある場合は"1"それ以外0
        attachDocFlag: 書類に代替書面・添付文書がある場合:1 それ以外:0
        englishDocFlag: 書類に英文ファイルがある場合1
        csvFlag: 書類にcsvがある場合1
        legalStatus: "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)
    
    参考: 11_EDINET_API仕様書（version 2）.pdfより
    """
    data: list[EdinetResponse]


class EdinetMetadataInputV2(BaseModel):
    date_api_param: date = Field(..., title="date", description="date(YYYY-MM-DD)")
    type_api_param: Literal[1, 2] = Field(2, title="type", description="1: Retrieves metadata only. 2: Retrieve the list of submitted documents and metadata.")
    api_key: str = Field(..., title="Subscription-Key", description="API Key")
    ope_date_time_api_param: Optional[str] = Field(None, title="opeDateTime", description="HH:MM:SS")

    def export(self):
        params = {"date": self.date_api_param, "type": self.type_api_param, "Subscription-Key": self.api_key}
        if self.ope_date_time_api_param:
            params["opeDateTime"] = self.ope_date_time_api_param
        return params

class RequestResponse(BaseModel):
    date_res: date = Field(..., title="date", description="date(YYYY-MM-DD)")
    status: Literal['success','failure'] = Field('succsess', title="result", description="success or failure")
    data: list[EdinetResponse] = Field(default=None, title="data", description="data")
    message: StrOrNone = Field(default="", title="message", description="message")


def normalize_date(v: Any) -> datetime:
    if v is None:
        return datetime.now()
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime.combine(v, datetime.min.time())
    if isinstance(v, str):
        s_v = v.strip()
        if not s_v or s_v.lower() in ["nan", "none"]:
            return datetime.now()
        # カンマやスラッシュ、ハイフンなどの区切りに対応
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                # 区切り文字を統一せず、多様な形式(fmt)で直接パースを試みる
                return datetime.strptime(s_v[:10], fmt)
            except Exception:
                continue
    return datetime.now()

class DateNormalizer(BaseModel):
    date_norm: Annotated[Optional[datetime], BeforeValidator(normalize_date)] = Field(default_factory=datetime.now)
    def export_date(self):
        return self.date_norm


@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def get_edinet_metadata(params: EdinetMetadataInputV2, session: Optional[requests.Session] = None) -> RequestResponse:
    """
    EDINET APIの書類一覧APIを利用して書類一覧を取得する
    """
    EDINET_API_url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
    
    if session is None:
        from data_engine.core.network_utils import GLOBAL_ROBUST_SESSION
        session = GLOBAL_ROBUST_SESSION

    # verify は Session 側の設定に従い、個別の get では True (デフォルト) を尊重する
    res = session.get(EDINET_API_url, params=params.export(), timeout=(20, 30))
    result_temp = {"date_res": params.date_api_param,"status": "success","data": [],"message":None}
    
    if res.status_code == 200:
        result_temp["status"] = "success"
        try:
            res_list=[]
            res_parsed = json.loads(res.text)
            if 'results' in res_parsed and res_parsed['results'] is not None:
                for res_day in res_parsed['results']:
                    res_day['access_date'] = datetime.today().strftime('%Y-%m-%d')
                    res_list.append({key: res_day[key] for key in get_columns(EdinetResponse)})
            else:
                logger.warning(f"EDINET metadata response does not contain 'results' for date: {params.date_api_param}")
            result_temp["data"] = res_list

        except json.JSONDecodeError as e:
            result_temp["status"] = "failure"
            result_temp["message"] = f"JSON Decoding Error: {str(e)}"
            pass
        except Exception as e:
            logger.error(f"EDINET metadata parsing error: {e}")
            result_temp["status"] = "failure"
            result_temp["message"] = f"Error: {str(e)}"
            pass
    else:
        result_temp["status"] = "failure"
        result_temp["message"] = f"Failure: {res.status_code}"
    
    result = RequestResponse(**result_temp)
    return result


class EdinetResponseDf(pa.DataFrameModel):
    """
    access_date: データ取得日
    seqNumber: 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
    docID: filename
    edinetCode: EDINETコード
    secCode: 証券コード
    JCN: 法人番号
    filerName: 提出者名
    fundCode: ファンドコード
    ordinanceCode: 政令コード
    formCode: 様式コード
    docTypeCode: 書類種別コード
    periodStart: 開始期間
    periodEnd: 終了期間
    submitDateTime: 書類提出日時 
    docDescription: EDINET の閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列
    issuerEdinetCode: 発行会社EDINETコード大量保有について発行会社のEDINETコード
    subjectEdinetCode: 公開買付けについて対象となるEDINETコード
    subsidiaryEdinetCode: 子会社のEDINETコードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    currentReportReason: 臨報提出事由、臨時報告書の提出事由が出力されます。複数存在する場合、","(カンマ)で結合した文字列が出力
    parentDocID: 親書類管理番号
    opeDateTime: 「2-1-6 財務局職員による書類情報修正」、「2-1-7 財務局職員による書類の不開示」、磁気ディスク提出及び紙面提出を行った日時が出力
    withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    disclosureStatus: 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は"3"、それ以外は"0"が出力
    xbrlFlag: 書類にXBRLがある場合は"1"それ以外0
    pdfFlag: 書類にPDFがある場合は"1"それ以外0
    attachDocFlag: 書類に代替書面・添付文書がある場合:1 それ以外:0
    englishDocFlag: 書類に英文ファイルがある場合1
    csvFlag: 書類にcsvがある場合1
    legalStatus: "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)
    
    参考: 11_EDINET_API仕様書（version 2）.pdfより
    """
    access_date: Series[date] = pa.Field(nullable=False) # access date
    seqNumber: Series[int] # 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
    docID: Series[str] # filename
    edinetCode: Series[str] = pa.Field(nullable=True) # EDINETコード
    secCode: Series[str] = pa.Field(nullable=True) # 証券コード
    JCN: Series[str] = pa.Field(nullable=True) # 法人番号
    filerName: Series[str] = pa.Field(nullable=True) # 提出者名
    fundCode: Series[str] = pa.Field(nullable=True) # ファンドコード
    ordinanceCode: Series[str] = pa.Field(nullable=True) # 政令コード
    formCode: Series[str] = pa.Field(nullable=True) # 様式コード
    docTypeCode: Series[str] = pa.Field(nullable=True) # 書類種別コード
    periodStart: Series[str] = pa.Field(nullable=True) # 開始期間
    periodEnd: Series[str] = pa.Field(nullable=True) # 終了期間
    submitDateTime: Series[str] = pa.Field(nullable=True) # 書類提出日時
    docDescription: StrOrNone = Field(
        default="",
        title="提出書類",
        description="EDINET の閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列"
    )
    issuerEdinetCode: Series[str] = pa.Field(nullable=True) # 発行会社EDINETコード 大量保有について発行会社の EDINETコード
    subjectEdinetCode: Series[str] = pa.Field(
        nullable=True
    ) # 公開買付けについて対象となるEDINETコード
    subsidiaryEdinetCode: Series[str] = pa.Field(
        nullable=True
    ) # 子会社の EDINET コードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    currentReportReason: Series[str] = pa.Field(
        nullable=True
    ) # 臨報提出事由、臨時報告書の提出事由が出力され ます。複数存在する場合、","(カンマ)で結合した文字列が出力
    parentDocID: Series[str] = pa.Field(nullable=True) # 親書類管理番号
    opeDateTime: StrOrNone = Field(
        default="",
        title="処理日時",
        description="「2-1-6 財務局職員による書類情報修正」などの日時"
    )
    withdrawalStatus: Series[str] = pa.Field(isin=['0','1','2']) # 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: Series[str] = pa.Field(isin=['0','1','2']) # 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    disclosureStatus: Series[str] = pa.Field(
        isin=['0','1','2','3']
    ) # 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は "3"、それ以外は"0"が出力
    xbrlFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にXBRLがある場合は"1"、それ以外0
    pdfFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にPDFがある場合は"1"、それ以外0
    attachDocFlag: Series[str] = pa.Field(
        isin=['0','1']
    ) # 書類に代替書面・添付文書がある場合: 1、それ以外: 0
    englishDocFlag: Series[str] = pa.Field(isin=['0','1']) # 書類に英文ファイルがある場合: 1
    csvFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にcsvがある場合1
    legalStatus: Series[str] = pa.Field(
        isin=['0','1','2']
    ) # "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)
    sector_label_33: Series[str] = pa.Field(nullable=True) # 33業種区分

    def __len__(self):
        return len(self.index)

class edinet_response_metadata():
    """
    書類一覧APIのレスポンススキーマのリストを保持するクラス
    """
    def __init__(self,filename=None,tse_sector_url:str=None,tmp_path_str:str=None):
        if filename:
            self.read_jsonl(filename)
        if tse_sector_url:
            self.tse_sector_url = tse_sector_url
            self.tmp_path = Path(tmp_path_str)

    def save(self,filename:str):
        """
        jsonl形式で保存
        """
        with open(filename, 'w') as file:
            for obj in self.data:
                file.write(obj.model_dump_json() + '\n')
    def read_jsonl(self,filename:str):
        """jsonl形式のファイルを読み込む"""
        response_metadata = []
        with open(filename, 'r') as file:
            data = file.readlines()
            for line in data:
                line_json = json.loads(line)
                if line_json["status"]=="success":
                    response_metadata.append(RequestResponse(**line_json))
                else:
                    logger.info(line_json["message"])
        self.data: list[EdinetResponse]= response_metadata
    
    #def load(self,filename):
    #    res_results=self.read_jsonl(filename)
    #    data_list=[]
    #    for res in res_results:
    #        if res.status == "success":
    #            data_list=data_list+res.data
    #    return data_list
    
    def set_data(self,data):
        self.data: list[EdinetResponse]=data
    
    def get_metadata_pandas_df(self)->pd.DataFrame:
        return pd.concat([pd.DataFrame(data.model_dump()['data']) for data in self.data]).reset_index()
    
    def get_yuho_df(self)->EdinetResponseDf:
        df = self.get_metadata_pandas_df()
        df_f = df.query("docTypeCode=='120' and ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2'")
        if self.tse_sector_url:
            self.tmp_path.mkdir(parents=True, exist_ok=True)
            r = requests.get(self.tse_sector_url, stream=True)
            sector_file_path = self.tmp_path / "sector_file.xls"
            with sector_file_path.open(mode="wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            business_class = pd.read_excel(
                sector_file_path,header=0,index_col=None,dtype={'コード':str}
                ).rename(columns={'日付':'date','コード':'secCode','33業種コード':'sector_code_33','33業種区分':'sector_label_33','17業種コード':'sector_code_17','17業種区分':'sector_label_17'})[['date','secCode','sector_code_33','sector_code_17','sector_label_33','sector_label_17']]
            from data_engine.core.utils import normalize_code
            business_class.secCode = business_class.secCode.apply(lambda x: normalize_code(x, nationality="JP"))
            df_f = pd.merge(
                df_f,
                business_class[['secCode','sector_label_33']],
                left_on='secCode',
                right_on='secCode',
                how='left')
        return EdinetResponseDf(df_f)

    #def get_teisei_yuho_df(self)->EdinetResponseDf:
    #    df = self.get_metadata_pandas_df()
    #    return df.query("docTypeCode=='130' and ordinanceCode == '010' and formCode == '030001' and docInfoEditStatus !='2'")

def request_term(api_key:str, start_date_str:str,end_date_str:str, ope_date_time_str:str=None, session: Optional[requests.Session] = None)->list[RequestResponse]:
    """
    書類一覧APIを利用して開始日と終了日を含む期間の書類一覧を取得します。
        start_date_str: 開始日(YYYY-MM-DD)
        end_date_str: 終了日(YYYY-MM-DD)
        ope_date_time_str: 前回取得した最後の操作日時(HH:MM:SS) 
                           原則として開始日(start_date)に対して適用される。
        session: 外部注入された requests.Session オブジェクト
    """

    start_date = DateNormalizer(date_norm=start_date_str).export_date()
    end_date = DateNormalizer(date_norm=end_date_str).export_date()

    res_results = []
    days_to_fetch = (end_date - start_date).days + 1
    
    for itr in tqdm(range(0, days_to_fetch)):
        target_date = start_date + timedelta(days=itr)
        input_dict = {
            "date_api_param" : target_date.strftime("%Y-%m-%d"),
            "type_api_param" : 2,
            "api_key":api_key,
            # 【main準拠】ope_date_time は初日のみ適用。
            # 初日: 前回取得以降の更新分のみ取得（増分同期）
            # 2日目以降: 全件取得（未見の日付のためフィルタ不要）
            "ope_date_time_api_param": ope_date_time_str if itr == 0 else None
        }
        params = EdinetMetadataInputV2(**input_dict)
        res_results.append(get_edinet_metadata(params, session=session))
        # 1秒間に1回のリクエスト制限を遵守 (0.5s -> 1.1s に微調整して余裕を持たせる)
        sleep(1.1)
    return res_results
# %% doc

class EdinetDocInputV2(BaseModel):
    type_api_param: Literal[1, 2, 3, 4, 5] = Field(1, title="type", description="1: xbrl, 2: pdf, 3:attach, 4:english, 5:csv")
    api_key: str = Field(..., title="Subscription-Key", description="API Key")
    def export(self):
        return {"type": self.type_api_param, "Subscription-Key": self.api_key}

class RequestResponseDoc(BaseModel):
    docid: str = Field(..., title="docid", description="docid")
    data_path: StrOrNone = Field(default="", title="data path", description="data path")
    status: Literal['success','failure'] = Field('succsess', title="result", description="Success or Failure")
    message: StrOrNone = Field(default="", title="message", description="message")

@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def request_doc(api_key: str, docid: str, out_filename_str: str, doc_type: int = 1, session: Optional[requests.Session] = None) -> RequestResponseDoc:
    # EDINET API version 2
    out_filename_path = Path(out_filename_str)
    EDINET_API_url = "https://api.edinet-fsa.go.jp/api/v2/documents/" + docid
    
    if session is None:
        from data_engine.core.network_utils import GLOBAL_ROBUST_SESSION
        session = GLOBAL_ROBUST_SESSION

    input_dict = {
        "type_api_param": doc_type, # 1:xbrl # 2: PDF 5:csv,
        "api_key": api_key
    }
    params = EdinetDocInputV2(**input_dict)
    result_temp = {"docid": docid, "status": "success", "data_path": None, "message": None}
    try:
        res = session.get(EDINET_API_url, params=params.export(), timeout=(20, 90))
        if res.status_code == 200:
            result_temp["status"] = "success"
            out_filename_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_filename_path, 'wb') as f:
                for chunk in res.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
            result_temp["data_path"] = str(out_filename_path)
        else:
            result_temp["status"] = "failure"
            result_temp["message"] = f"failure: {res.status_code}"
    except Exception as e:
        result_temp["status"] = "failure"
        result_temp["message"] = f"Error: {str(e)}"
        result_temp["data_path"] = None
        pass
    result = RequestResponseDoc(**result_temp)
    return result


