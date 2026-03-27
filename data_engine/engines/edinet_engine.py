from pathlib import Path
from typing import Dict, List

from loguru import logger

from data_engine.core.models import EdinetDocument

# 内部モジュール（外部ライブラリ）のインポート
from data_engine.engines.parsing.edinet.edinet_api import (
    edinet_response_metadata,
    request_doc,
    request_term,
)
from data_engine.engines.parsing.edinet.link_base_file_analyzer import account_list_common


class EdinetEngine:
    def __init__(self, api_key: str, data_path: Path, taxonomy_urls: Dict[str, str] = None):
        self.api_key = api_key
        self.data_path = data_path
        self.taxonomy_urls = taxonomy_urls or {}
        # 共通の堅牢なセッションを取得
        from data_engine.core.network_utils import GLOBAL_ROBUST_SESSION

        self.session = GLOBAL_ROBUST_SESSION
        logger.debug("EdinetEngine を初期化しました (Persistent Session 注入済)。")

    def fetch_metadata(self, start_date: str, end_date: str, ope_date_time: str = None) -> List[Dict]:
        """
        指定期間の全書類メタデータを取得し、Pydanticでバリデーション
        ope_date_timeを指定すると、API V2 の増分同期機能を使用して差分のみを取得。
        """
        # 入力値の前後空白を除去 (Pydanticバリデーションエラー対策)
        start_date = start_date.strip() if isinstance(start_date, str) else start_date
        end_date = end_date.strip() if isinstance(end_date, str) else end_date

        logger.info(f"EDINETメタデータ取得開始: {start_date} ~ {end_date} (増分基準: {ope_date_time or 'なし'})")

        # サブモジュールの新インターフェース (session注入) を使用
        res_results = request_term(
            api_key=self.api_key,
            start_date_str=start_date,
            end_date_str=end_date,
            ope_date_time_str=ope_date_time,
            session=self.session,
        )

        from data_engine.core.config import TSE_URL

        tse_url = TSE_URL
        meta = edinet_response_metadata(tse_sector_url=tse_url, tmp_path_str=str(self.data_path))
        meta.set_data(res_results)

        df = meta.get_metadata_pandas_df()

        if df.empty:
            logger.warning("No documents found for the specified period.")
            return []

        records = df.to_dict("records")
        validated_records = []
        for rec in records:
            try:
                # Pydantic モデルでバリデーション & 正規化
                doc = EdinetDocument(**rec)
                validated_records.append(doc.model_dump(by_alias=True))
            except Exception as e:
                logger.error(f"Validation failed for metadata (docID: {rec.get('docID')}): {e}")

        logger.info(f"Metadata fetch completed: {len(validated_records)} documents")
        return validated_records

    def download_doc(self, doc_id: str, save_path: Path, doc_type: int = 1) -> bool:
        """書類をダウンロード保存 (1=XBRL, 2=PDF)"""
        # サブモジュールの新インターフェース (session注入) を使用し、ロジックを委譲
        try:
            res = request_doc(
                api_key=self.api_key, docid=doc_id, out_filename_str=str(save_path), doc_type=doc_type, session=self.session
            )
            if res.status == "success":
                logger.debug(f"取得成功: {doc_id} (type={doc_type})")
                return True
            else:
                logger.error(f"DL失敗: {doc_id} ({res.message})")
                return False
        except Exception:
            logger.exception(f"DLエラー: {doc_id}")
            return False

    def get_account_list(self, taxonomy_year: str):
        """解析用タクソノミの取得"""
        try:
            logger.debug(f"タクソノミ取得試行: data_path={self.data_path}, year={taxonomy_year}")
            # サブモジュールの新インターフェース (session/taxonomy_urls注入) を使用
            acc = account_list_common(
                self.data_path, taxonomy_year, session=self.session, taxonomy_urls=self.taxonomy_urls
            )
            return acc
        except Exception:
            logger.exception(f"タクソノミ取得エラー (Year: {taxonomy_year})")
            return None
