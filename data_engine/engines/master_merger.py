import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger


class MasterMerger:
    def __init__(self, hf_repo: str, hf_token: str, data_path: Path):
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.api = HfApi() if hf_repo and hf_token else None

    def get_bin_id(self, row: dict) -> str:
        """物理的事実に基き、不変の分散キー (EDINET Code 最優先) を導出する"""
        e_raw = row.get("edinet_code") or ""
        e_code = str(e_raw).strip()
        
        if e_code and len(e_code) >= 2 and e_code.lower() not in ["none", "nan", "null"]:
            return f"E{e_code[-2:]}"

        c_raw = row.get("code") or ""
        c_code = str(c_raw).strip().split(":")[-1]
        
        if c_code and len(c_code) >= 2 and c_code.lower() not in ["none", "nan", "null"]:
            # JPX等コード (e.g. 72030) - 5桁目の0を避け実質的な末尾2桁を取得
            return f"P{c_code[-3:-1]}"

        jcn_raw = row.get("jcn") or ""
        jcn_val = str(jcn_raw).strip()
        
        if jcn_val and len(jcn_val) >= 2 and jcn_val.lower() not in ["none", "nan", "null"]:
            return f"J{jcn_val[-2:]}"

        return "No"

    def merge_and_upload(
        self,
        bin_id: str,
        master_type: str,
        new_data: pd.DataFrame,
        worker_mode: bool = False,
        catalog_manager=None,
        run_id: str = None,
        chunk_id: str = None,
        defer: bool = False,
    ) -> bool:
        """業種別にParquetをロード・結合・アップロード"""
        if new_data.empty:
            return True

        # bin_id が指定されていない場合は先頭レコードから導出
        if not bin_id:
            bin_id = self.get_bin_id(new_data.iloc[0].to_dict())

        if worker_mode:
            filename = f"{master_type}_bin{bin_id}.parquet"

            return catalog_manager.save_delta(
                key=master_type,
                df=new_data,
                run_id=run_id,
                chunk_id=chunk_id,
                custom_filename=filename,
                defer=defer,
                local_only=True,
            )

        repo_path = f"master/{master_type}/bin={bin_id}/data.parquet"

        # 1. 既存データのロード
        try:
            m_path = hf_hub_download(repo_id=self.hf_repo, filename=repo_path, repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(m_path)
            logger.debug(f"既存Master読み込み: bin={bin_id} ({len(master_df)} rows)")
            combined_df = pd.concat([master_df, new_data], ignore_index=True)
        except Exception:
            logger.info(f"新規Master作成: bin={bin_id} ({master_type})")
            combined_df = new_data

        # 2. 重複排除 (最新優先)
        subset = ["docid", "key", "context_ref"] if master_type == "financial_values" else ["docid", "key"]

        if "submitDateTime" in combined_df.columns:
            combined_df = combined_df.sort_values("submitDateTime", ascending=False)

        combined_df = combined_df.drop_duplicates(subset=subset, keep="first")

        # 3. 保存とアップロード
        local_file = self.data_path / f"master_bin{bin_id}_{master_type}.parquet"

        # 【極限ガード】NULL 基底アーキテクチャの死守
        # 全量文字列化を廃止し、型の誠実性を保つ
        combined_df = combined_df.convert_dtypes()

        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 【Phase 7: Perfect Integrity】レジストリから金型を抽出し適用
        from data_engine.core.models import ARIA_SCHEMAS
        schema = ARIA_SCHEMAS.get(master_type)
        
        # ビンファイルはXBRLの動的拡張を含む可能性があるため、
        # スキーマが既知の場合はそれを優先し、未知の場合は推論に委ねる（ただし真のNullは維持）
        combined_df.to_parquet(local_file, compression="zstd", index=False, schema=schema)

        if self.api:
            # 【重要】defer=True の場合は、モードに関わらずコミットバッファに積む
            if defer and catalog_manager:
                catalog_manager.add_commit_operation(repo_path, local_file)
                logger.debug(f"Master更新をバッファに追加: bin={bin_id} ({master_type})")
                return True

            max_retries = 5  # 3回から5回に強化
            for attempt in range(max_retries):
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=repo_path,
                        repo_id=self.hf_repo,
                        repo_type="dataset",
                        token=self.hf_token,
                    )
                    logger.success(f"Master更新成功: bin={bin_id} ({master_type})")
                    return True
                except Exception as e:
                    if isinstance(e, HfHubHTTPError) and e.response.status_code == 429:
                        wait_time = int(e.response.headers.get("Retry-After", 60)) + 5
                        logger.warning(f"Master Rate limit exceeded. Waiting {wait_time}s... ({attempt + 1}/5)")
                        time.sleep(wait_time)
                        continue

                    # 5xx エラー等もリトライ対象に追加
                    if isinstance(e, HfHubHTTPError) and e.response.status_code >= 500:
                        wait_time = 15 * (attempt + 1)
                        logger.warning(
                            f"Master HF Server Error ({e.response.status_code}). "
                            f"Waiting {wait_time}s... ({attempt + 1}/5)"
                        )
                        time.sleep(wait_time)
                        continue

                    logger.error(f"Masterアップロード失敗: bin={bin_id} - {e}")
                    return False
            return False
        return True
