import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from data_engine.core.network_utils import is_404
from data_engine.core.utils import force_gc


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

        # 出力先ファイル (全パスで共通)
        local_file = self.data_path / f"master_bin{bin_id}_{master_type}.parquet"
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. 既存データのロード & マージ & 重複排除 → ファイル直接出力
        #    【メモリ最適化】DuckDB の COPY TO でファイルに直接書き出し、
        #    Python ヒープに結果を一切載せない。巨大 Bin (3GB+) でも安全。
        try:
            m_path = hf_hub_download(repo_id=self.hf_repo, filename=repo_path, repo_type="dataset", token=self.hf_token)
            self._merge_with_duckdb(m_path, new_data, master_type, bin_id, local_file)
        except Exception as e:
            if is_404(e):
                logger.info(f"新規Master作成: bin={bin_id} ({master_type})")
                from data_engine.core.models import ARIA_SCHEMAS
                schema = ARIA_SCHEMAS.get(master_type)
                new_data.to_parquet(local_file, compression="zstd", index=False, schema=schema)
            else:
                logger.error(f"Master取得中に通信エラーが発生しました (404以外): {e}")
                raise e

        # 2. アップロード (local_file は確実に存在)
        if self.api:
            if defer and catalog_manager:
                catalog_manager.add_commit_operation(repo_path, local_file)
                logger.debug(f"Master更新をバッファに追加: bin={bin_id} ({master_type})")
                force_gc()
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
                    force_gc()
                    return True
                except Exception as e:
                    if isinstance(e, HfHubHTTPError) and e.response.status_code == 429:
                        wait_time = int(e.response.headers.get("Retry-After", 60)) + 5
                        logger.warning(f"Master Rate limit exceeded. Waiting {wait_time}s... ({attempt + 1}/5)")
                        time.sleep(wait_time)
                        continue

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

    def _merge_with_duckdb(
        self,
        master_path: str,
        new_data: pd.DataFrame,
        master_type: str,
        bin_id: str,
        output_path: Path,
    ) -> None:
        """DuckDB COPY TO によるゼロコピー・マージ

        結果を Python メモリに一切載せず、DuckDB が直接 Parquet ファイルに書き出す。
        DuckDB は独自バッファ管理 (memory_limit 制御) を持ち、
        必要に応じてディスクにスピルするため、巨大 Bin でも安全に処理可能。

        処理フロー:
        1. new_data を一時 Parquet に書き出す
        2. DuckDB で既存マスタと結合・重複排除
        3. COPY TO で結果を直接 output_path に書き出す (fetchdf 不使用)
        4. 一時ファイルを確実に削除
        """
        import duckdb

        temp_new = self.data_path / f"_temp_new_{bin_id}_{master_type}.parquet"

        try:
            new_data.to_parquet(temp_new, compression="zstd", index=False)

            if master_type == "financial_values":
                partition_cols = "docid, key, context_ref"
            else:
                partition_cols = "docid, key"

            has_submit_dt = "submitDateTime" in new_data.columns
            order_clause = "ORDER BY submitDateTime DESC NULLS LAST" if has_submit_dt else "ORDER BY 1"

            master_esc = str(master_path).replace("'", "''")
            temp_esc = str(temp_new).replace("'", "''")
            out_esc = str(output_path).replace("'", "''")

            con = duckdb.connect()
            try:
                con.execute("SET memory_limit='3GB'")
                con.execute(f"""
                    COPY (
                        SELECT * EXCLUDE (rn) FROM (
                            SELECT *, ROW_NUMBER() OVER (
                                PARTITION BY {partition_cols}
                                {order_clause}
                            ) AS rn
                            FROM read_parquet(
                                ['{master_esc}', '{temp_esc}'],
                                union_by_name=true
                            )
                        ) WHERE rn = 1
                    ) TO '{out_esc}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
            finally:
                con.close()

            logger.debug(f"DuckDB マージ完了 (COPY TO): bin={bin_id}")

        finally:
            if temp_new.exists():
                temp_new.unlink()
            force_gc()

