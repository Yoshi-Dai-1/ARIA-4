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
        """2段階ストリーミング・マージ (ROW_NUMBER 不使用)

        ROW_NUMBER() Window 関数はブロッキング演算子でありディスクスピルが
        できないため、以下の2段階に分割して処理する:

        Step 1: DuckDB ORDER BY + COPY TO
            - 2つの Parquet を結合し、重複排除キー + submitDateTime DESC でソート
            - ORDER BY はスピル対応のため、巨大データでも安全
            - 結果をソート済み一時ファイルに書き出す

        Step 2: PyArrow ストリーミング重複排除
            - ソート済みファイルをバッチ単位で読み込み
            - 同一キーの最初の行（= 最新 submitDateTime）のみを書き出す
            - メモリ使用量: O(バッチサイズ) ≈ 数十MB
        """
        import duckdb
        import shutil

        temp_new = self.data_path / f"_temp_new_{bin_id}_{master_type}.parquet"
        temp_sorted = self.data_path / f"_temp_sorted_{bin_id}_{master_type}.parquet"
        temp_dir = self.data_path / "_duckdb_temp"

        try:
            new_data.to_parquet(temp_new, compression="zstd", index=False)

            # 重複排除キーを決定
            if master_type == "financial_values":
                key_cols = ["docid", "key", "context_ref"]
            else:
                key_cols = ["docid", "key"]

            has_submit_dt = "submitDateTime" in new_data.columns
            key_csv = ", ".join(key_cols)
            if has_submit_dt:
                order_by = f"ORDER BY {key_csv}, submitDateTime DESC NULLS LAST"
            else:
                order_by = f"ORDER BY {key_csv}"

            master_esc = str(master_path).replace("'", "''")
            temp_esc = str(temp_new).replace("'", "''")
            sorted_esc = str(temp_sorted).replace("'", "''")

            # Step 1: DuckDB でソートしてファイル出力 (ORDER BY はスピル対応)
            temp_dir.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(config={
                "memory_limit": "4GB",
                "threads": 2,
                "preserve_insertion_order": False,
                "temp_directory": str(temp_dir),
            })
            try:
                con.execute("SET max_temp_directory_size='50GB'")
                con.execute(f"""
                    COPY (
                        SELECT * FROM read_parquet(
                            ['{master_esc}', '{temp_esc}'],
                            union_by_name=true
                        )
                        {order_by}
                    ) TO '{sorted_esc}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
            finally:
                con.close()
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            # Step 2: ソート済みファイルからストリーミング重複排除
            self._streaming_dedup(temp_sorted, output_path, key_cols)

            logger.debug(f"DuckDB マージ完了 (2段階): bin={bin_id}")

        finally:
            for f in [temp_new, temp_sorted]:
                if f.exists():
                    f.unlink()
            force_gc()

    @staticmethod
    def _streaming_dedup(
        sorted_path: Path, output_path: Path, key_cols: list[str]
    ) -> None:
        """ソート済み Parquet からストリーミングで重複排除する

        データがキー列でソートされているため、同一キーの行は連続して並ぶ。
        各グループの最初の行（= submitDateTime が最大の行）のみを保持する。
        メモリ使用量はバッチサイズ（数十 MB）のみ。
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        reader = pq.ParquetFile(str(sorted_path))
        writer = None
        prev_key = None
        actual_key_cols = None

        for batch in reader.iter_batches(batch_size=50_000):
            table = pa.Table.from_batches([batch])
            n = len(table)
            if n == 0:
                continue

            # 初回バッチで実際のキーカラムを確定
            if actual_key_cols is None:
                actual_key_cols = [c for c in key_cols if c in table.column_names]

            # キー列を Python リストに変換してバッチ内比較
            key_lists = {c: table.column(c).to_pylist() for c in actual_key_cols}
            keep = []

            for i in range(n):
                current_key = tuple(key_lists[c][i] for c in actual_key_cols)
                if current_key != prev_key:
                    keep.append(True)
                    prev_key = current_key
                else:
                    keep.append(False)

            filtered = table.filter(pa.array(keep, type=pa.bool_()))

            if len(filtered) > 0:
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(output_path), filtered.schema, compression="zstd"
                    )
                writer.write_table(filtered)

        if writer:
            writer.close()
