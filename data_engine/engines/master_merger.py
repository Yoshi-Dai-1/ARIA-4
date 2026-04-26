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
        #    【メモリ最適化】PyArrow ストリーミングでバッチ単位に処理し、
        #    Python ヒープに全データを載せない。巨大 Bin (数十 GB) でも安全。
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
        """純 PyArrow ストリーミング・マージ (DuckDB 不使用)

        メモリ使用量: O(new_data のキーセット + バッチサイズ)
        巨大マスタ (数十 GB) でもメモリ制約に一切左右されない。

        処理フロー:
        1. new_data のキーセットを構築（数千〜数万件、数 MB）
        2. マスタを 50,000 行ずつ読み込み、新データと重複するキーの行を除外
        3. 除外後の行を出力ファイルに書き出す
        4. new_data の全行を出力ファイルに追記
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 重複排除キーを決定
        if master_type == "financial_values":
            key_cols = ["docid", "key", "context_ref"]
        else:
            key_cols = ["docid", "key"]

        # 1. new_data のキーセットを構築 (O(new_data) メモリ ≈ 数 MB)
        actual_new_keys = [c for c in key_cols if c in new_data.columns]
        new_key_set = set(
            new_data[actual_new_keys].apply(tuple, axis=1).tolist()
        )

        # 2. スキーマ統合: マスタと new_data のカラムを統合
        master_schema = pq.read_schema(master_path)
        new_arrow = pa.Table.from_pandas(new_data, preserve_index=False)
        try:
            unified_schema = pa.unify_schemas(
                [master_schema, new_arrow.schema],
                promote_options="permissive",
            )
        except pa.ArrowInvalid:
            # 型の不整合時はマスタのスキーマを優先
            unified_schema = master_schema

        # 3. マスタをストリーミング読み込み → 重複キー除外 → 書き出し
        reader = pq.ParquetFile(master_path)
        writer = pq.ParquetWriter(
            str(output_path), unified_schema, compression="zstd"
        )

        try:
            for batch in reader.iter_batches(batch_size=50_000):
                table = pa.Table.from_batches([batch])
                n = len(table)
                if n == 0:
                    continue

                # バッチ内のキー列を取得して比較
                batch_key_cols = [c for c in key_cols if c in table.column_names]
                key_lists = {
                    c: table.column(c).to_pylist() for c in batch_key_cols
                }

                keep = []
                for i in range(n):
                    row_key = tuple(key_lists[c][i] for c in batch_key_cols)
                    keep.append(row_key not in new_key_set)

                filtered = table.filter(pa.array(keep, type=pa.bool_()))

                if len(filtered) > 0:
                    # マスタのスキーマを統合スキーマに合わせる
                    aligned = self._align_table(filtered, unified_schema)
                    writer.write_table(aligned)

            # 4. new_data を統合スキーマに合わせて追記
            aligned_new = self._align_table(new_arrow, unified_schema)
            writer.write_table(aligned_new)

        finally:
            writer.close()

        logger.debug(
            f"ストリーミングマージ完了: bin={bin_id} "
            f"(除外キー数: {len(new_key_set)})"
        )

    @staticmethod
    def _align_table(table: "pa.Table", target_schema: "pa.Schema") -> "pa.Table":
        """テーブルをターゲットスキーマに合わせる

        - ターゲットにあってテーブルにないカラム → NULL 列を追加
        - テーブルにあってターゲットにないカラム → 除外
        - 型が異なるカラム → 安全にキャスト
        """
        import pyarrow as pa

        columns = {}
        for field in target_schema:
            if field.name in table.column_names:
                col = table.column(field.name)
                if col.type != field.type:
                    try:
                        col = col.cast(field.type, safe=False)
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                        pass  # キャスト不可の場合は元の型を維持
                columns[field.name] = col
            else:
                # 欠損カラムを NULL で埋める
                columns[field.name] = pa.array(
                    [None] * len(table), type=field.type
                )

        return pa.table(columns)
