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

        # 1. 既存データのロード & マージ & 重複排除
        #    【メモリ最適化】DuckDB のディスクベースエンジンを使用し、
        #    pd.concat の 2X+Y メモリピークを回避する。
        #    DuckDB は独自バッファ管理で大規模 Bin (3GB+) も安全に処理可能。
        try:
            m_path = hf_hub_download(repo_id=self.hf_repo, filename=repo_path, repo_type="dataset", token=self.hf_token)
            combined_df = self._merge_with_duckdb(m_path, new_data, master_type, bin_id)
        except Exception as e:
            # 【究極の統一】共通ヘルパー is_404 を使用し、物理的なファイル不在 (404) の時のみ新規作成を行う。
            # 直前の HfStorage において修正された判定基準と完全に同期させることで、システム全体の整合性を担保する。
            if is_404(e):
                logger.info(f"新規Master作成: bin={bin_id} ({master_type})")
                combined_df = new_data
            else:
                # 503 等の通信障害時は、既存データを救うため例外をそのまま投げる（中断）。
                logger.error(f"Master取得中に通信エラーが発生しました (404以外): {e}")
                raise e

        # 2. 保存とアップロード
        local_file = self.data_path / f"master_bin{bin_id}_{master_type}.parquet"
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
                # 【メモリ最適化】ファイルに保存済み・パス参照のみ保持されるため DataFrame は不要
                del combined_df
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
                    del combined_df
                    force_gc()
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

    def _merge_with_duckdb(
        self, master_path: str, new_data: pd.DataFrame, master_type: str, bin_id: str
    ) -> pd.DataFrame:
        """DuckDB を使用したディスクベースのマージ・重複排除

        pd.concat の 2X+Y メモリピーク問題を根本解決する。
        DuckDB は独自バッファ管理を持ち、Parquet を直接読み取って
        結合・ソート・重複排除を SQL で実行するため、
        Python ヒープに全データを載せる必要がない。

        処理フロー:
        1. new_data を一時 Parquet ファイルに書き出す
        2. DuckDB で既存マスタと一時ファイルを結合 (union_by_name でスキーマ差異を吸収)
        3. ROW_NUMBER() Window 関数で重複排除 (submitDateTime DESC, 最新優先)
        4. 結果を pandas DataFrame として返却
        5. 一時ファイルを確実に削除
        """
        import duckdb

        temp_new = self.data_path / f"_temp_new_{bin_id}_{master_type}.parquet"

        try:
            # 1. new_data を一時ファイルに書き出し (スキーマ互換性のため)
            new_data.to_parquet(temp_new, compression="zstd", index=False)

            # 2. 重複排除キーを決定
            if master_type == "financial_values":
                partition_cols = "docid, key, context_ref"
            else:
                partition_cols = "docid, key"

            # 3. submitDateTime の存在チェック (旧コードの防御的ガードを継承)
            has_submit_dt = "submitDateTime" in new_data.columns
            if has_submit_dt:
                order_clause = "ORDER BY submitDateTime DESC NULLS LAST"
            else:
                order_clause = "ORDER BY 1"  # 決定論的だが順序無関係 (submitDateTime 非存在時)

            # 4. DuckDB で結合 → 重複排除
            #    ROW_NUMBER() OVER (PARTITION BY ... ORDER BY submitDateTime DESC) = 1
            #    は pandas の sort_values(ascending=False) + drop_duplicates(keep="first") と等価
            #    union_by_name=true: pd.concat 同等のスキーマ差異吸収 (欠損列は NULL 埋め)
            master_path_esc = str(master_path).replace("'", "''")
            temp_new_esc = str(temp_new).replace("'", "''")

            con = duckdb.connect()
            try:
                result_df = con.execute(f"""
                    SELECT * EXCLUDE (rn) FROM (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY {partition_cols}
                            {order_clause}
                        ) AS rn
                        FROM read_parquet(
                            ['{master_path_esc}', '{temp_new_esc}'],
                            union_by_name=true
                        )
                    ) WHERE rn = 1
                """).fetchdf()
            finally:
                con.close()

            logger.debug(f"DuckDB マージ完了: bin={bin_id} ({len(result_df)} rows)")
            return result_df

        finally:
            # 一時ファイルの確実な削除
            if temp_new.exists():
                temp_new.unlink()
            force_gc()
