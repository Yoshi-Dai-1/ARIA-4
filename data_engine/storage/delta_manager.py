"""
Delta Manager — GHA Worker/Merger 間のデルタファイル管理を担当するモジュール。

CatalogManager から分離された関心事:
- デルタファイル（中間成果物）の保存・収集・マージ
- チャンク完了フラグの管理
- 一時ファイルのクリーンアップ
"""

import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
from huggingface_hub import CommitOperationDelete, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from data_engine.core.models import ARIA_SCHEMAS


class DeltaManager:
    """GHA Worker/Merger 間のデルタファイル管理"""

    def __init__(self, hf_storage, data_path: Path, paths: Dict[str, str], clean_fn: Callable = None):
        """
        Args:
            hf_storage: HfStorage インスタンス（アップロード・コミットに使用）
            data_path: ローカルデータディレクトリ
            paths: 内部キーからリポジトリパスへの対応表
            clean_fn: DataFrame に適用するクレンジング関数
        """
        self.storage = hf_storage
        self.data_path = data_path
        self.paths = paths
        self._clean_fn = clean_fn

    def save_delta(
        self,
        key: str,
        df: pd.DataFrame,
        run_id: str,
        chunk_id: str,
        custom_filename: str = None,
        defer: bool = False,
        local_only: bool = False,
    ) -> bool:
        """
        デルタファイルを保存してアップロード。
        local_only=True の場合、HFにはアップロードせずローカルディレクトリに保存のみ行う (GHA Artifact用)。
        """
        if df.empty:
            return True

        if custom_filename:
            filename = custom_filename
        else:
            filename = f"{Path(self.paths[key]).stem}.parquet"

        # リポジトリ内パス
        delta_repo_path = f"temp/deltas/{run_id}/{chunk_id}/{filename}"

        # ローカル保存先 (Mergerが収集しやすいように構造化)
        local_delta_dir = self.data_path / "deltas" / str(run_id) / str(chunk_id)
        local_delta_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_delta_dir / filename

        # 【絶対ガード】保存直前に最終クレンジング
        if self._clean_fn:
            df = self._clean_fn(key, df)

        # 【工学的主権】物理型を強制同期して型ブレ（object型）を最小化
        df = df.convert_dtypes()

        # 【Phase 3: 金型アーキテクチャ】明示スキーマで型ブレを物理的に排除
        schema = ARIA_SCHEMAS.get(key)
        df.to_parquet(local_file, index=False, compression="zstd", schema=schema)

        if local_only:
            logger.debug(f"Delta saved locally (local_only): {local_file}")
            return True

        return self.storage.upload_raw(local_file, delta_repo_path, defer=defer)

    def mark_chunk_success(self, run_id: str, chunk_id: str, defer: bool = False, local_only: bool = False) -> bool:
        """チャンク処理成功フラグ (_SUCCESS) を作成"""
        success_repo_path = f"temp/deltas/{run_id}/{chunk_id}/_SUCCESS"

        local_delta_dir = self.data_path / "deltas" / str(run_id) / str(chunk_id)
        local_delta_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_delta_dir / "_SUCCESS"
        local_file.touch()

        if local_only:
            logger.debug(f"Chunk success marked locally: {local_file}")
            return True

        return self.storage.upload_raw(local_file, success_repo_path, defer=defer)

    def load_deltas(self, run_id: str) -> Dict[str, pd.DataFrame]:
        """
        全デルタを収集してマージ (Merger用)
        ローカル (data/deltas/{run_id}) とリモート (HF) の両方をスキャンする。
        """
        deltas = {}
        processed_chunks = set()

        # --- A. ローカルスキャン (GHA Artifacts 等でダウンロード済みの場合) ---
        local_run_dir = self.data_path / "deltas" / str(run_id)
        if local_run_dir.exists():
            logger.info(f"Checking local deltas in {local_run_dir}")
            for chunk_dir in local_run_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue

                chunk_id = chunk_dir.name
                if not (chunk_dir / "_SUCCESS").exists():
                    logger.warning(f"⚠️ 未完了のローカルチャンクをスキップ: {chunk_id}")
                    continue

                processed_chunks.add(chunk_id)
                for p_file in chunk_dir.glob("*.parquet"):
                    key = self._get_key_from_filename(p_file.name)
                    if key:
                        try:
                            df = pd.read_parquet(p_file)
                            deltas.setdefault(key, []).append(df)
                        except Exception as e:
                            logger.error(f"❌ ローカルデルタ読み込み失敗 ({p_file.name}): {e}")

        # --- B. リモートスキャン (Hugging Face Repository) ---
        if self.storage.api:
            try:
                folder = f"temp/deltas/{run_id}"
                files = []
                # 反映遅延に対処
                for attempt in range(3):
                    files = self.storage.api.list_repo_files(repo_id=self.storage.hf_repo, repo_type="dataset")
                    target_files = [f for f in files if f.startswith(folder)]
                    if target_files:
                        break
                    if attempt < 2:
                        logger.warning(f"リモートデルタフォルダが見つかりません。再試行中... ({attempt + 1}/3)")
                        time.sleep(10)

                # チャンクごとにグループ化
                remote_chunks = {}
                for f in target_files:
                    parts = f.split("/")
                    if len(parts) < 4:
                        continue
                    chunk_id = parts[3]
                    if chunk_id in processed_chunks:
                        continue
                    remote_chunks.setdefault(chunk_id, []).append(f)

                valid_remote_count = 0
                for chunk_id, file_list in remote_chunks.items():
                    if not any(f.endswith("_SUCCESS") for f in file_list):
                        logger.warning(f"⚠️ 未完了のリモートチャンクをスキップ: {chunk_id}")
                        continue

                    valid_remote_count += 1
                    for remote_path in file_list:
                        if remote_path.endswith("_SUCCESS"):
                            continue

                        key = self._get_key_from_filename(Path(remote_path).name)
                        if key:
                            attempts = 2
                            for att in range(attempts):
                                try:
                                    local_path = hf_hub_download(
                                        repo_id=self.storage.hf_repo,
                                        filename=remote_path,
                                        repo_type="dataset",
                                        token=self.storage.hf_token,
                                    )
                                    df = pd.read_parquet(local_path)
                                    deltas.setdefault(key, []).append(df)
                                    break
                                except Exception as e:
                                    if att == attempts - 1:
                                        logger.error(f"❌ リモートデルタ読み込み失敗 ({remote_path}): {e}")
                                    else:
                                        time.sleep(5)

                logger.info(f"収集結果: Local Chunks={len(processed_chunks)}, Remote Chunks={valid_remote_count}")

            except Exception as e:
                logger.error(f"リモートデルタ収集失敗: {e}")

        # --- C. 最終マージ ---
        merged = {}
        for key, df_list in deltas.items():
            if df_list:
                merged[key] = pd.concat(df_list, ignore_index=True)
            else:
                merged[key] = pd.DataFrame()
        return merged

    def _get_key_from_filename(self, fname: str) -> Optional[str]:
        """ファイル名から内部キーを判定する"""
        if fname == "documents_index.parquet":
            return "catalog"
        if fname == "stocks_master.parquet":
            return "master"
        if fname == "listing_history.parquet":
            return "listing"
        if fname == "name_history.parquet":
            return "name"
        if fname.startswith("financial_values_bin"):
            bin_id = fname.replace("financial_values_bin", "").replace(".parquet", "")
            return f"financial_bin{bin_id}"
        if fname.startswith("qualitative_text_bin"):
            bin_id = fname.replace("qualitative_text_bin", "").replace(".parquet", "")
            return f"text_bin{bin_id}"
        if fname.startswith("financial_values_"):
            sector = fname.replace("financial_values_", "").replace(".parquet", "")
            return f"financial_{sector}"
        if fname.startswith("qualitative_text_"):
            sector = fname.replace("qualitative_text_", "").replace(".parquet", "")
            return f"text_{sector}"
        return None

    def cleanup_deltas(self, run_id: str, cleanup_old: bool = True):
        """一時ファイルのクリーンアップ (Merger用)"""
        if not self.storage.api:
            return

        try:
            files = self.storage.api.list_repo_files(repo_id=self.storage.hf_repo, repo_type="dataset")
            delta_root = "temp/deltas"

            delete_files = []

            if cleanup_old:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc)
                expired_runs = set()

                for f in files:
                    if not f.startswith(delta_root):
                        continue
                    parts = f.split("/")
                    if len(parts) < 3:
                        continue
                    r_id = parts[2]

                    try:
                        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", r_id)
                        if date_match:
                            run_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            if (now - run_date).total_seconds() > 86400:
                                delete_files.append(f)
                                expired_runs.add(r_id)
                        else:
                            try:
                                timestamp = int(r_id)
                                if (now.timestamp() - timestamp) > 86400:
                                    delete_files.append(f)
                                    expired_runs.add(r_id)
                            except ValueError:
                                delete_files.append(f)
                                expired_runs.add(r_id)
                    except Exception:
                        pass

                if delete_files:
                    logger.info(f"古い一時フォルダを清掃中... (24時間以上経過: {len(expired_runs)} runs)")

            else:
                target_prefix = f"{delta_root}/{run_id}"
                delete_files = [f for f in files if f.startswith(target_prefix)]
                if delete_files:
                    logger.info(f"今回の一時ファイルを削除中... {run_id} ({len(delete_files)} files)")

            if not delete_files:
                return

            batch_size = 500
            total_batches = (len(delete_files) + batch_size - 1) // batch_size

            for i in range(0, len(delete_files), batch_size):
                batch = delete_files[i : i + batch_size]
                del_ops = [CommitOperationDelete(path_in_repo=p) for p in batch]

                batch_num = (i // batch_size) + 1
                commit_msg = f"Cleanup deltas (Batch {batch_num}/{total_batches})"

                max_retries = 10
                success = False
                for attempt in range(max_retries):
                    try:
                        self.storage.api.create_commit(
                            repo_id=self.storage.hf_repo,
                            repo_type="dataset",
                            operations=del_ops,
                            commit_message=commit_msg,
                            token=self.storage.hf_token,
                        )
                        success = True
                        break
                    except Exception as e:
                        if isinstance(e, HfHubHTTPError) and e.response.status_code == 429:
                            wait_time = int(e.response.headers.get("Retry-After", 60)) + 5
                            logger.warning(
                                f"Cleanup Rate limit exceeded. Waiting {wait_time}s... "
                                f"(Batch {batch_num}/{total_batches}, Attempt {attempt + 1})"
                            )
                            time.sleep(wait_time)
                            continue
                        logger.warning(
                            f"Cleanup error: {e}. Retrying... "
                            f"(Batch {batch_num}/{total_batches}, Attempt {attempt + 1})"
                        )
                        time.sleep(10 * (attempt + 1))

                if success:
                    logger.debug(f"Cleanup batch {batch_num}/{total_batches} done.")
                    if batch_num < total_batches:
                        time.sleep(2)
                else:
                    logger.error(f"❌ Cleanup batch {batch_num} failed permanently.")

            logger.success("Cleanup sequence completed.")

        except Exception as e:
            logger.error(f"クリーンアップ全体失敗: {e}")
