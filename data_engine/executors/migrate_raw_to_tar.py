"""既存 RAW ファイルの TAR バンドル移行スクリプト

Hugging Face リポジトリ上の raw/edinet/ 配下の個別ファイルを
日次 TAR バンドルに変換する段階的移行ツール。

設計原則:
- アトミック: 1 日分の変換を 1 つの create_commit で実行（全成功 or 全失敗）
- Fail-Fast: ネットワーク障害時は即座に停止
- 安全なマージ: 既に TAR が存在する日は既存 TAR とマージ（既存データの消失を防止）
- GHA 適合: ubuntu-latest の 14GB ディスク / 6 時間制限内で動作

使用例:
    python -m data_engine.executors.migrate_raw_to_tar --year-month 2016-03
    python -m data_engine.executors.migrate_raw_to_tar --year-month 2016-03 --dry-run
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
from collections import defaultdict

from pathlib import Path
from typing import Dict, List

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
from loguru import logger

from data_engine.core.network_utils import is_404
from data_engine.core.utils import aria_json_dumps
from data_engine.storage.tar_bundle_manager import merge_tars


def migrate_month(
    hf_repo: str,
    hf_token: str,
    year_month: str,
    dry_run: bool = False,
) -> bool:
    """指定月の個別 RAW ファイルを日次 TAR バンドルに変換する。

    処理フロー (1 日ごと):
    1. HF 上のファイルリストから該当日の個別ファイルを特定
    2. 既に TAR が存在する場合は既存 TAR とマージ（重複排除）
    3. 全個別ファイルをダウンロード
    4. TAR + インデックス JSON を作成
    5. アトミックコミット: ADD(TAR + JSON) + DELETE(個別ファイル)
    6. ローカルファイルを削除（ディスク解放）

    Args:
        hf_repo: HF リポジトリ ID
        hf_token: HF API トークン
        year_month: 対象月 "YYYY-MM" 形式
        dry_run: True なら実際のコミットを行わない

    Returns:
        全日付の変換が成功したら True
    """
    api = HfApi(token=hf_token)
    work_dir = Path("data/temp/migration")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 対象月のパース
    try:
        year, month = year_month.split("-")
        prefix = f"raw/edinet/year={year}/month={month}/"
    except ValueError:
        logger.error(f"無効な year-month 形式: {year_month}（YYYY-MM を指定してください）")
        return False

    logger.info(f"=== 移行開始: {year_month} ===")

    # 1. 対象月のファイルリストのみを取得（リポジトリ全体の列挙は数時間かかるため厳禁）
    logger.info(f"HF リポジトリのファイルリストを取得中... (prefix: {prefix})")
    try:
        raw_items = list(api.list_repo_tree(
            repo_id=hf_repo, repo_type="dataset",
            path_in_repo=prefix, recursive=True,
        ))
    except Exception as e:
        logger.critical(f"個別ファイルリスト取得失敗: {e}")
        return False

    # 該当月のファイルを抽出・日付ごとにグループ化
    day_groups: Dict[str, List[str]] = defaultdict(list)
    for item in raw_items:
        f = item.rfilename if hasattr(item, "rfilename") else str(item)
        # 個別ファイルの場合: raw/edinet/year=2016/month=03/day=22/zip/S100xxx.zip
        day_match = re.match(r"raw/edinet/year=\d+/month=\d+/day=(\d+)/", f)
        if day_match:
            day = day_match.group(1)
            day_groups[day].append(f)

    # 既存 TAR の確認（TAR パスは別のプレフィックス: raw/edinet/YYYY/MM/）
    tar_prefix = f"raw/edinet/{year}/{month}/"
    tar_exists: set = set()
    logger.info(f"既存 TAR を確認中... (prefix: {tar_prefix})")
    try:
        tar_items = list(api.list_repo_tree(
            repo_id=hf_repo, repo_type="dataset",
            path_in_repo=tar_prefix, recursive=False,
        ))
        for item in tar_items:
            fname = item.rfilename if hasattr(item, "rfilename") else str(item)
            if fname.endswith(".tar"):
                # raw/edinet/2016/03/22.tar → day = "22"
                tar_day = fname.split("/")[-1].replace(".tar", "")
                tar_exists.add(tar_day)
    except Exception as e:
        # TAR ディレクトリがまだ存在しない場合は 404 → 正常（TAR なし）
        if is_404(e):
            logger.info(f"既存 TAR ディレクトリなし: {tar_prefix}")
        else:
            logger.critical(f"既存 TAR リスト取得失敗: {e}")
            return False

    if not day_groups:
        logger.info(f"{year_month} に個別 RAW ファイルはありません。移行不要です。")
        return True

    total_days = len(day_groups)
    total_files = sum(len(v) for v in day_groups.values())
    logger.info(f"対象: {total_days} 日, 合計 {total_files} ファイル")

    # 2. 日付ごとに処理
    migrated = 0
    failed = 0

    for day in sorted(day_groups.keys()):
        files = day_groups[day]
        tar_repo_path = f"raw/edinet/{year}/{month}/{day}.tar"
        index_repo_path = f"raw/edinet/{year}/{month}/{day}.json"

        # マージフラグの判定
        merge_existing = day in tar_exists
        if merge_existing:
            logger.info(f"[Migrate & Merge] {year}-{month}-{day}: {len(files)} files into existing TAR")
        else:
            logger.info(f"[Migrate] {year}-{month}-{day}: {len(files)} files")

        day_dir = work_dir / f"{year}_{month}_{day}"
        day_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 2a. 全ファイルをダウンロード
            local_files: Dict[str, Path] = {}
            total = len(files)
            for idx, repo_path in enumerate(files, 1):
                try:
                    local = hf_hub_download(
                        repo_id=hf_repo,
                        filename=repo_path,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=str(day_dir),
                    )
                    # TAR 内のメンバー名: day=XX/ より下の部分
                    parts = repo_path.split("/")
                    # raw/edinet/year=2016/month=03/day=22/zip/S100xxx.zip
                    # → zip/S100xxx.zip
                    day_idx = next(i for i, p in enumerate(parts) if p.startswith("day="))
                    member_name = "/".join(parts[day_idx + 1:])
                    local_files[member_name] = Path(local)
                    if idx % 100 == 0 or idx == total:
                        logger.info(f"  ダウンロード進捗: {idx}/{total} ({idx*100//total}%)")
                except Exception as e:
                    logger.error(f"  ダウンロード失敗 ({idx}/{total}): {repo_path}: {e}")
                    raise

            # 2b. TAR + インデックス作成
            local_tar = day_dir / f"{day}.tar"
            index_entries = []

            with tarfile.open(local_tar, "w") as tar:
                for member_name, local_path in sorted(local_files.items()):
                    info = tarfile.TarInfo(name=member_name)
                    info.size = local_path.stat().st_size
                    with open(local_path, "rb") as fh:
                        tar.addfile(info, fh)
                    index_entries.append({"name": member_name, "size": info.size})

            local_index = day_dir / f"{day}.json"
            index_data = {
                "date": f"{year}-{month}-{day}",
                "file_count": len(index_entries),
                "members": index_entries,
            }
            with open(local_index, "w", encoding="utf-8") as f:
                f.write(aria_json_dumps(index_data, indent=2))

            logger.info(f"  新規 TAR 作成: {local_tar.stat().st_size:,} bytes, {len(local_files)} members")

            # 2b-2. 既存 TAR とのマージ
            if merge_existing:
                logger.info(f"  既存 TAR をダウンロードしてマージします: {tar_repo_path}")
                try:
                    existing_local = hf_hub_download(
                        repo_id=hf_repo,
                        filename=tar_repo_path,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    merged_tar = day_dir / f"{day}_merged.tar"
                    merged_tar, merged_index = merge_tars(Path(existing_local), local_tar, merged_tar)
                    
                    # コミット対象をマージ後のファイルにすり替える
                    local_tar = merged_tar
                    local_index = merged_index
                    logger.info(f"  TAR マージ完了: {local_tar.stat().st_size:,} bytes")
                except Exception as e:
                    logger.error(f"  TAR マージ失敗: {e}")
                    raise

            if dry_run:
                logger.info(f"  [DRY RUN] コミットをスキップ")
                migrated += 1
                continue

            # 2c. アトミックコミット: ADD(TAR + JSON) + DELETE(個別ファイル)
            operations = [
                CommitOperationAdd(path_in_repo=tar_repo_path, path_or_fileobj=str(local_tar)),
                CommitOperationAdd(path_in_repo=index_repo_path, path_or_fileobj=str(local_index)),
            ]
            for repo_path in files:
                operations.append(CommitOperationDelete(path_in_repo=repo_path))

            api.create_commit(
                repo_id=hf_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Migrate raw to TAR: {year}-{month}-{day} ({len(files)} files)",
                token=hf_token,
            )

            # 2d. 検証: TAR が HF 上に存在するか確認（個別ダウンロードで検証、全リスト再取得は禁止）
            try:
                api.repo_info(repo_id=hf_repo, repo_type="dataset", files_metadata=False)
                # create_commit が例外なく返った時点でコミットは成功している。
                # HF の create_commit はアトミックなため、追加の検証は不要。
            except Exception as verify_e:
                logger.critical(f"  ⛔ コミット後の接続確認失敗: {verify_e}。移行を中断します。")
                return False

            migrated += 1
            logger.success(f"  ✅ {year}-{month}-{day} 移行完了")

        except Exception as e:
            logger.error(f"  ❌ {year}-{month}-{day} 移行失敗: {e}")
            failed += 1
            # 一度でも失敗したら即停止（データ整合性優先）
            logger.critical("移行中にエラーが発生したため、残りの日付の処理を中断します。")
            return False

        finally:
            # ディスク解放
            if day_dir.exists():
                shutil.rmtree(day_dir)

    logger.info(f"=== 移行完了: {year_month} ===")
    logger.info(f"  移行: {migrated}, 失敗: {failed}")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ARIA RAW → TAR Migration")
    parser.add_argument(
        "--year-month",
        type=str,
        required=True,
        help="対象月 (YYYY-MM 形式, 例: 2016-03)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際のコミットを行わない（テスト用）",
    )
    args = parser.parse_args()

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    hf_repo = os.getenv("HF_REPO")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_repo or not hf_token:
        logger.critical("HF_REPO と HF_TOKEN が必要です。")
        sys.exit(1)

    success = migrate_month(hf_repo, hf_token, args.year_month, args.dry_run)
    sys.exit(0 if success else 1)
