"""TAR バンドルマネージャ — ARIA の RAW ファイル永続化戦略の中核モジュール

Hugging Face の 100 万ファイル制限を永久的に回避するため、
1 日分の RAW ファイル（ZIP, PDF, English, Attach）を日次 TAR バンドルにまとめる。

設計原則:
- SSOT: TAR パスの導出は utils.get_tar_repo_path() に一元化
- Fail-Fast: ネットワーク障害で既存 TAR の状態が不明な場合、アップロードを拒否
- アトミック: push_commit のアトミック性を活用し、部分的なデータ消失を防止
- 重複排除: 既存 TAR とのマージ時にメンバー名ベースで重複を排除
"""

import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from data_engine.core.utils import aria_json_dumps, get_tar_repo_path


def bundle_raw_files_by_date(raw_edinet_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """ローカルの raw/edinet/ 配下のファイルを日付ごとに TAR バンドルにまとめる。

    Args:
        raw_edinet_dir: ローカルの raw/edinet/ ディレクトリ

    Returns:
        {tar_repo_path: (local_tar_path, local_index_path)} の辞書
        例: {"raw/edinet/2024/10/23.tar": ("/data/temp/.../23.tar", "/data/temp/.../23.json")}
    """
    if not raw_edinet_dir.exists():
        return {}

    # 1. ファイルを日付ごとにグループ化
    #    ローカルのディレクトリ構造: raw/edinet/year=YYYY/month=MM/day=DD/...
    date_groups: Dict[str, List[Path]] = defaultdict(list)

    for f in raw_edinet_dir.rglob("*"):
        if not f.is_file():
            continue
        # 相対パスから日付を構築: year=2024/month=10/day=23/...
        rel = f.relative_to(raw_edinet_dir)
        parts = rel.parts  # ('year=2024', 'month=10', 'day=23', 'zip', 'S100BK7G.zip')
        if len(parts) < 3:
            continue

        try:
            y = parts[0].split("=")[1]
            m = parts[1].split("=")[1]
            d = parts[2].split("=")[1]
            date_key = f"{y}-{m}-{d}"
        except (IndexError, ValueError):
            logger.warning(f"不明なディレクトリ構造をスキップ: {rel}")
            continue

        date_groups[date_key].append(f)

    if not date_groups:
        return {}

    logger.info(f"TAR バンドル対象: {len(date_groups)} 日付, 合計 {sum(len(v) for v in date_groups.values())} ファイル")

    # 2. 日付ごとに TAR を作成
    results: Dict[str, Tuple[Path, Path]] = {}
    temp_dir = raw_edinet_dir.parent.parent / "temp" / "tar_bundles"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for date_key, files in sorted(date_groups.items()):
        tar_repo_path = get_tar_repo_path(date_key)
        if not tar_repo_path:
            logger.warning(f"TAR パスを生成できません: {date_key}")
            continue

        local_tar = temp_dir / tar_repo_path.replace("/", "_")
        local_index = local_tar.with_suffix(".json")

        tar_path, index_path = _create_tar_with_index(
            files, local_tar, raw_edinet_dir, date_key
        )
        results[tar_repo_path] = (tar_path, index_path)

    return results


def _create_tar_with_index(
    files: List[Path],
    output_tar: Path,
    base_dir: Path,
    date_key: str,
) -> Tuple[Path, Path]:
    """TAR ファイルとサイドカーインデックスを作成する。

    TAR 内のメンバー名は、ローカルのディレクトリ構造から導出する。
    例: year=2024/month=10/day=23/zip/S100BK7G.zip → zip/S100BK7G.zip

    インデックス JSON は、各メンバーの名前とサイズを記録する。
    """
    index_entries = []
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_tar, "w") as tar:
        for f in sorted(files, key=lambda x: x.name):
            # メンバー名: year=.../month=.../day=.../ より下の部分
            rel = f.relative_to(base_dir)
            parts = rel.parts  # ('year=2024', 'month=10', 'day=23', 'zip', 'S100BK7G.zip')
            if len(parts) > 3:
                # 日付部分を除去: zip/S100BK7G.zip, pdf/S100BK7G.pdf, attach/S100BK7G/file.pdf
                member_name = str(Path(*parts[3:]))
            else:
                member_name = rel.name

            info = tar.gettarinfo(name=str(f), arcname=member_name)
            with open(f, "rb") as fh:
                tar.addfile(info, fh)

            index_entries.append({
                "name": member_name,
                "size": info.size,
            })

    # サイドカーインデックス JSON を生成
    index_path = output_tar.with_suffix(".json")
    index_data = {
        "date": date_key,
        "file_count": len(index_entries),
        "members": index_entries,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(aria_json_dumps(index_data, indent=2))

    logger.info(f"TAR 作成完了: {output_tar.name} ({len(files)} files, {output_tar.stat().st_size:,} bytes)")
    return output_tar, index_path


def extract_file_from_tar(
    tar_path: Path,
    member_name: str,
    dest_path: Path,
) -> bool:
    """TAR ファイルから特定のメンバーを抽出する。

    Args:
        tar_path: ローカルの TAR ファイルパス
        member_name: TAR 内のメンバー名（例: "zip/S100BK7G.zip"）
        dest_path: 抽出先のローカルパス

    Returns:
        抽出成功なら True
    """
    try:
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmember(member_name)
            extracted = tar.extractfile(member)
            if extracted is None:
                logger.error(f"TAR メンバーはファイルではありません: {member_name}")
                return False

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as out:
                out.write(extracted.read())

            return True
    except KeyError:
        logger.debug(f"TAR 内にメンバーが見つかりません: {member_name} in {tar_path}")
        return False
    except Exception as e:
        logger.error(f"TAR 抽出エラー: {member_name} from {tar_path}: {e}")
        return False


def extract_directory_from_tar(
    tar_path: Path,
    prefix: str,
    dest_dir: Path,
) -> int:
    """TAR ファイルから特定のプレフィックスに一致するファイルをすべて抽出する。

    English/Attach のディレクトリ単位の復元に使用する。
    例: prefix="english/S100BK7G/" → english/S100BK7G/ 配下の全ファイルを抽出

    Args:
        tar_path: ローカルの TAR ファイルパス
        prefix: TAR 内のプレフィックス
        dest_dir: 抽出先のディレクトリ

    Returns:
        抽出したファイル数
    """
    extracted_count = 0
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.startswith(prefix) and member.isfile():
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue

                    # プレフィックスを除去した相対パスで保存
                    rel_name = member.name[len(prefix):]
                    if not rel_name:
                        # プレフィックス自体がファイル名の場合
                        rel_name = Path(member.name).name

                    out_path = dest_dir / rel_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as out:
                        out.write(extracted.read())
                    extracted_count += 1
    except Exception as e:
        logger.error(f"TAR ディレクトリ抽出エラー: {prefix} from {tar_path}: {e}")

    return extracted_count


def merge_tars(
    existing_tar: Path,
    new_tar: Path,
    output_tar: Path,
) -> Tuple[Path, Path]:
    """2つの TAR ファイルをマージする（メンバー名ベースで重複排除）。

    既存 TAR の内容を優先し、新規 TAR から既存に無いメンバーのみを追加する。
    これにより、リトライ等で同一日付の TAR が複数回生成される場合でも、
    以前の run で保存したデータが消失しない。

    Args:
        existing_tar: HF からダウンロードした既存 TAR ファイル
        new_tar: 今回の run で作成した TAR ファイル
        output_tar: マージ結果の出力先

    Returns:
        (merged_tar_path, merged_index_path) のタプル
    """
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    index_entries = []

    with tarfile.open(output_tar, "w") as out:
        # 1. 既存 TAR の全メンバーをコピー（優先）
        existing_names = set()
        with tarfile.open(existing_tar, "r") as old:
            for member in old.getmembers():
                if member.isfile():
                    existing_names.add(member.name)
                    data = old.extractfile(member)
                    if data:
                        out.addfile(member, data)
                        index_entries.append({"name": member.name, "size": member.size})

        # 2. 新規 TAR から、既存に無いメンバーのみ追加
        added = 0
        with tarfile.open(new_tar, "r") as new:
            for member in new.getmembers():
                if member.isfile() and member.name not in existing_names:
                    data = new.extractfile(member)
                    if data:
                        out.addfile(member, data)
                        index_entries.append({"name": member.name, "size": member.size})
                        added += 1

    # インデックス再生成
    index_path = output_tar.with_suffix(".json")
    index_data = {
        "file_count": len(index_entries),
        "members": index_entries,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(aria_json_dumps(index_data, indent=2))

    logger.info(
        f"TAR マージ完了: {output_tar.name} "
        f"(既存: {len(existing_names)}, 追加: {added}, 合計: {len(index_entries)})"
    )
    return output_tar, index_path
