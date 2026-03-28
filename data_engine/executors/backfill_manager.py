import argparse
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

from data_engine.core.config import CONFIG

# 設定 (SSOT から取得)
DATA_PATH = CONFIG.DATA_PATH
META_DIR = DATA_PATH / "meta"
CURSOR_FILE = "backfill_cursor.json"
HF_REPO = CONFIG.HF_REPO
HF_TOKEN = CONFIG.HF_TOKEN
# 1回の遡り期間（7日＝1週間）
BACKFILL_DAYS = 7


def get_jst_today():
    return datetime.now(ZoneInfo("Asia/Tokyo")).date()


def get_dynamic_limit_date():
    """EDINET APIの仕様に基づく「取得可能な最古の日付」を動的に算出（10年前－5日の安全マージン）"""
    today = get_jst_today()
    try:
        # 安全マージンを加味し「ぴったり10年前」からさらに5日遡る
        base_limit = today.replace(year=today.year - 10)
    except ValueError:
        # うるう年（2月29日）のフォールバック
        base_limit = today.replace(year=today.year - 10, day=28)

    return base_limit - timedelta(days=5)


# 限界日（これより前はAPIリストからの取得が不可）
LIMIT_DATE = get_dynamic_limit_date()


def load_cursor():
    """HFからカーソルファイルをダウンロードして読み込む"""
    try:
        META_DIR.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"meta/{CURSOR_FILE}",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(DATA_PATH),
        )
        with open(local_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Cursor load failed (First run?): {e}")
        return None


def save_cursor(next_start_date_str):
    """次の開始日（より過去の日付）を保存してアップロード"""
    cursor_data = {"next_target_start": next_start_date_str}
    local_path = META_DIR / CURSOR_FILE

    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(cursor_data, f, indent=2)

    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"meta/{CURSOR_FILE}",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Update backfill cursor to {next_start_date_str}",
        )
        logger.info(f"Cursor updated: {next_start_date_str}")
    except Exception as e:
        logger.error(f"Failed to upload cursor: {e}")
        # カーソル更新失敗は致命的ではない（再実行されるだけ）が、ログは残す


def calculate_next_period():
    """
    カーソルを確認し、次に取得すべき期間（start_date, end_date）を決定する。
    期間は 'start_date' から 'end_date' へと未来に向かって進む（保全優先：古い方から順に確保）。
    """
    cursor = load_cursor()

    if cursor and "next_target_start" in cursor:
        # カーソルがある場合：その日付から BACKFILL_DAYS 分進める (過去->未来)
        start_date = datetime.strptime(cursor["next_target_start"], "%Y-%m-%d").date()
    else:
        # 初回：取得可能限界日（10年前−5日）を開始点とする
        # 理由：APIから消滅する直前のデータを確実に捉えるため、限界の少し手前から「助走」を開始する
        start_date = LIMIT_DATE

    # 終了日の計算
    end_date = start_date + timedelta(days=BACKFILL_DAYS - 1)

    # 未来に行き過ぎないようクリップ（昨日は日次バッチがやるので、その前日まで）
    yesterday = get_jst_today() - timedelta(days=1)

    if start_date >= yesterday:
        print("FINISHED")
        return None, None

    if end_date >= yesterday:
        end_date = yesterday - timedelta(days=1)

    return start_date, end_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Print dates and exit")
    parser.add_argument("--update-cursor", type=str, help="Update cursor to specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.update_cursor:
        # カーソル更新：完了した期間の「翌日」を次の開始点にする
        done_end_date = datetime.strptime(args.update_cursor, "%Y-%m-%d").date()
        next_start = done_end_date + timedelta(days=1)
        save_cursor(next_start.strftime("%Y-%m-%d"))
        return

    start, end = calculate_next_period()

    if start is None:
        print("FINISHED")  # GHA側で検知するためのキーワード
        return

    # 【追加】再試行期間の計算（カーソルの開始日から過去60日分）
    retry_start = start - timedelta(days=60)
    if retry_start < LIMIT_DATE:
        retry_start = LIMIT_DATE
    retry_end = start - timedelta(days=1)

    print(f"START={start.strftime('%Y-%m-%d')}")
    print(f"END={end.strftime('%Y-%m-%d')}")

    if retry_start < start:
        print(f"RETRY_START={retry_start.strftime('%Y-%m-%d')}")
        print(f"RETRY_END={retry_end.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
