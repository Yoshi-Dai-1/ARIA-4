import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from data_engine.catalog_manager import CatalogManager
from data_engine.core.config import CONFIG
from data_engine.core.models import SCHEMA_INDEX
from data_engine.core.network_utils import patch_all_networking
from data_engine.engines.market_engine import MarketDataEngine

# グローバル設定は CONFIG インスタンス化時に適用済み


def run_market_pipeline(target_date: str, mode: str = "all"):
    logger.info(f"=== Market Data Pipeline Started (Target Date: {target_date}) ===")

    # 全体的な通信の堅牢化を適用
    patch_all_networking()

    # 【究極の統合】システムの SSOT から初期化 (環境変数バリデーション含む)
    # Market Data 更新時は EDINET API を使用しないため、edinet=False で初期化して API KEY 依存を排除
    # 指数は全上場銘柄を対象とするため、内部スコープは "All" に固定 (ユーザー設定を汚染しない)
    catalog = CatalogManager(edinet=False, sync_master=False, scope="All")

    # 初期化 (物理パスは CONFIG から取得)
    data_path = CONFIG.DATA_PATH
    temp_dir = CONFIG.TEMP_DIR
    data_path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    engine = MarketDataEngine(data_path)

    import shutil

    try:
        # 1. Stock Master Update (Attributes only, NO Active/Inactive Logic)
        if mode in ["all", "master"]:
            try:
                new_master = engine.fetch_jpx_master()

                # 【究極の統合】カタログマネージャの属性承継ロジックに委ねる
                # これにより、最新の社名(EDINET)に対し、JPXの業種(sector)・市場(market)属性が引き継がれる。
                # 生殺与奪権(is_active)は完全にEDINET(catalog_manager側)に移譲済みのため、ここでは関与しない。
                if catalog.update_stocks_master(new_master):
                    logger.success("Market Master attributes (sector/market) reconciled successfully.")
                else:
                    raise ValueError("Failed to reconcile market master attributes.")
            except Exception as e:
                logger.critical(f"Stock Master更新失敗 (Fatal): {e}")
                raise  # 即座に停止し、部分的なコミットを防止する

        # 2. Index Updates (Nikkei225, TOPIX, etc.)
        if mode in ["all", "indices"]:
            # 動的に全戦略を取得
            indices = list(engine.strategies.keys())

            for index_name in indices:
                logger.info(f"--- Processing {index_name} ---")
                # 名前解決の便宜上の初期化 (破損防止のためカラムを明示)
                history_cols = ["date", "index_name", "code", "type", "old_value", "new_value"]
                df_hist_current = pd.DataFrame(columns=history_cols)
                hist_path = f"master/indices/{index_name}/history.parquet"
                local_hist = data_path / f"{index_name}_history.parquet"
                try:
                    # A. Fetch Latest Data
                    df_new = engine.fetch_index_data(index_name)
                    # 【修正】Nikkei High Dividend 50 等、銘柄数が少ない指数を考慮して閾値を 40 に緩和
                    if df_new.empty or len(df_new) < 40:
                        logger.error(f"取得データが少なすぎます ({len(df_new)} rows). スキップします。")
                        continue

                    # Clean df_new just in case
                    if "rec" in df_new.columns:
                        df_new.drop(columns=["rec"], inplace=True)

                    # B. Save Snapshot (Year partitioning)
                    # path: master/indices/{index}/constituents/year={YYYY}/data_{YYYYMMDD}.parquet
                    year = target_date[:4]
                    snap_path = (
                        f"master/indices/{index_name}/constituents/"
                        f"year={year}/data_{target_date.replace('-', '')}.parquet"
                    )
                    local_snap = data_path / f"{index_name}_{target_date}.parquet"
                    # 【Phase 3 注記】指数構成銘柄の動的カラム構成のため、固定スキーマ不適用
                    df_new.to_parquet(local_snap, index=False, compression="zstd")

                    catalog.hf.upload_raw(local_snap, snap_path, defer=True)
                    logger.info(f"Snapshot staged: {snap_path}")

                    # C. Update History (Events)
                    # 前日のSnapshotを探す
                    try:
                        dt_target = datetime.strptime(target_date, "%Y-%m-%d")
                        dt_prev = dt_target - timedelta(days=1)
                        prev_date = dt_prev.strftime("%Y-%m-%d")  # "YYYY-MM-DD"
                        prev_year = prev_date[:4]
                        prev_fname = f"data_{prev_date.replace('-', '')}.parquet"
                        prev_path = f"master/indices/{index_name}/constituents/year={prev_year}/{prev_fname}"

                        # 前日Snapshotのロード (Catalog経由でダウンロード)
                        try:
                            from huggingface_hub import hf_hub_download

                            hf_hub_download(
                                repo_id=CONFIG.HF_REPO,
                                filename=prev_path,
                                token=CONFIG.HF_TOKEN,
                                local_dir=str(temp_dir),  # ディレクトリ指定
                                local_dir_use_symlinks=False,
                            )
                            downloaded_path = temp_dir / prev_path
                            if downloaded_path.exists():
                                df_old = pd.read_parquet(downloaded_path)
                                # 【重要】既存データからの汚染除去
                                if "rec" in df_old.columns:
                                    df_old.drop(columns=["rec"], inplace=True)
                                logger.info(f"Loaded Previous Snapshot: {prev_date}")
                            else:
                                logger.warning(f"Previous Snapshot file not found at local: {downloaded_path}")
                                df_old = pd.DataFrame(columns=["code", "weight"])

                        except Exception as e_dl:
                            logger.warning(f"Previous Snapshot not found in Repo ({prev_date}): {e_dl}")
                            df_old = pd.DataFrame(columns=["code", "weight"])

                        # Diff生成
                        diff_events = pd.DataFrame()
                        if not df_old.empty:
                            diff_events = engine.generate_index_diff(index_name, df_old, df_new, target_date)
                        else:
                            logger.info(f"Initial run for {index_name}. Baseline established.")

                        if not diff_events.empty:
                            # Historyファイルのロードと追記
                            # master/indices/{index_name}/history.parquet

                            # 既存History取得
                            try:
                                hf_hub_download(
                                    repo_id=CONFIG.HF_REPO,
                                    filename=hist_path,
                                    token=CONFIG.HF_TOKEN,
                                    local_dir=str(temp_dir),
                                    local_dir_use_symlinks=False,
                                )
                                dl_hist_path = temp_dir / hist_path
                                if dl_hist_path.exists():
                                    df_hist_current = pd.read_parquet(dl_hist_path)
                                    # 【重要】既存データからの汚染除去
                                    if "rec" in df_hist_current.columns:
                                        df_hist_current.drop(columns=["rec"], inplace=True)
                                else:
                                    df_hist_current = pd.DataFrame(columns=history_cols)
                            except Exception:
                                df_hist_current = pd.DataFrame(columns=history_cols)

                            # Merge
                            df_hist_new = pd.concat([df_hist_current, diff_events], ignore_index=True).drop_duplicates()
                            # 【重要】保存前最終チェック
                            if "rec" in df_hist_new.columns:
                                df_hist_new.drop(columns=["rec"], inplace=True)

                            # Save (Deferred)
                            df_hist_new.to_parquet(local_hist, index=False, compression="zstd", schema=SCHEMA_INDEX)
                            catalog.hf.upload_raw(local_hist, hist_path, defer=True)
                            logger.info(f"History staged: {index_name}")
                        elif df_hist_current.empty:
                            # Save (Deferred)
                            df_hist_current.to_parquet(local_hist, index=False, compression="zstd", schema=SCHEMA_INDEX)
                            catalog.hf.upload_raw(local_hist, hist_path, defer=True)
                            logger.info(f"History initialized: {index_name}")
                        else:
                            logger.info(f"No changes detected for {index_name}")

                    except Exception as e_hist:
                        logger.error(f"Failed to update history for {index_name}: {e_hist}")

                except Exception as e:
                    logger.error(f"{index_name} 更新失敗: {e}")

        # Final Push
        if catalog.push_commit(f"Market Data Update: {target_date}"):
            logger.success("=== Market Data Pipeline Completed ===")
        else:
            logger.error("=== Market Data Pipeline Failed (Push Error) ===")
            sys.exit(1)

    finally:
        # 一時ファイルの削除
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary files cleaned up.")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", type=str, help="YYYY-MM-DD (Default: Yesterday)")
    parser.add_argument("--mode", type=str, choices=["all", "master", "indices"], default="all", help="Execution mode")
    args = parser.parse_args()

    # 日付計算
    if args.target_date:
        t_date = args.target_date
    else:
        # デフォルトは昨日
        t_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    run_market_pipeline(t_date, args.mode)
