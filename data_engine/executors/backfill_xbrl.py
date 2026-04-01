import argparse
import sys
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from loguru import logger
from huggingface_hub import hf_hub_download

from data_engine.catalog_manager import CatalogManager
from data_engine.core.config import CONFIG, TEMP_DIR
from data_engine.core.utils import normalize_code
from data_engine.engines.worker_engine import parse_worker
from data_engine.engines.parsing.edinet.fs_tbl import linkbasefile


def run_backfill(catalog: CatalogManager, run_id: str, limit: int = None):
    logger.info(f"Starting XBRL Backfill Process. Run ID: {run_id}")
    
    # 1. Fetch catalog
    df_catalog = catalog.catalog_df
    if df_catalog.empty:
        logger.info("Catalog is empty. Nothing to backfill.")
        return True

    # 2. Identify targets
    # conditions: xbrl_flag == True, doc_type_code in target types, status is not success or retracted, has raw_zip_path
    condition_xbrl = df_catalog["xbrl_flag"] == True
    condition_type = df_catalog["doc_type"].isin(CONFIG.XBRL_TARGET_DOC_TYPES)
    condition_status = ~df_catalog["processed_status"].isin(["success", "retracted"])
    condition_zip = df_catalog["raw_zip_path"].notna() & (df_catalog["raw_zip_path"] != "")

    df_targets = df_catalog[condition_xbrl & condition_type & condition_status & condition_zip]
    
    if df_targets.empty:
        logger.info("No documents require backfilling.")
        return True
        
    if limit:
        df_targets = df_targets.head(limit)

    target_count = len(df_targets)
    logger.info(f"Found {target_count} documents for backfill.")

    # 3. Process setup
    loaded_acc = {}
    tasks = []
    potential_catalog_records = {}

    for _, row in df_targets.iterrows():
        doc_id = row["doc_id"]
        raw_zip_path = row["raw_zip_path"]

        # Reconstruct exactly what parse_worker needs
        bridge_row = {"jcn": row.get("jcn"), "edinet_code": row.get("edinet_code"), "code": row.get("code")}
        bin_id = catalog.merger.get_bin_id(bridge_row)

        record = {
            "docID": doc_id,
            "processed_status": row.get("processed_status"),
            "bin_id": bin_id,
            "accounting_standard": row.get("accounting_standard")
        }
        potential_catalog_records[doc_id] = record

        # Download ZIP from HF
        local_zip = None
        try:
            local_zip = hf_hub_download(
                repo_id=CONFIG.HF_REPO,
                filename=raw_zip_path,
                repo_type="dataset",
                token=CONFIG.HF_TOKEN,
                local_dir=str(CONFIG.DATA_PATH),
            )
        except Exception as e:
            logger.error(f"Failed to download ZIP for {doc_id} from HF: {e}")
            record["processed_status"] = "failure"
            continue

        local_zip_path = Path(local_zip)
        
        # Taxonomy check
        detect_dir = TEMP_DIR / f"detect_backfill_{doc_id}"
        try:
            lb = linkbasefile(zip_file_str=str(local_zip_path), temp_path_str=str(detect_dir))
            lb.read_linkbase_file()
            ty = lb.detect_account_list_year()

            if ty == "-":
                raise ValueError(f"Taxonomy year not identified for {doc_id}")

            if ty not in loaded_acc:
                acc = catalog.edinet.get_account_list(ty)
                if not acc:
                    raise ValueError(f"Taxonomy version '{ty}' not found via EDINET API")
                loaded_acc[ty] = acc

            # Meta row expected by parse_worker
            meta_row = row.to_dict()
            tasks.append((doc_id, meta_row, loaded_acc[ty], local_zip_path))
            
        except Exception as e:
            logger.error(f"Taxonomy check failed for {doc_id}: {e}")
            record["processed_status"] = "failure"
        finally:
            if detect_dir.exists():
                shutil.rmtree(detect_dir)

    # 4. Execute parsing
    all_quant_dfs = []
    all_text_dfs = []
    processed_infos = []
    BATCH_PARALLEL_SIZE = 8

    if tasks:
        logger.info(f"Executing parses... (Task Count: {len(tasks)})")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=CONFIG.PARALLEL_WORKERS) as executor:
            for i in range(0, len(tasks), BATCH_PARALLEL_SIZE):
                batch = tasks[i : i + BATCH_PARALLEL_SIZE]
                futures = [executor.submit(parse_worker, t) for t in batch]

                for f in as_completed(futures):
                    did, res_df, err, accounting_std = f.result()

                    target_rec = potential_catalog_records.get(did)
                    if not target_rec:
                        continue
                        
                    if err:
                        logger.error(f"Parse failed: {did} - {err}")
                        if "No objects to concatenate" not in err:
                            target_rec["processed_status"] = "failure"
                    elif res_df is not None:
                        if target_rec.get("processed_status") != "failure":
                            target_rec["processed_status"] = "parsed"

                        quant_only = res_df[res_df["isTextBlock_flg"] == 0]
                        if not quant_only.empty:
                            all_quant_dfs.append(quant_only)

                        txt_only = res_df[res_df["isTextBlock_flg"] == 1]
                        if not txt_only.empty:
                            all_text_dfs.append(txt_only)

                        if accounting_std:
                            target_rec["accounting_standard"] = str(accounting_std)

                        meta_row = df_targets[df_targets["doc_id"] == did].iloc[0]
                        processed_infos.append(
                            {
                                "doc_id": did,
                                "bin": target_rec["bin_id"]
                            }
                        )
                logger.info(f"Progress: {min(i + BATCH_PARALLEL_SIZE, len(tasks))} / {len(tasks)} tasks.")

    # 5. Merge and Upload
    for record in potential_catalog_records.values():
        if record["processed_status"] == "parsed":
            record["processed_status"] = "success"

    processed_df = pd.DataFrame(processed_infos).drop_duplicates()
    if not processed_df.empty:
        bins = processed_df["bin"].unique()

        # Merge Quant
        if all_quant_dfs:
            full_quant_df = pd.concat(all_quant_dfs, ignore_index=True)
            for b_val in bins:
                bin_docids = processed_df[processed_df["bin"] == b_val]["doc_id"].tolist()
                sec_quant = full_quant_df[full_quant_df["docid"].isin(bin_docids)]
                if not sec_quant.empty:
                    catalog.merger.merge_and_upload(
                        b_val, "financial_values", sec_quant, worker_mode=True,
                        catalog_manager=catalog, run_id=run_id, defer=True
                    )

        # Merge Text
        if all_text_dfs:
            full_text_df = pd.concat(all_text_dfs, ignore_index=True)
            for b_val in bins:
                bin_docids = processed_df[processed_df["bin"] == b_val]["doc_id"].tolist()
                sec_text = full_text_df[full_text_df["docid"].isin(bin_docids)]
                if not sec_text.empty:
                    catalog.merger.merge_and_upload(
                        b_val, "qualitative_text", sec_text, worker_mode=True,
                        catalog_manager=catalog, run_id=run_id, defer=True
                    )

    # 6. Commit outputs to HF
    logger.info("Committing data outputs to HF...")
    catalog.hf.push_commit(message=f"Backfill Data outputs for {run_id}")

    # 7. Update Catalog Index (Only delta, inplace)
    logger.info("Updating Catalog Index flags...")
    has_updates = False
    for doc_id, rec in potential_catalog_records.items():
        if rec["processed_status"] in ["success", "failure"]:
            has_updates = True
            idx = catalog.catalog_df["doc_id"] == doc_id
            if idx.any():
                catalog.catalog_df.loc[idx, "processed_status"] = rec["processed_status"]
                if rec.get("accounting_standard"):
                    catalog.catalog_df.loc[idx, "accounting_standard"] = rec["accounting_standard"]
            
    if has_updates:
        catalog.hf.save_and_upload("catalog", catalog.catalog_df, defer=True)
        catalog.hf.push_commit(message=f"Backfill Catalog Delta for {run_id}")

    logger.success("Backfill process completed successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ARIA XBRL Backfill Execution")
    parser.add_argument("--run-id", type=str, help="Execution Run ID")
    parser.add_argument("--limit", type=int, help="Limit number of documents to backfill", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("backfill_%Y%m%d_%H%M%S")
    
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    catalog = CatalogManager(sync_master=False, force_refresh=False)
    run_backfill(catalog, run_id, args.limit)
