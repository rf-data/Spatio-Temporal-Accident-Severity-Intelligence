## harmonize_schema.py
# import
import json
import pandas as pd
import os
from pathlib import Path
import click

import src.utils.df_helper as dfh
import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.agent_helper as ah

from src.agentic_AI.report.raw_data_report import load_eda_summary

from src.core.session import session
from src.core.logger import create_logger

def extract_geo_cols(report):

    geo_cols = set()

    metrics = report.get("metrics", {})

    for file_data in metrics.values():

        for tool in file_data:
            if tool["tool_name"] == "detect_geo_columns":
                geo_cand = tool.recommendation_hint.get("geo_col_columns")
                geo_cols.update(geo_cand)

    return sorted(list(geo_cols))


def extract_datetime_cols(report):

    dt_cols = set()

    processing = report.get("metrics", {})

    for file_data in processing.values():

        for tool in file_data:

            if tool["tool_name"] == "detect_datetime_candidates":

                candidates = tool["metrics"].get("dt_candidates", {})

                dt_cols.update(candidates.get("single_col", []))
                dt_cols.update(candidates.get("splitted_cols", []))

    return sorted(dt_cols)


def extract_primary_keys(report):

    pk = set()

    merge_info = report.get("merge", {})

    for _, strategy in merge_info.items():

        keys = strategy.get("join_key")

        if keys:
            pk.update(keys)

    return sorted(pk)


# ------------------------------
# WRAPPER FUNCTION
# ------------------------------
@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def harmonize_schema(name):
    run_harmonize_schema(name)

    return 


# ------------------------------
# MAIN FUNCTION
# ------------------------------
def run_harmonize_schema(name):
    # (1) load config + parse arguments
    gh.load_env_vars()

    data_raw = os.getenv("PATH_RAW")
    data_processed = os.getenv("PATH_PROCESSED")

    config = fh.get_yaml_config(name)
    
    general_config = config.get("general_args", {})
    log_name = general_config["name_log"]
    name_logfile = general_config["name_logfile"]
    start_year = general_config.get("start_year", 1900)
    data_folder = Path(general_config["data_folder"])
    df_path = Path(f"{data_raw}/{data_folder}")

    # setup logger 
    logger = create_logger(name=log_name, file_name=name_logfile)
    session.logger = logger

    # args = ah.parse_args()
    # load report
    report_dict = load_eda_summary(general_config)
    report = report_dict["report"]
    files = report_dict["files"]

    # ------------------------------
    # REQUIRED COLUMNS
    # ------------------------------
    schema_info = {
        "primary_keys": extract_primary_keys(report),
        "datetime_cols": extract_datetime_cols(report),
        "geo_cols": extract_geo_cols(report)
        }

    required_columns = set(
            schema_info["primary_keys"]
            + schema_info["datetime_cols"]
            + schema_info["geo_cols"]
            )

    # ------------------------------
    # COLLECT UNION SCHEMA
    # ------------------------------
    column_config = config.get("columns", {})
    rename_dict = column_config["rename"]
    col_to_drop = column_config["to_drop"]

    all_columns = set()
    logger.info("Scanning schemas...")

    for file in files:
        logger.info("processing: %s (dtype=%s)", 
                    file, 
                    type(file))
        f_name = Path(file).stem
        path = f"{df_path}/{str(f_name).strip()}.csv"

        df = dfh.read_french_csv_smart(path, nrows=10)

        all_columns.update(df.columns)

    all_columns.update(required_columns)

    # print("[DEBUG] all_columns:\n", all_columns)
    all_col_clean = set([col for col in all_columns if col not in col_to_drop])
    all_columns = sorted(all_col_clean)

    all_cols_renamed = list(set([rename_dict[col] for col in all_columns]))
    logger.info("Unified schema (renamed):")
    logger.info(all_cols_renamed)

    # ------------------------------
    # HARMONIZE FILES
    # ------------------------------

    for file in files:
        f_name = Path(file).stem
        year = f_name.split("_")[1]

        if int(year) < start_year:
            logger.info("Skipping file from %s (start_year=%s)", 
                        year,
                        start_year)
            continue

        path = f"{df_path}/{str(f_name).strip()}.csv"

        df = dfh.read_french_csv_smart(path)

        # add missing columns
        missing_cols = []
        
        df_renamed = df.rename(columns=rename_dict)
        
        for col in all_cols_renamed:
            if col not in df_renamed.columns:
                df_renamed[col] = None
                missing_cols.append(col)

        if missing_cols:
            logger.info("Added columns: %s", missing_cols)

        # reorder columns
        df_renamed = df_renamed[all_cols_renamed]

        # save
        out_folder = f"{data_processed}/{data_folder}"
        out_name = f"{str(f_name).strip()}_harmonized"

        dfh.save_df_to_parquet(df_renamed, out_name, out_folder, chunked=True)
        # df_renamed.to_parquet(out_path)

        # print("Saved df to:\t", ph.shorten_path(out_path))

    logger.info("\nSchema harmonization complete.")

if __name__ == "__main__":
    harmonize_schema()