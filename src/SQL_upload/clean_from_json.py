## clean_from_json.py
# import
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import click

import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.df_helper as dfh

from src.agentic_AI.report.raw_data_report import load_eda_summary
from src.core.session import session
from src.core.logger import create_logger


def apply_drop_duplicates(df, action):

    subset = action.get("params", {}).get("subset")
    keep = action.get("params", {}).get("keep", "first")

    if subset:
        return df.drop_duplicates(subset=subset, keep=keep)

    return df.drop_duplicates(keep=keep)


def apply_parse_datetime(df, action):

    for col in action.get("target", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def apply_impute_nan(df, action):

    strategy = action.get("params", {}).get("strategy", "median")

    for col in action.get("target", []):

        if col not in df.columns:
            continue

        series = df[col]

        # numeric columns
        if pd.api.types.is_numeric_dtype(series):

            if strategy == "median":
                df[col] = series.fillna(series.median())

            elif strategy == "mean":
                df[col] = series.fillna(series.mean())

            elif strategy == "zero":
                df[col] = series.fillna(0)
        
        # categorical / string columns
        else:
            if strategy in ["median", "mean"]:
                # fallback to mode
                if not series.mode().empty:
                    df[col] = series.fillna(series.mode()[0])

            elif strategy == "zero":
                df[col] = series.fillna("0")

            elif strategy == "mode":
                if not series.mode().empty:
                    df[col] = series.fillna(series.mode()[0])

    return df


def apply_skewness(df, action):

    for col in action.get("target", []):

        if col in df.columns:
            df[f"scaled_{col}"] = (
                df[col] - df[col].mean()
            ) / df[col].std()

    return df


def apply_kurtosis(df, action):

    for col in action.get("target", []):

        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def adapt_col_dtypes(df, column_config):
    # setup logger
    logger = session.logger

    # 
    df_corr = df.copy()

    num_cols = column_config.get("numeric", [])
    cat_cols = column_config.get("categorical", [])

    if not num_cols and not cat_cols:
        logger.info("""
                    No column specified as 'numeric' or 'categorical'. 
                    Start converting all df.columns to numeric: %s
                    """, 
                    df_corr.columns)
        for col in df_corr.columns:
            try:
                df_corr[col] = pd.to_numeric(df_corr[col])
            except Exception as e:
                logger.info("Column '%s' not convertible to numeric dtype:\n%s",
                            col,
                            e)        
                pass

    # numeric
    for col in num_cols:
        if col not in df_corr.columns:
            continue
        
        logger.info("Start dtype_conversion to numeric_columns: %s", num_cols)
        
        df_corr[col] = pd.to_numeric(df_corr[col], errors="coerce")

    # categorical
    for col in cat_cols:
        if col not in df_corr.columns:
            continue
        
        logger.info("Start dtype_conversion to categorical columns: %s", 
                    cat_cols)
               
        df_corr[col] = df_corr[col].astype("string")

    return df_corr


ACTION_DISPATCH = {
    "drop_duplicates": apply_drop_duplicates,
    "parse_datetime": apply_parse_datetime,
    "impute_nan": apply_impute_nan,
    "handle_skewness": apply_skewness,
    "handle_kurtosis": apply_kurtosis
}


# ------------------------------
# WRAPPER FUNCTION
# ------------------------------
@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def clean_from_json(name):
    run_clean_from_json(name)

    return 


# ------------------------------
# MAIN FUNCTION
# ------------------------------
def run_clean_from_json(name):
    # (1) load config + parse arguments
    gh.load_env_vars()

    data_processed = os.getenv("PATH_PROCESSED")
    
    config = fh.get_yaml_config(name)

    general_config = config.get("general_args", {})
    log_name = general_config["name_log"]
    name_logfile = general_config["name_logfile"]
    selected_years = general_config["selected_years"]
    data_folder = Path(general_config["data_folder"])
    df_path = Path(f"{data_processed}/{data_folder}")

    report_dict = load_eda_summary(general_config)
    processing = report_dict["processing"]

    column_config = config.get("columns", {})

    # setup logger 
    logger = create_logger(name=log_name, file_name=name_logfile)
    session.logger = logger

    for file, actions in processing.items():

        f_name = Path(file).stem
        year = f_name.split("_")[1]
        
        if int(year) not in selected_years:
            logger.info("Skipping file from %s (start_year=%s)", 
                        year,
                        selected_years)
            continue

        path = f"{df_path}/{str(f_name).strip()}_harmonized.parquet"

        logger.info("Cleaning: %s",  
                    ph.shorten_path(path))

        df = pd.read_parquet(path)

        df_corr = adapt_col_dtypes(df, column_config)

        for action in actions:
            action_name = action.get("action")

            handler = ACTION_DISPATCH.get(action_name)

            if not handler:
                logger.warning("No runtime handler for %s", 
                               action_name)
                continue

            df_corr = handler(df_corr, action)

        # save
        out_folder = f"{df_path}"
        out_name = f"{str(f_name).strip()}_clean"

        dfh.save_df_to_parquet(df_corr, out_name, out_folder, chunked=True)

    logger.info("\nCleaning dfs complete.")

if __name__ == "__main__":
    clean_from_json()