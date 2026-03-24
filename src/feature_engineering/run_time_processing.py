## run_ts_processing.py
# import
import click
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

from src.utils.file_helper import load_files_from_folder, get_yaml_config, save_df_to_parquet
from src.agentic_AI.feature_engineering.time_columns import (add_time_cols, 
                                                             create_timestamp_col, 
                                                             cyclic_encode_col)
from src.agentic_AI.report.raw_data_report import load_eda_summary

import src.utils.general_helper as gh

@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def time_preprocessing(name):

    run_time_processing(name)

    return

def run_time_processing(name):
    # (1) load config + parse arguments
    gh.load_env_vars()

    # data_raw = os.getenv("PATH_RAW")
    data_processed = os.getenv("PATH_PROCESSED")

    config = get_yaml_config(name)
    data_folder = Path(config.get("general_args", {}).get("data_folder", {}))
    # h3_folder = Path(config.get("general_args", {}).get("h3_folder", {}))
    folder = Path(f"{data_processed}/{data_folder}")

    time_dict = config.get("time_processing", {})
    # dt_format = time_dict.get("dt_format")
    # time_col = time_dict.get("time_col")
    cols_needed = time_dict.get("necessary_cols", [])
    time_col_new = time_dict.get("time_col_new", "timestamp")
    # timestamp_col = time_dict.get("timestamp_col", None)
    df_ts_cols = time_dict.get("df_ts_cols")
    cyclic_encode = df_ts_cols.get("cyclic_encode", [])
    # extract_cols = df_ts_cols.get("to_extract", [])
    new_cols = {
            "add_cols": df_ts_cols.get("to_add", None), 
            "time_col_new": time_col_new,
            # "cyclic_encode": df_ts_cols.get("cyclic_encode", [])
            }
    
    # load report and df_dict 
    report = load_eda_summary(config)
    df_names = report.get("files", [])

    df_dict = load_files_from_folder(
                                folder,
                                df_names,
                                "harmonized", 
                                f_type="parquet"
                                )

    # (2B) Feature Engineering 'time'
    timestamp_dict = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hour": "hour",
        "minute": "minute",
        "time_processing": time_dict
    }
    
    for name, df in df_dict.items():
        print(f"Creating timestamp_col in df '{name}'.")   

        # reduce df
        df.columns = [col for col in df.columns if col in cols_needed]  
        df[time_col_new] = create_timestamp_col(df, timestamp_dict)
    
        print("[DEBUG - CREATING TIMESTAMP] df head 'datetime':\n", df[time_col_new].head(3))

        if new_cols:
            print(f"Adding further times_cols in df '{name}'.")  
            df = add_time_cols(df, new_cols)

            print("[DEBUG - ADDING COLS] df head:\n", df.head(3))

        if cyclic_encode:
            df = cyclic_encode_col(df, cyclic_encode)

            print("[DEBUG - CYCLIC_ENCODE] df head:\n", df.head(3))

        f_name = f"{name.split(".")[0]}_time"
        save_df_to_parquet(df, f_name, folder, chunked=True)
    
    ### HIER 
        
if __name__ == "__main__":
    time_preprocessing()

      
        # if dt_format and extract_cols:
        #     print(f"Start parsing dt_col '{time_col}' in df '{name}'.")
        #     dt_col = "time_parsed"

        #     df[dt_col] = pd.to_datetime(
        #                             df[time_col],
        #                             format=dt_format,
        #                             errors="coerce"
        #                                 )
        
        #     for col in :
                
        #         df[col] = col_add_dict[col]