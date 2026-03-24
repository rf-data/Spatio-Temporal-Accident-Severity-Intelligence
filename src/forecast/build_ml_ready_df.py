## build_ml_ready_df.py
# imports
import click
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.df_helper import save_df_to_parquet
from src.utils.file_helper import get_yaml_config
from src.agentic_AI.report.raw_data_report import load_eda_summary
import src.feature_engineering.time_columns as time

import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.df_helper as dfh


def add_cyclic_time_features(df, period_in, period_col):

    period = time.check_translate_freq(period_in)

    if period in ["W", "2W"]:
        use = ["month", "weekday"]

    elif period == "ME":
        use = ["month"]

    else:
        use = ["month"]

    df = time.extract_col_from_datetime(df, period_col, use)
    df = time.cyclic_encode_col(df, use)

    return df



def add_lag_roll_features(df, config):

    target_col = config.get("target_col", "n_accidents")
    h3_col = config.get("h3_col", "h3_res4")
    period_col = config.get("period_col", "time_bin")

    time_dict = config.get("time_features", {})
    lag_values = time_dict.get("lag", [])
    rolling_values = time_dict.get("rolling_values", [])
    rolling_stats = time_dict.get("rolling_stats", [])

    df = df.sort_values([h3_col, period_col])
    g = df.groupby(h3_col)

    print("[DEBUG] lag_values:\t", lag_values)
    for lag in lag_values: 
        lag_int = int(lag)
        df[f"lag_{lag_int}"] = g[target_col].shift(lag_int)
    
    # for stat in rolling_stats:
    print("[DEBUG] rolling_values:\t", rolling_values)
    for roll in rolling_values:
        roll_int = int(roll)

        shifted = g[target_col].shift(1)
        if "mean" in rolling_stats:
            df[f"roll_mean_{roll_int}"] = (
                    shifted.groupby(df[h3_col])
                            .rolling(roll_int)
                            .mean()
                            .reset_index(level=0, drop=True)
                            )
        if "sum" in rolling_stats:
            df[f"roll_sum_{roll_int}"] = (
                    shifted.groupby(df[h3_col])
                            .rolling(roll_int)
                            .sum()
                            .reset_index(level=0, drop=True)
                            )
    
    df = df.fillna(0)
    print("[DEBUG] df head:", df.head(20))
    
    return df


@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def build_charac_df(name):  
    
    run_build_charac_df(name)
    return

def run_build_charac_df(name):   
    """
    Builds minimal aggregated dataset
    """
    # (1) load config + parse arguments
    gh.load_env_vars()
    data_processed = os.getenv("PATH_PROCESSED")

    config = get_yaml_config(name)

    charac_config = config.get("charac", {})
    h3_col = charac_config.get("h3_col", "h3_res6")
    period_col = charac_config.get("period_col", "time_bin")
    target_col = charac_config.get("target_col", "n_accidents")
    dt_col = charac_config.get("dt_col", "datetime")
    # cyclic_encoding = charac_config.get("cyclic_encoding", False)
    period = charac_config.get("period", "weekly")

    general_dict = config.get("general_args", {})
    data_folder = Path(general_dict.get("save_folder", ""))
    df_folder = Path(f"{data_processed}/{data_folder}")
    new_file_suffix = general_dict.get("new_file_suffix", "new")

    time_feat_dict = charac_config.get("time_features", {})

    # cols_new = {
    #     "time_col_new": dt_col,
    #     "add_cols":
    # }
    # 1. load dfs 
    charac_dfs = dfh.load_merge_processed_files(charac_config)

    for df_name, df in charac_dfs.items():
        # 2. add 'timebin' column
        # df_time = time.add_time_cols(df, cols_new):

        df = df.dropna(subset=[dt_col])
        df[dt_col] = pd.to_datetime(df[dt_col])

        per_safe = time.check_translate_freq(period)
        df[period_col] = df[dt_col].dt.to_period(per_safe).dt.start_time

        # 2. aggregate dfs
        df_agg = (
        df.groupby([h3_col, period_col])
        .size()
        .reset_index(name=target_col)
        )
        
        # --- 4. Cyclic Features ---
        # if cyclic_encoding:
        df_time = add_cyclic_time_features(df_agg, period, period_col)

        # add lag and rolling features
        df_time = add_lag_roll_features(df_agg, charac_config)
        
        f_name = f"{df_name}_{new_file_suffix}_v2"
        save_df_to_parquet(df_time, f_name, df_folder, chunked=True)

    return 


if __name__ == "__main__":
    build_charac_df()

   