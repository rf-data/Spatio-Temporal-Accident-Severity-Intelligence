## ml_prep_helper.py
# import
from datetime import datetime
import numpy as np
import pandas as pd

import src.utils.df_helper as dfh


def merge_df_ml_ready(general_args, col_dict, save_df=False):
    # (1) load config + parse arguments
    data_folder = general_args.get("data_folder", None)

    h3_col = col_dict.get("h3_col", "h3_res4")
    period_col = col_dict.get("period_col", "time_bin")
    target_col = col_dict.get("target_col", "n_accidents")


    # load df_merged
    dfs = dfh.load_processed_files(general_args)

    dfs_clean = []
    for df_name, df_list in dfs.items():
        if len(df_list) == 0:
            print(f"[WARNING] Skipping {df_name} (no data)")
            continue

        dfs_clean.append(df_list[0])


    df_all = pd.concat(dfs_clean, ignore_index=True)

    df_agg = (
        df_all
        .groupby([h3_col, period_col], as_index=False)
        .agg({target_col: "sum"})
        )
    
    if save_df:
        now = datetime.now().strftime("%Y-%m-%d_%H:%M")
        dfh.save_df_to_parquet(df_agg, f"{now}_df_merged", data_folder)

        return df_agg, now
    
    return df_agg


