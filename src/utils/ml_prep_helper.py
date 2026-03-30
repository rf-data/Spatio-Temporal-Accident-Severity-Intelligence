## ml_prep_helper.py
# import
from datetime import datetime
import numpy as np
import pandas as pd

import src.utils.df_helper as dfh


# def compute_smoothed_target_encoding(train_df, group_col, target_col, alpha=10):

#     global_mean = train_df[target_col].mean()

#     stats = (
#         train_df
#         .groupby(group_col)[target_col]
#         .agg(["mean", "count"])
#     )

#     smooth = (
#         (stats["count"] * stats["mean"] + alpha * global_mean)
#         /
#         (stats["count"] + alpha)
#     )

#     return smooth, global_mean


def target_encode_col(train_df):
    # setup logger
    logger = session.logger

    #
    group_col = session.exp_params.get("encode_space", None)
    target_col = session.exp_params.get("target_final", None)

    mean_val = train_df.sort_values(group_col).groupby(group_col)[target_col].mean()

    glob_mean = train_df[target_col].mean()

    df_enc = train_df.drop(columns=[group_col]).copy()
    df_enc[f"{group_col}_te"] = (
        train_df[group_col].map(mean_val)
        # .fillna(glob_mean)
    )

    logger.info(
        "NaN count in '%s':\t%s",
        f"{group_col}_te",
        df_enc[f"{group_col}_te"].isna().sum(),
    )

    # encode_dict = {mean_val[i, 0]:mean_val[i, 1] for i in len(mean_val)}

    # df_encoded = df.replace(encode_dict)

    return df_enc


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


