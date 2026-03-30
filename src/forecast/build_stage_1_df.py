## build_ml_ready_df.py
# imports
import click
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# from src.utils.df_helper import save_df_to_parquet
from src.utils.file_helper import get_yaml_config
# from src.agentic_AI.report.raw_data_report import load_eda_summary
import src.feature_engineering.time_columns as time

import src.utils.general_helper as gh
# import src.utils.path_helper as ph
import src.utils.df_helper as dfh

from src.core.session import session


def add_cyclic_time_features(df, period_in, period_col):
    logger = session.logger

    period = time.check_translate_freq(period_in)

    if period in ["W", "2W"]:
        use = ["month", "weekday"]

    elif period == "ME":
        use = ["month"]

    else:
        use = ["month"]


    logger.info("Extracting cols from dt_col and adding cyclic time features of %s (%s)",
                use,
                period)
    
    df = time.extract_col_from_datetime(df, period_col, use)
    df = time.cyclic_encode_col(df, use)

    return df


def compute_zero_streak(x):
    streak = np.zeros(len(x), dtype=int)
    
    for i in range(1, len(x)):
        if x.iloc[i] == 0:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 0
    return streak
    
    # for val in x:
    #     if val == 0:
    #         current += 1
    #     else:
    #         current = 0
    #     streak.append(current)
    # return streak


def add_time_features(df, config):

    target_col = config["target_col"]   # , "has_accident")
    h3_col = config["h3_col"]  # , "h3_res4")
    period_col = config["period_col"]   # , "time_bin")

    time_dict = config["time_features"]     # , {})
    lag_values = time_dict["lag"]        # , [])
    rolling_values = time_dict["rolling_values"]        # , [])
    rolling_stats = time_dict["rolling_stats"]       # , [])

    df = (
        df.groupby([h3_col, period_col], 
                   as_index=False)
        .agg({target_col: "max"})
        )
    df = df.sort_values([h3_col, period_col])
    
    print("[DEBUG] 'sorted' df shape:", df.shape)
    print("[DEBUG] 'sorted' df head:\n", df.head(3))
    dups = df.duplicated([h3_col, period_col])
    if dups.any():
        dups_df = df[df.duplicated([h3_col, period_col], keep=False)]
        print(dups_df.head(20))

        print("[DEBUG] 'sorted' df duplicates total count:", dups.sum())

        n_dups = dups_df.groupby([h3_col, period_col])[target_col].nunique()
        print("[DEBUG] 'sorted' df duplicates n_unique duplicates:\n", n_dups)
        
    assert df[period_col].is_monotonic_increasing is False
    assert not df.duplicated([h3_col, period_col]).any()

    g = df.groupby(h3_col)

    print("[DEBUG] lag_values:\t", lag_values)
    for lag in lag_values: 
        lag_int = int(lag)
        df[f"lag_{lag_int}"] = g[target_col].shift(lag_int)
    
    # for stat in rolling_stats:
    print("[DEBUG] rolling_values:\t", rolling_values)
    for roll in rolling_values:
        roll_int = int(roll)

        # shifted = g[target_col].shift(1)
        if "mean" in rolling_stats:
            df[f"roll_mean_{roll_int}"] = (
                                    g[target_col]
                                    .shift(1)
                                    .rolling(roll_int)
                                    .mean()
                                )
        if "sum" in rolling_stats:
            df[f"roll_sum_{roll_int}"] = (
                                    g[target_col]
                                    .shift(1)
                                    .rolling(roll_int)
                                    .sum()
                                    .reset_index(level=0, drop=True)
                                    )
        if "std" in rolling_stats:
            df[f"roll_std_{roll_int}"] = (
                                    g[target_col]
                                    .shift(1)
                                    .groupby(df[h3_col])
                                    .transform(lambda x: 
                                               x.rolling(roll_int)
                                               .std())
                                    )

    df["zero_streak"] = (
                    g["has_accident"]
                    .transform(compute_zero_streak)
                    )
    assert not ((df["has_accident"] == 1) & (df["zero_streak"] != 0)).any() 

    lag_roll_cols = [c for c in df.columns if "lag_" in c or "roll_" in c]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)

    if "lag_1" in df.columns:
        df["had_accident_last_k"] = (df["lag_1"] > 0).astype(int)
    # print("[DEBUG] df head:", df.head(20))
    
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
    new_file_suffix = charac_config["suffix_df"]   # , "new")
    # target_col = charac_config.get("target_col", "n_accidents")
    # dt_col = charac_config.get("dt_col", "datetime")
    # cyclic_encoding = charac_config.get("cyclic_encoding", False)
    # period = charac_config.get("period", "weekly")
    # min_events = config.get("min_events", 2)
    # period_col = config.get("period_col", "time_bin")
    # h3_col = config.get("h3_col", "h3_res4")

    general_dict = config.get("general_args", {})
    data_folder = Path(general_dict["save_folder"]) # , "")
    df_folder = Path(f"{data_processed}/{data_folder}")

    # time_feat_dict = charac_config.get("time_features", {})

    # cols_new = {
    #     "time_col_new": dt_col,
    #     "add_cols":
    # }
    # 1. load dfs 
    # charac_dfs = dfh.load_merge_processed_files(charac_config)  # dict(year: df_geo_time)

    # now = datetime.now().strftime("%Y-%m-%d")
    # # base_dfs = []
    # dfs = []
    # for df_name, df in charac_dfs.items():
    #     print("Start processing df:\t", df_name)

    #     df_processed = process_single_year(df, charac_config)
    #     dfs.append(df_processed)

    #     save_name = f"{now}_{df_name}_{new_file_suffix}"
    #     dfh.save_df_to_parquet(
    #                     df=df_processed, 
    #                     f_name=save_name, 
    #                     folder=df_folder,
    #                     chunked=True
    #                        )

    now = "2026-03-24"
    suf_dict = {
            new_file_suffix: [],
            }
    
    dfs_dict = dfh.load_processed_files(
                            charac_config,
                            data_folder=data_folder, 
                            suffixes=suf_dict,
                            prefix=f"{now}_"
                            )

    dfs = [df[0] for df in dfs_dict.values() if len(df) > 0]

    df_all = pd.concat(dfs)   # load_all_processed()

    df_all = add_time_features(df_all, charac_config)
   
    dfh.save_df_to_parquet(
                        df=df_all, 
                        f_name=f"{now}_{new_file_suffix}_complete", 
                        folder=df_folder,
                        chunked=True
                           )
    
    return  
    
    ################

def process_single_year(df_in: pd.DataFrame, config: dict) -> pd.DataFrame:

    df = df_in.copy()

    h3_col = config["h3_col"]   # , "h3_res4")
    dt_col = config["dt_col"]   # , "datetime")
    period = config["period"]    # , "weekly")
    period_col = config["period_col"]   # , "time_bin")
    target_prelim = config["target_prelim"] # , "n_accidents")
    target_col = config["target_col"]
    # min_events = config["min_events"]   # , 2)
    cyclic_encoding = config["cyclic_encoding"]
    
    # --- datetime ---
    df = df.dropna(subset=[dt_col])
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.dropna(subset=[dt_col])

    # --- time binning --- 
    per_safe = time.check_translate_freq(period)
    df[period_col] = (
                df[dt_col]
                .dt.to_period(per_safe)
                .dt.start_time
                )

    # 3. extract 'base' df
    # df_base = df[[h3_col, period_col]].copy()
    # base_dfs.append(df_base)

    df_infl = dfh.inflate_df(df, config)

    # # --- active cells ---
    # cell_activity = df.groupby(h3_col)[target_prelim].sum()
    # active_cells = cell_activity[
    #                         cell_activity >= min_events
    #                         ].index
    
    # # --- complete grid ---
    # # all_nodes = list(h3_to_node.keys())
    # all_times = df[period_col].sort_values().unique()

    # df_grid = dfh.create_complete_grid(
    #                                 df, 
    #                                 idx_name=h3_col, 
    #                                 idx_values=active_cells, 
    #                                 col_name=period_col,
    #                                 col_values=all_times
    #                                 )

    assert df_infl[h3_col].nunique() > 0, "No cells left"
    assert df_infl[period_col].nunique() > 0, "No time bins"
    # print("[DEBUG] df_full head:", df_grid.head(3))

    # --- aggregation ---
    # inflate_df already returns one row per (h3, time_bin) with `target_prelim`.
    # Keep these counts directly; regrouping with `.size()` would collapse counts to 0/1.
    df_agg = df_infl[[h3_col, period_col, target_prelim]].copy()
    
    df_agg[target_col] = (df_agg[target_prelim] > 0).astype("int")

    if cyclic_encoding: 
        df_agg = add_cyclic_time_features(
            df_agg,
            period_in=period,
            period_col=period_col
            )
        
    return df_agg    


if __name__ == "__main__":
    build_charac_df()



        # --- 6. Cyclic Features ---
        # if cyclic_encoding:
        # df_time = add_cyclic_time_features(df_agg, period, period_col)

        # add lag and rolling features
        # df_time = add_lag_roll_features(df_agg, charac_config)
        
        # f_name = f"{df_name}_{new_file_suffix}_v2"
        # save_df_to_parquet(df_time, f_name, df_folder, chunked=True)


    # df_agg["log_target"] = np.log1p(df_agg[target_prelim])
    # df_agg["is_weekend"] = (df_agg["weekday"] >= 5).int()

    
    # # weather_1_count        # for top2 categories + "other"
    # weather_1_ratio
    # # weather_2_count        # for top2 categories + "other"
    # weather_2_ratio
    # # weather_other_count        # for top2 categories + "other"
    # weather_other_ratio
    # from scipy.stats import entropy
    # # niedrig → homogen (immer gleich)
    # # hoch → chaotisch
    # weather_entropy
    # has_bad_weather -> ratio
    # n_unique_weather
    # max_ratio_weather       # Dominance Feature, anstelle 'mmodus'


    # collision_?_count
    # collision_?_ratio
    
    # interception_?_count
    # interception_?_ratio

    # light_?_count
    # light_?_ratio
    
    
    # collision_entropy
    # light_entropy
    # interception_entropy

    # has_bad_weather
    # has_night

    # n_unique_weather
    # n_unique_collision

    # max_ratio_weather       # anstelle 'modus'