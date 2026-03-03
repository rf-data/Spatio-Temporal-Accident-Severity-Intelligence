## prepare_data_logreg.py
# imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import gc

import src.utils.general_helper as gh
import src.utils.file_helper as fh
import src.utils.path_helper as ph
import src.utils.FeatEng_helper as feat

from src.core.session import session
from src.core.logger import create_logger
from configuration.Phase_1_month_base_LogReg_cyclic import config

# ------------------
# HELPER FUNCTION
# ------------------


def load_h3_df(h3_idx):
    # setup logger
    logger = session.logger

    # load data
    df_name = session.prep_params.get("df_name", None)
    folder = os.getenv("PATH_PROCESSED")
    h3_df_path = Path(folder) / f"{df_name}.csv"

    df = pd.read_csv(h3_df_path)
    logger.info("Loaded data from '...%s'", ph.shorten_path(h3_df_path))

    df = fh.enforce_datetime(df)

    cols_needed = session.prep_params.get("cols_needed", None)
    cols_needed.append(h3_idx)

    return df[cols_needed].copy()


def add_lag_features(df):
    # setup logger
    logger = session.logger

    df_new = df.copy()

    feats = session.exp_params.get("features", None)
    h3_idx = session.prep_params.get("h3_idx", None)

    # lag features (monthly base)
    for lag in [1, 2, 3, 4, 12]:  #
        feat = f"lag_{lag}"
        if feat in feats:
            logger.info("Adding lag feature to df: '%s'", feat)
            df_new[f"lag_{lag}"] = df_new["n_accidents"].shift(lag)

    return df_new


def add_rolling_features(df):
    # setup logger
    logger = session.logger

    df_new = df.copy()
    feats = session.exp_params.get("features", None)
    # h3_idx = session.h3_idx

    # rolling features
    for roll in ["mean", "sum"]:
        for window in [3, 4, 12]:
            feat = f"roll_{roll}_{window}"
            if feat in feats:
                logger.info("Adding rolling feature to df: '%s'", feat)

                df_new[f"roll_{roll}_{window}"] = (
                    df_new["n_accidents"]
                    .rolling(window)
                    .agg(roll)
                    .reset_index(level=0, drop=True)
                )

            gc.collect()

    return df_new


def add_datetime_feats(df, period):
    # setup logger
    logger = session.logger

    df_new = df.copy()
    feats = session.exp_params.get("features", None)
    if not pd.api.types.is_datetime64_any_dtype(df_new["datetime"]):
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="raise")
        logger.info(
            "Changed 'datetime' dtype from %s to 'datetime64'.",
            df_new["datetime"].dtype,
        )

    df_new["period"] = df_new["datetime"].dt.to_period(period)

    if "year" in feats:
        df_new["year"] = df_new["datetime"].dt.year

    if "month" in feats:
        df_new["month"] = df_new["datetime"].dt.month

    # if "week" in feats:
    #     df_new["week"] = df_new["datetime"].dt.isocalendar().week # .isocalendar().week

    if "weekday" in feats:
        df_new["weekday"] = df_new["datetime"].dt.weekday

    logger.info("Added datetime features to df.")

    return df_new


def add_season_dummies(df):
    # setup logger
    logger = session.logger

    df_new = df.copy()
    # feats = session.exp_params.get("features", None)
    dummy_cols = session.exp_params.get("encode_time", None)
    # h3_idx = session.h3_idx

    # seasonality features
    for col in dummy_cols:
        df_new = pd.get_dummies(df_new, columns=[col], drop_first=True)

    logger.info("Added season dummy_columns from %s to df.", dummy_cols)

    return df_new


def add_zero_features(df):
    # setup logger
    logger = session.logger

    # freq = session.prep_params.get("frequency", None)
    feats = session.exp_params.get("features", None)

    df_new = df.copy()
    # h3_idx = session.h3_idx

    # Zero Persistence
    if "zero_streak" in feats:
        df_new["was_zero"] = (df_new["n_accidents"] == 0).astype(int)
        df_new["zero_streak"] = df_new["was_zero"].transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )

    logger.info("Added 'was_zero' and 'zero_streak' cols to df.")

    return df_new


def add_temporal_features(df):

    cyclic_encode = session.exp_params.get("cyclic_encode", None)

    # add time + seasonality features
    df_1 = add_lag_features(df)
    df_2 = add_rolling_features(df_1)

    if cyclic_encode:
        df_3 = feat.cyclic_encode_col(df_2)
    else:
        df_3 = add_season_dummies(df_2)

    # for func in [

    #             ]:
    #     df = func(df)

    return df_3


#  "id",
#                         "lat_norm",
#                         "lon_norm",
#                         "datetime",
#                         "light conditions",
#                         "intersection type",
#                         "weather",
#                         "collision type"
#                         ],


def aggregate_df(df):
    # setup logger
    logger = session.logger

    freq = session.prep_params.get("frequency", None)
    h3_idx = session.prep_params.get("h3_idx", None)
    feats = session.exp_params.get("features", None)
    period_dict = session.exp_params.get("period_dict", None)
    # split_cols = session.exp_params.get("split_col_names")

    df_dt = add_datetime_feats(df, period_dict[freq])

    group_cols = [h3_idx, "period"]

    dt_cols = [feat for feat in ["year", "month", "weekday", "week"] if feat in feats]

    mean_cols = [
        *[c for c in df.columns if c.startswith("weath_")],
        *[c for c in df.columns if c.startswith("inter_")],
        *[c for c in df.columns if c.startswith("coll_")],
        *[c for c in df.columns if c.startswith("light_")],
    ]

    agg_dict = {
        "id": "count",
        **{col: "min" for col in dt_cols},
        **{col: "mean" for col in mean_cols},
    }

    df_group = (
        df_dt.groupby(group_cols)  # [mean_cols]
        .agg(agg_dict)
        .rename(columns={"id": "n_accidents"})
        .reset_index()
        .sort_values(group_cols)
        .copy()
    )

    logger.info(
        "Aggregated df and created descriptive metrics.\nDF HEAD:\n%s",
        df_group.head(3).T,
    )

    # check_df_agg()

    return df_group


# def has_gap(x):
#     periods = pd.PeriodIndex(
#         x["year"].astype(str) + "-" + x[freq].astype(str),
#         freq="W"
#     )
#     expected = pd.period_range(periods.min(), periods.max(), freq="W")
#     return len(periods) != len(expected)


# def check_gaps(x):
#     periods = x["year"].astype(str) + "-" + x[freq].astype(str)
#     idx = pd.PeriodIndex(periods, freq="W")
#     return idx.is_monotonic_increasing and len(idx) == len(idx.unique())


# def check_df_agg(df):
#     freq = session.prep_params.get("frequency", None)
#     h3_idx = session.prep_params.get("h3_idx", None)

#     duplicates = df.duplicated(subset=[h3_idx, "year", freq]).any()
# if duplicates:
#   raise ???()
#
#  df = df.sort_values([h3_idx, "year", freq])

# check_monotonic = df.groupby(h3_idx).apply(
#     lambda x: x.index.is_monotonic_increasing
# )

#     check_df = df.groupby(h3_idx)[["year", freq]].apply(
#                                     lambda x:
#                                     x.sort_values(["year", freq])
#                                     .equals(x)
#                                     )

#     return check_df.all()


def add_target(df):
    # setup logger
    logger = session.logger

    # "Option A" - konservative: target = 1 if n_accidents >= 2 else 0,
    # "Option B" - risk-based: target = 1 if n_accidents >= q75,
    # "Option C" - hotspot: target = 1 if n_accidents >= q90

    target_final = session.exp_params.get("target_final", None)
    q95 = session.exp_params.get("q95", None)  # here: 3
    q75 = session.exp_params.get("q75", None)  # here: 1
    classification_style = session.exp_params.get("classification", None)

    # if classification_style == "binary":
    #     else_cond = 1
    # elif classification_style == "multi_class":
    #     else_cond = np.where(df["n_accidents"].shift(-1) < q95,
    #                             1,
    #                             2)
    # else:
    #     logger.error("Entered invalid value for 'classification_style' (from session):\n--> %s",
    #                 classification_style)
    #     raise ValueError("Entered invalid value for 'classification_style' (from session):\n--> %s",
    #                     classification_style)

    df["risk_next"] = np.where(df["n_accidents"].shift(-1) >= 1, 1, 0)

    # delete last row since no value
    df_new = df.iloc[:-1].copy()

    logger.info("Added final target column '%s' to df.", target_final)

    return df_new


def inflate_df(df):
    # setup logger
    logger = session.logger

    #
    h3_idx = session.prep_params.get("h3_idx", None)
    time_col = session.exp_params.get("time_col", None)
    target_init = session.exp_params.get("target_init", None)

    # create full_index
    all_periods = get_period_range(df, time_col)
    all_h3 = df[h3_idx].unique()

    full_index = pd.MultiIndex.from_product(
        [all_h3, all_periods], names=[h3_idx, time_col]
    )
    logger.info("Created full index (col = %s).", [h3_idx, time_col])

    # create full_df
    df_full = df.set_index([h3_idx, "period"]).reindex(full_index).reset_index().copy()

    # zero_fill only target_col
    df_full[target_init] = df_full[target_init].fillna(0)
    df_full["year"] = df_full["period"].dt.year
    df_full["month"] = df_full["period"].dt.month

    logger.info(
        "Zero_inflated df and filled col '%s' (with 0), 'year' and 'month'", target_init
    )

    return df_full


def get_period_range(df, period_col):
    # setup logger
    logger = session.logger

    # retrieve settings from 'session'
    freq = session.prep_params.get("frequency", None)
    period_dict = session.exp_params.get("period_dict", None)
    period = period_dict[freq]

    # determine lowest + highest plausible period
    df_period = df[period_col].copy()

    lower_bound = pd.Period("2005-01", freq="M")
    upper_bound = pd.Period("2024-12", freq="M")

    min_period = max(df_period.min(), lower_bound)
    max_period = min(df_period.max(), upper_bound)

    # set
    full_periods = pd.period_range(min_period, max_period, freq=period)

    logger.info(
        "Created period index [min=%s, max=%s, freq=%s]", min_period, max_period, period
    )

    return full_periods


def cutoff_invalid_values(df):
    # setup logger
    logger = session.logger

    #
    h3_idx = session.prep_params.get("h3_idx", None)
    max_lag = session.exp_params.get("max_lag", None)
    group_cols = [h3_idx, "period"]

    # sort df + cutoff first rows per h3_idx
    logger.info(
        "Start sorting df and cutting off first %s rows for each 'h3_idx'", max_lag
    )

    df_sort = df.sort_values(group_cols)

    df_group = df_sort.assign(pos=lambda x: x.groupby(h3_idx).cumcount())

    df_group = (
        df_group[df_group["pos"] >= max_lag].drop(columns="pos").reset_index(drop=True)
    )

    logger.info("Shape final df: %s", df_group.shape)

    return df_group


# ------------------
# MAIN FUNCTION
# ------------------


def create_aggregated_dataset():
    # load env variables
    gh.load_env_vars()

    # load configurations
    session.load_config(config)
    log_name = session.gen_params.get("log_name", None)  # "ETL_CHARACTERISTICS"
    name_logfile = session.gen_params.get("log_file")  # "etl_characteristics"
    h3_idx = session.prep_params.get("h3_idx", None)

    # extract configurations from session
    # df_path = session.exp_params.get("df_path", None)
    # split = session.exp_params.get("split_method", None)
    # feats = session.exp_params.get("features", None)

    # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    logger.info("Start preparing df for LogReg.")

    # (0) load non-aggregated data
    df = load_h3_df(h3_idx)

    # (1) add 'splitted_cat_feats'
    # df_cat = add_splitted_cat_feats(df)

    # (2) aggregate + inflate data df
    df_group = aggregate_df(df)
    df_inf = inflate_df(df_group)

    # (3) add seasonality + time and 'zero_features'
    df_temp = add_temporal_features(df_inf)
    df_zero = add_zero_features(df_temp)

    # (4) add 'target'
    df_target = add_target(df_zero)

    # (5) remove first and last rows per h3_index
    df_final = cutoff_invalid_values(df_target)

    # print("Describe 'year':\n", df_final["year"].describe())
    # save df
    f_name = session.prep_params.get("df_prep_name", None)  # "lr_prep_p1_ZeroInf"
    fh.save_df_to_parquet(df_final, f_name, chunked=True)

    return


if __name__ == "__main__":
    create_aggregated_dataset()
