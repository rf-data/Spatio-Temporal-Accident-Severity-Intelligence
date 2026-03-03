## prepare_data_logreg.py
# imports
import os
from pathlib import Path
import numpy as np
import pandas as pd

import src.utils.general_helper as gh
import src.utils.file_helper as fh
import src.utils.path_helper as ph
import src.utils.FeatEng_helper as feat

from src.core.session import session
from src.core.logger import create_logger
from configuration.Phase_2A_LogReg_base import config

# ------------------
# HELPER FUNCTION
# ------------------


def load_base_df():
    # setup logger
    logger = session.logger

    # load data
    df_name = session.prep_params.get("df_name", None)
    folder = os.getenv("PATH_PROCESSED")
    df_path = Path(folder) / f"{df_name}.csv"

    df = pd.read_csv(df_path)
    logger.info("Loaded data from '...%s'", ph.shorten_path(df_path))

    df = fh.enforce_datetime(df)

    cols_needed = session.prep_params.get("cols_needed", None)
    # cols_needed.append(h3_idx)

    return df[cols_needed].copy()


def split_intersection_col(df):
    # setup logger
    logger = session.logger

    """
    1 - Out of intersection
    2 - Intersection in X
    3 - Intersection in T
    4 - Intersection in Y
    5 - Intersection with more than 4 branches
    6 - Giratory
    7 - Place
    8 - Level crossing
    9 - Other intersection
    """
    # df = df_in.copy()

    df["inter_no"] = np.where(df["intersection type"] == 1, 1, 0)
    df["inter_X"] = np.where(df["intersection type"] == 2, 1, 0)
    df["inter_T"] = np.where(df["intersection type"] == 3, 1, 0)
    df["inter_Y"] = np.where(df["intersection type"] == 4, 1, 0)
    df["inter_4+"] = np.where(df["intersection type"] == 5, 1, 0)
    df["inter_gira"] = np.where(df["intersection type"] == 6, 1, 0)
    df["inter_place"] = np.where(df["intersection type"] == 7, 1, 0)
    df["inter_level"] = np.where(df["intersection type"] == 8, 1, 0)
    df["inter_other"] = np.where(df["intersection type"] == 9, 1, 0)

    df_new = df.drop(columns=["intersection type"]).copy()

    logger.info("Splitted 'intersection' column into distinct feature columns.")

    return df_new


def split_weather_col(df):
    # setup logger
    logger = session.logger

    """
    1 - Normal
    2 - Light rain
    3 - Heavy rain
    4 - Snow - hail
    5 - Fog - smoke
    6 - Strong wind - storm
    7 - Dazzling weather
    8 - Cloudy weather
    9 - Other
    """

    df["weath_normal"] = np.where(df["weather"] == 1, 1, 0)
    df["weath_l_rainy"] = np.where(df["weather"] == 2, 1, 0)
    df["weath_h_rainy"] = np.where(df["weather"] == 3, 1, 0)
    df["weath_snow_hail"] = np.where(df["weather"] == 4, 1, 0)
    df["weath_foggy"] = np.where(df["weather"] == 5, 1, 0)
    df["weath_stormy"] = np.where(df["weather"] == 6, 1, 0)
    df["weath_dazzle"] = np.where(df["weather"] == 7, 1, 0)
    df["weath_cloudy"] = np.where(df["weather"] == 8, 1, 0)
    df["weath_other"] = np.where(df["weather"] == 9, 1, 0)

    df_new = df.drop(columns=["weather"]).copy()
    logger.info("Splitted 'weather' column into distinct feature columns.")

    return df_new


def split_light_col(df):
    # setup logger
    logger = session.logger

    """
    1 - Full day
    2 - Twilight or dawn
    3 - Night without public lighting
    4 - Night with public lighting not lit
    5 - Night with public lighting on
    """
    # df = df_in.copy()

    df["light_full_day"] = np.where(df["light conditions"] == 1, 1, 0)
    df["light_dawn"] = np.where(df["light conditions"] == 2, 1, 0)
    df["light_no_night_light"] = np.where(df["light conditions"] == 3, 1, 0)
    df["light_night_unlit"] = np.where(df["light conditions"] == 4, 1, 0)
    df["light_night_lit"] = np.where(df["light conditions"] == 5, 1, 0)

    df_new = df.drop(columns=["light conditions"]).copy()
    logger.info("Splitted 'light conditions' column into distinct feature columns.")

    return df_new


def split_collision_col(df):
    # setup logger
    logger = session.logger

    """ 
    1- Two vehicles - frontal
    2 - Two vehicles - from the rear
    3 - Two vehicles - by the side
    4 - Three vehicles and more - in chain
    5 - Three or more vehicles - multiple collisions
    6 - Other collision
    7 - Without collision
    """

    df["coll_2_front"] = np.where(df["collision type"] == 1, 1, 0)
    df["coll_2_rear"] = np.where(df["collision type"] == 2, 1, 0)
    df["coll_2_side"] = np.where(df["collision type"] == 3, 1, 0)
    df["coll_3+_chain"] = np.where(df["collision type"] == 4, 1, 0)
    df["coll_3+_multi"] = np.where(df["collision type"] == 5, 1, 0)
    df["coll_other"] = np.where(df["collision type"] == 6, 1, 0)
    df["coll_no"] = np.where(df["collision type"] == 7, 1, 0)

    df_new = df.drop(columns=["collision type"]).copy()
    logger.info("Splitted 'collision type' column into distinct feature columns.")

    return df_new


def add_splitted_cat_feats(df):
    df_new = df.copy()
    target_col = session.exp_params.get("target_col", None)

    #
    split_cols = [split_light_col, split_weather_col, split_intersection_col]

    if target_col != "collision type":
        split_cols.append(split_collision_col)

    for func in split_cols:
        df_new = func(df_new)

    return df_new


def add_target(df):
    # setup logger
    logger = session.logger

    # "Option A" - konservative: target = 1 if n_accidents >= 2 else 0,
    # "Option B" - risk-based: target = 1 if n_accidents >= q75,
    # "Option C" - hotspot: target = 1 if n_accidents >= q90

    q95 = session.exp_params.get("q95", None)  # here: 3
    q75 = session.exp_params.get("q75", None)  # here: 1
    classification_style = session.exp_params.get("classification", None)

    if classification_style == "binary":
        else_cond = 1
    elif classification_style == "multi_class":
        else_cond = np.where(df["n_accidents"].shift(-1) < q95, 1, 2)
    else:
        logger.error(
            "Entered invalid value for 'classification_style' (from session):\n--> %s",
            classification_style,
        )
        raise ValueError(
            "Entered invalid value for 'classification_style' (from session):\n--> %s",
            classification_style,
        )

    df["risk_next"] = np.where(df["n_accidents"].shift(-1) < q75, 0, else_cond)

    # delete last row since no value
    df_new = df.iloc[:-1].copy()

    logger.info("Added 'target' column to df.")

    return df_new


def add_time_cols(df):
    # setup logger
    logger = session.logger

    #
    feats = session.exp_params.get("features", None)

    if "year" in feats:
        df["year"] = df["datetime"].dt.year
        logger.info("Added 'year' col to df.")

    if "month" in feats:
        df["month"] = df["datetime"].dt.month
        logger.info("Added 'month' col to df.")

    if "week" in feats:
        df["week"] = df["datetime"].dt.isocalendar().week
        logger.info("Added 'week' col to df.")  # .isocalendar().week

    if "weekday" in feats:
        df["weekday"] = df["datetime"].dt.weekday
        logger.info("Added 'weekday' col to df.")

    if "hour" in feats:
        df["hour"] = df["datetime"].dt.hour
        logger.info("Added 'hour' col to df.")

    return df.drop(columns=["datetime"]).copy()


# ,id,
# lat_norm,lon_norm,
# datetime,
# light conditions, intersection type, weather,
# collision type,
# h3_res4,h3_res5,h3_res6,h3_res7

# ------------------
# MAIN FUNCTION
# ------------------


def prepare_data_2A():
    # load env variables
    gh.load_env_vars()

    # load configurations
    session.load_config(config)
    log_name = session.gen_params.get("log_name", None)  # "ETL_CHARACTERISTICS"
    name_logfile = session.gen_params.get("log_file")  # "etl_characteristics"
    target_col = session.exp_params.get("target_col", None)
    # h3_idx = session.prep_params.get("h3_idx", None)

    # extract configurations from session
    # df_path = session.exp_params.get("df_path", None)
    # split = session.exp_params.get("split_method", None)
    # feats = session.exp_params.get("features", None)

    # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    logger.info(
        "Start preparing df for Phase 2A (Influence factors on '%s').", target_col
    )

    # (0) load non-aggregated data
    df = load_base_df()

    for col in ["light conditions", "intersection type", "weather", "collision type"]:
        logger.info("NaN count in '%s':\t%s", col, df[col].isna().sum())
        logger.info("Inf count in '%s':\t%s", col, np.sum(np.isinf(df[col])))

    df = df.dropna(
        subset=["light conditions", "intersection type", "weather", "collision type"]
    )
    # (1) add encoded 'time'
    df_1 = add_time_cols(df)
    df_time = feat.cyclic_encode_col(df_1)

    print(df_time.head())
    # (2) aggregate + inflate data df
    # df_group = aggregate_df(df_cat)
    # df_inf = inflate_df(df_group)

    # # (3) add seasonality + time and 'zero_features'
    # df_temp = add_temporal_features(df_inf)
    # df_zero = add_zero_features(df_temp)

    # # (4) add 'target'
    # df_final = add_target(df_zero)

    # save df
    f_name = session.prep_params.get("df_prep_name", None)
    fh.save_df_to_parquet(df_time, f_name, chunked=True)

    return


if __name__ == "__main__":
    prepare_data_2A()
