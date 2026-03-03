## 🚧 DEV: Logistische Regression für Unfallprognose 🚧
# imports
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.file_helper as fh
import src.utils.evaluation_helper as eval

from src.core.session import session
from src.core.logger import create_logger
from configuration.Phase_2A_LogReg_base import config

# from src.forecast.phase_1_evaluate import phase_1_evaluation
# from src.forecast.threshold_sweep_analysis import threshold_sweep_analysis


def load_prepared_data():
    # setup logger
    logger = session.logger

    #
    df_name = session.prep_params.get("df_prep_name", None)
    df_folder = os.getenv("PATH_PROCESSED")
    f_path = Path(df_folder) / f"{df_name}.parquet"
    df = pd.read_parquet(f_path)

    logger.info(
        "Loaded prepared df from '../%s'\nshape:\t%s", ph.shorten_path(f_path), df.shape
    )

    return df


def build_logreg_model():
    # class_weight = session.exp_params.get("class_weight", None)
    # l1_ratio = session.exp_params.get("l1_ratio", 0.0)
    solver = session.exp_params.get("solver", "lbfgs")
    random_state = session.exp_params.get("random_state", 42)
    max_iter = session.exp_params.get("max_iter", 1000)
    # multi_class = session.exp_params.get("multi_class", None)

    lr_model = LogisticRegression(
        # multi_class=multi_class,
        max_iter=max_iter,
        # l1_ratio=l1_ratio,
        # class_weight=class_weight,
        random_state=random_state,
        solver=solver,
    )

    return lr_model


def build_preprocessing():

    cat_feats = session.prep_params.get("cat_feats", None)
    num_feats = session.prep_params.get("cat_feats", None)

    #     cat_cols = ["weather", "intersection_type", "light_conditions"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), cat_feats),
            # ("num", StandardScaler(), num_feats)
        ],
        # remainder="passthrough"
    )

    return preprocessor


def split_data(df, vali=False):
    # setup logger
    logger = session.logger

    target_col = session.exp_params.get("target_col", None)
    test_size = session.exp_params.get("test_size", float(0.2))
    random_state = session.exp_params.get("random_state", 42)

    #
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # print("X:", type(X), getattr(X, "shape", None))
    # print("\ny:", type(y), getattr(y, "shape", None))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if vali:
        X_tr, X_vali, y_tr, y_vali = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            "Splitted data into train, vali and test group (test_size=%s)", test_size
        )

        return X_tr, y_tr, X_vali, y_vali, X_test, y_test

    logger.info("Splitted data into train and test group (test_size=%s)", test_size)

    return X_train, y_train, X_test, y_test

    # for name, t in data_list:
    #     # t_enc = target_encode_col(t)
    #     # # cyclic_encode_col()
    #     # t_clean = clean_prep_data(t_enc)
    #     logger.info("head of '%s_df':\n%s",
    #                 name,
    #                 t.head())

    #     data_dict[f"X_{name}"] = t.drop(columns=target_col).copy()
    #     data_dict[f"y_{name}"] = t[target_col]


def analyse_by_logreg(df, val=None):
    # setup logger
    logger = session.logger

    # prepare pipeline
    pipe = Pipeline(
        [("preprocess", build_preprocessing()), ("model", build_logreg_model())]
    )

    # split data
    data_dict = {}

    X_train, y_train, X_test, y_test = split_data(df, vali=False)

    # train model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]

    logger.info(
        "Check after model_fit.\nShape of 'coef:\t%s\nClasses:\n%s",
        model.coef_.shape,
        model.classes_,
    )

    fh.save_model(pipe, "base_LogReg_pipe_p2")

    data_dict["X_train"] = X_train
    data_dict["y_train"] = y_train
    data_dict["X_test"] = X_test
    data_dict["y_test"] = y_test

    # make predictions
    data_dict["y_pred"] = pipe.predict(X_test)
    # data_dict["y_proba"] = pipe.predict_proba(X_test)[:, 1]

    proba = pipe.predict_proba(X_test)

    data_dict["y_proba"] = pd.DataFrame(
        proba, columns=pipe.named_steps["model"].classes_
    )

    if val is not None:
        X_val = data_dict["X_val"]
        # data_dict["y_pred_val"] = pipe.predict(X_val)
        data_dict["y_proba_val"] = pipe.predict_proba(X_val)[:, 1]

    logger.info("Completed training and prediction.")

    session.exp_params["now"] = now

    # if "preprocess" in pipe.named_steps:
    # feats = pipe.named_steps["preprocess"].get_feature_names_out()

    # else:
    #     feats = X_train.columns

    eval.create_coef_df(pipe, save=True, data_viz=True)

    eval.importance_by_permutation(pipe, data_dict, data_viz=True)

    return data_dict


# print(feature_importance)

# def clean_prep_data(df):
#     # setup logger
#     logger = session.logger
#     time_col = session.exp_params.get("time_col", None)

#     #
#     col_to_drop = ["year",
#                    "lag_3",     # löschen oder 'lag_2' ergänzen + 'rol_mean_3' löschen
#                    "was_zero",
#                    "n_accidents",
#                    time_col,
#                 #    group_col
#                    ]

#     for p in ["month", "week"]:
#         if p in df.columns:
#             col_to_drop.append(p)

#     df_clean = df.drop(columns=col_to_drop).copy()

#     for col in df_clean.columns:
#         if col.startswith("month_"):
#             df_clean[col] = df[col].astype(int)

#     logger.info("Distribution:\n%s",
#                 df_clean["risk_next"].value_counts(normalize=True))

#     return df_clean


# def target_encode_col(train_df):
#     # setup logger
#     logger = session.logger

#     #
#     group_col = session.exp_params.get("encode_space", None)
#     target_col = session.exp_params.get("target_final", None)

#     mean_val = (train_df
#                 .sort_values(group_col)
#                 .groupby(group_col)[target_col]
#                 .mean()
#                 )

#     glob_mean = train_df[target_col].mean()

#     df_enc = train_df.drop(columns=[group_col]).copy()
#     df_enc[f"{group_col}_te"] = (
#                         train_df[group_col]
#                         .map(mean_val)
#                         # .fillna(glob_mean)
#                         )

#     logger.info("NaN count in '%s':\t%s",
#           f"{group_col}_te",
#           df_enc[f"{group_col}_te"].isna().sum())

#     # encode_dict = {mean_val[i, 0]:mean_val[i, 1] for i in len(mean_val)}

#     # df_encoded = df.replace(encode_dict)

#     return df_enc


# ------------------
# MAIN FUNCTION
# ------------------


def risk_analysis():
    # load env variables
    gh.load_env_vars()

    # load configurations
    session.load_config(config)
    log_name = session.gen_params.get("log_name", None)  # "ETL_CHARACTERISTICS"
    name_logfile = session.gen_params.get("log_file", None)

    # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    # (0) load data
    df = load_prepared_data()

    # (1) build pipeline + train model +
    data_dict = analyse_by_logreg(df)

    return


if __name__ == "__main__":
    risk_analysis()

    # Daten sortieren (sehr wichtig!)
    # df_sort = df.sort_values(time_col).reset_index(drop=True).copy()

    # if  split == "sklearn":
    #     idx_dict = sklearn_time_split(df_sort)

    # if split == "simple":
    #     train, test, val = simple_time_split(df_sort)

    # # evaluate performance of baseline run
    # phase_1_evaluation(data_dict)

    # # threshold_sweep_analysis to optimize initial results
    # threshold_sweep_analysis(data_dict, metric="f2")

    # extract configurations from session
    # # h3_idx = session.h3_idx
    # split = session.exp_params.get("split_method", None)
    # # feats = session.features
    # time_col = session.exp_params.get("time_col", None)
    # num_feats = session.gen_params.get("num_feats", None)


# def clean_prep_data(df):
#     # setup logger
#     logger = session.logger

#     # add missing 'months' + 'years'
#     df["year"] = df["period"].dt.year
#     df["month"] = df["period"].dt.month

#     # sort df + drop first rows
#     h3_idx = session.prep_params.get("h3_idx", None)
#     max_lag = session.exp_params.get("max_lag", None)
#     group_cols = [h3_idx, "period"]

#     df_sort = df.sort_values(group_cols)
#     df_group = df_sort.groupby(group_cols).apply(
#                             lambda x: x.iloc[max_lag:]
#                             ).reset_index(drop=True)


# # (2) Basis-Features (Minimal sinnvolle Version)
# # lag features (monthly base)
# df["lag_1"] = df.groupby("h3_index")["n_accidents"].shift(1)
# df["lag_2"] = df.groupby("h3_index")["n_accidents"].shift(2)
# # df["lag_4"] = df.groupby("h3_index")["n_accidents"].shift(4)      # lag features (weekly base)
# # df["lag_12"] = df.groupby("h3_index")["n_accidents"].shift(12)

# # rolling features
# df["roll_mean_3"] = (
#     df.groupby("h3_index")["n_accidents"]
#       .rolling(3)
#       .mean()
#       .reset_index(level=0, drop=True)
# )
# # df["roll_mean_4"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(4)
# #       .mean()
# #       .reset_index(level=0, drop=True)
# # )
# # df["roll_mean_12"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(12)
# #       .mean()
# #       .reset_index(level=0, drop=True)
# # )

# df["roll_sum_3"] = (
#     df.groupby("h3_index")["n_accidents"]
#       .rolling(3)
#       .sum()
#       .reset_index(level=0, drop=True)
# )
# # df["roll_sum_4"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(4)
# #       .sum()
# #       .reset_index(level=0, drop=True)
# # )
# # df["roll_sum_12"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(12)
# #       .sum()
# #       .reset_index(level=0, drop=True)
# # )

# # Zero Persistence
# df["was_zero"] = (df["n_accidents"] == 0).astype(int)
# df["zero_streak"] = (
#     df.groupby("h3_index")["was_zero"]
#       .cumsum()
# )

# # seasonality features
# df["month"] = df["datetime"].dt.month
# df = pd.get_dummies(df, columns=["month"], drop_first=True)

# df["weekday"] = df["datetime"].dt.weekday
# # df["weekday"] = df["week_start"].dt.weekday
# df = pd.get_dummies(df, columns=["weekday"], drop_first=True)

# # df["week"] = df[datetime"].dt.week
# # df = pd.get_dummies(df, columns=["week"], drop_first=True)

# # (3) Spatial features
# # mean_neighbor_risk_t

# # (5) Data Cleaning
# df = df.dropna()

# # (6) Build feature matrix
# features = ""

# month_cols = [c for c in df.columns if c.startswith("month_")]
# week_cols = [c for c in df.columns if c.startswith("week_")]
