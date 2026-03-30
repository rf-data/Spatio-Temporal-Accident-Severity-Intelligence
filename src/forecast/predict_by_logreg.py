## predict_by_logreg.py
# imports
# import os
# from pathlib import Path
from datetime import datetime
# from typing import Callable
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# import numpy as np
# import pandas as pd

# import src.utils.general_helper as gh
# import src.utils.path_helper as ph
import src.utils.evaluation_helper as eval

from src.core.session import session
from src.core.logger import ModelLogger

# import json
# import time
# 


# ------------------
# MAIN FUNCTION
# ------------------

def predict_by_logreg(
                    data: dict, 
                    lr_config, 
                    preprocess: ColumnTransformer=None
                    ):
    # setup logger
    logger = session.logger
    logger.info("Start Training 'Logistic Regression'")

    # extract data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"] 

    # prepare pipeline
    model = build_logreg_model(lr_config)
    
    if preprocess:
        model_pipe = Pipeline(
            [("preprocess", preprocess), ("model", model)]
        )
        logger.info("Created pipeline (preprocessing + log_reg_model)")
    
    else: 
        model_pipe = model
        logger.info("Created pipeline (log_reg_model only)")

    # train model
    model_pipe.fit(X_train, y_train)

    # make forecast
    y_pred = model_pipe.predict(X_test)
    y_proba = model_pipe.predict_proba(X_test)[:, 1]

    if len(X_val) > 0:
        # data_dict["y_pred_val"] = pipe.predict(X_val)
        y_val_pred = model_pipe.predict(X_val)
        y_val_proba = model_pipe.predict_proba(X_val)[:, 1]

    else:
        y_val_pred = None
        y_val_proba = None

    logger.info("Completed LogReg training and prediction.")

    folder = session.save_folder
    timestamp=session.now

    ml_logger = ModelLogger(base_path=folder)
    ml_logger.log_run(
                    model,
                    X_test,
                    y_test,
                    y_pred,
                    y_proba,
                    y_val_pred,
                    y_val_proba,
                    timestamp
                    )
    
    return 



if __name__ == "__main__":
    predict_by_logreg()

# # 1️⃣ LogReg + time-aware split
# # Du brauchst:
# # Lag-Features (t-1, t-7, t-30)
# # Rolling Means
# # Seasonality Features
# # Weather
# # Weekday
# # Zero-Include nur in aktiven Zellen.

# # Sonst explodiert Feature-Matrix.

# # def create_feature_matrix(df):
# #     # (2) Basis-Features
# #     df_features = create_features(df)

# #     # (3) Data Cleaning
# #     df_clean = df_features.dropna().copy()

# #     # (4) Build feature matrix
# #     # features = session.features
# #     month_cols = [c for c in df_clean.columns if c.startswith("month_")]
# #     week_cols = [c for c in df_clean.columns if c.startswith("week_")]

# #     if len(month_cols) > 0:
# #         cols = month_cols
# #     elif len(week_cols) > 0:
# #         cols = week_cols
# #     else:
# #         logger.error("No month or week columns found")
# #         raise ValueError("No month or week columns found")

# #     # X = df_clean[features + cols]   # week_cols
# #     # y = df_clean["target"]

# #     return df_clean, cols   # X, y


# """
#     Wichtige Best Practices
#     Sortieren: Stellen Sie sicher, dass Ihre Daten vor dem Split zwingend nach Zeit sortiert sind.
#     Shuffle=False: Falls Sie train_test_split aus Sklearn nutzen, setzen Sie shuffle=False, um das zufällige Mischen zu verhindern.
#     Lücken (Gap): Manchmal ist es sinnvoll, eine Lücke zwischen Trainings- und Testset zu lassen (z.B. gap=1 in TimeSeriesSplit), um zu verhindern, dass Autokorrelationen die Ergebnisse verfälschen.
#     Ziel: Die Testmenge sollte so gewählt werden, dass sie zukünftige, unbekannte Daten simuliert. 
#     """


# def simple_time_split(df_sort):
#     """
#     Einfacher Cutoff-Split (Holdout)
#     Dies ist die einfachste Methode, bei der Sie einen festen Zeitpunkt (Stichtag) wählen.
#     Training:   Alle Datenpunkte vor dem Stichtag.
#     Test:       Alle Datenpunkte nach dem Stichtag.
#     """
#     # setup logger
#     logger = session.logger
#     train_size = session.exp_params.get("train_size", None)

#     # Festlegen des Trennzeitpunkts (z.B. 80% Training, 20% Test)
#     test_idx = int(len(df_sort) * float(train_size))
#     test = df_sort.iloc[test_idx:]

#     val_idx = int(len(test) * float(train_size))
#     train = df_sort.iloc[:val_idx]
#     val = df_sort.iloc[val_idx:test_idx]

#     # Überprüfung
#     # logger.info("Conducted simple time-aware split on col '%s' (train_size = %s)",
#     #             time_col,
#     #             train_size)
#     # logger.info("Period 'train': %s to %s",
#     #       train[time_col].min(),
#     #       train[time_col].max())
#     # logger.info("Period 'test': %s to %s",
#     #       test[time_col].min(),
#     #       test[time_col].max())

#     return train, test, val


# def sklearn_time_split(df_sort):
#     # setup logger
#     logger = session.logger

#     # 3 Splits erzeugen
#     n_splits = session.exp_params.get("n_splits", None)
#     gap = session.exp_params.get("gap", 0)

#     tscv = TimeSeriesSplit(n_splits, gap)

#     idx_dict = {}
#     for i, (train_index, test_index) in enumerate(tscv.split(df_sort)):
#         idx_dict[f"split_{i+1}"] = {"train": train_index, "test": test_index}
#         logger.info("[SPLIT %s] Period 'train': %s", i + 1, train_index)
#         logger.info("[SPLIT %s] Period 'test': %s", i + 1, test_index)

#     logger.info(
#         "Completed sklearn time-aware split (n_splits = %s, gap = %s)", n_splits, gap
#     )

#     return idx_dict

#     # sliding/rolling window
#     """
#     3. Sliding/Rolling Window
#     Hierbei wird ein festes Zeitfenster als Training genutzt, um die unmittelbar 
#     folgenden Zeitschritte vorherzusagen. Das Fenster bewegt sich (slidet) durch 
#     die Zeit. Dies ist ideal, wenn sich die Datenstruktur über die Zeit stark ändert. 
#     """
#     # return


# def load_prepared_data():
#     # setup logger
#     logger = session.logger

#     #
#     df_name = session.prep_params.get("df_prep_name", None)
#     df_folder = os.getenv("PATH_PROCESSED")
#     f_path = Path(df_folder) / f"{df_name}.parquet"
#     df = pd.read_parquet(f_path)

#     logger.info("Loaded prepared df from '../%s'", ph.shorten_path(f_path))

#     return df



# def clean_prep_data(df):
#     # setup logger
#     logger = session.logger
#     time_col = session.exp_params.get("time_col", None)

#     #
#     col_to_drop = [
#         "year",
#         "lag_3",  # löschen oder 'lag_2' ergänzen + 'rol_mean_3' löschen
#         "was_zero",
#         "n_accidents",
#         time_col,
#         #    group_col
#     ]

#     for p in ["month", "week"]:
#         if p in df.columns:
#             col_to_drop.append(p)

#     df_clean = df.drop(columns=col_to_drop).copy()

#     for col in df_clean.columns:
#         if col.startswith("month_"):
#             df_clean[col] = df[col].astype(int)

#     logger.info("Distribution:\n%s", df_clean["risk_next"].value_counts(normalize=True))

#     return df_clean


# # def compute_smoothed_target_encoding(train_df, group_col, target_col, alpha=10):

# #     global_mean = train_df[target_col].mean()

# #     stats = (
# #         train_df
# #         .groupby(group_col)[target_col]
# #         .agg(["mean", "count"])
# #     )

# #     smooth = (
# #         (stats["count"] * stats["mean"] + alpha * global_mean)
# #         /
# #         (stats["count"] + alpha)
# #     )

# #     return smooth, global_mean


# def target_encode_col(train_df):
#     # setup logger
#     logger = session.logger

#     #
#     group_col = session.exp_params.get("encode_space", None)
#     target_col = session.exp_params.get("target_final", None)

#     mean_val = train_df.sort_values(group_col).groupby(group_col)[target_col].mean()

#     glob_mean = train_df[target_col].mean()

#     df_enc = train_df.drop(columns=[group_col]).copy()
#     df_enc[f"{group_col}_te"] = (
#         train_df[group_col].map(mean_val)
#         # .fillna(glob_mean)
#     )

#     logger.info(
#         "NaN count in '%s':\t%s",
#         f"{group_col}_te",
#         df_enc[f"{group_col}_te"].isna().sum(),
#     )

#     # encode_dict = {mean_val[i, 0]:mean_val[i, 1] for i in len(mean_val)}

#     # df_encoded = df.replace(encode_dict)

#     return df_enc


# def phase_1_prediction():
#     # load env variables
#     gh.load_env_vars()

#     # load configurations
#     session.load_config(config)
#     log_name = session.gen_params.get("log_name", None)  # "ETL_CHARACTERISTICS"
#     name_logfile = session.gen_params.get("log_file", None)

#     # extract configurations from session
#     # h3_idx = session.h3_idx
#     split = session.exp_params.get("split_method", None)
#     # feats = session.features
#     time_col = session.exp_params.get("time_col", None)
#     # num_feats = session.gen_params.get("num_feats", None)

#     # load logger
#     logger = create_logger(name=log_name, file_name=name_logfile)

#     session.logger = logger

#     # (0) load data
#     df = load_prepared_data()

#     # (1) time_aware data split
#     # Daten sortieren (sehr wichtig!)
#     df_sort = df.sort_values(time_col).reset_index(drop=True).copy()

#     if split == "sklearn":
#         idx_dict = sklearn_time_split(df_sort)

#     if split == "simple":
#         train, test, val = simple_time_split(df_sort)

#     # build pipeline + train model +
#     data_dict = predict_by_logreg(train, test, val)

#     # evaluate performance of baseline run
#     phase_1_evaluation(data_dict)

#     # threshold_sweep_analysis to optimize initial results
#     threshold_sweep_analysis(data_dict, metric="f2")

#     return



# # def clean_prep_data(df):
# #     # setup logger
# #     logger = session.logger

# #     # add missing 'months' + 'years'
# #     df["year"] = df["period"].dt.year
# #     df["month"] = df["period"].dt.month

# #     # sort df + drop first rows
# #     h3_idx = session.prep_params.get("h3_idx", None)
# #     max_lag = session.exp_params.get("max_lag", None)
# #     group_cols = [h3_idx, "period"]

# #     df_sort = df.sort_values(group_cols)
# #     df_group = df_sort.groupby(group_cols).apply(
# #                             lambda x: x.iloc[max_lag:]
# #                             ).reset_index(drop=True)


# # # (2) Basis-Features (Minimal sinnvolle Version)
# # # lag features (monthly base)
# # df["lag_1"] = df.groupby("h3_index")["n_accidents"].shift(1)
# # df["lag_2"] = df.groupby("h3_index")["n_accidents"].shift(2)
# # # df["lag_4"] = df.groupby("h3_index")["n_accidents"].shift(4)      # lag features (weekly base)
# # # df["lag_12"] = df.groupby("h3_index")["n_accidents"].shift(12)

# # # rolling features
# # df["roll_mean_3"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(3)
# #       .mean()
# #       .reset_index(level=0, drop=True)
# # )
# # # df["roll_mean_4"] = (
# # #     df.groupby("h3_index")["n_accidents"]
# # #       .rolling(4)
# # #       .mean()
# # #       .reset_index(level=0, drop=True)
# # # )
# # # df["roll_mean_12"] = (
# # #     df.groupby("h3_index")["n_accidents"]
# # #       .rolling(12)
# # #       .mean()
# # #       .reset_index(level=0, drop=True)
# # # )

# # df["roll_sum_3"] = (
# #     df.groupby("h3_index")["n_accidents"]
# #       .rolling(3)
# #       .sum()
# #       .reset_index(level=0, drop=True)
# # )
# # # df["roll_sum_4"] = (
# # #     df.groupby("h3_index")["n_accidents"]
# # #       .rolling(4)
# # #       .sum()
# # #       .reset_index(level=0, drop=True)
# # # )
# # # df["roll_sum_12"] = (
# # #     df.groupby("h3_index")["n_accidents"]
# # #       .rolling(12)
# # #       .sum()
# # #       .reset_index(level=0, drop=True)
# # # )

# # # Zero Persistence
# # df["was_zero"] = (df["n_accidents"] == 0).astype(int)
# # df["zero_streak"] = (
# #     df.groupby("h3_index")["was_zero"]
# #       .cumsum()
# # )

# # # seasonality features
# # df["month"] = df["datetime"].dt.month
# # df = pd.get_dummies(df, columns=["month"], drop_first=True)

# # df["weekday"] = df["datetime"].dt.weekday
# # # df["weekday"] = df["week_start"].dt.weekday
# # df = pd.get_dummies(df, columns=["weekday"], drop_first=True)

# # # df["week"] = df[datetime"].dt.week
# # # df = pd.get_dummies(df, columns=["week"], drop_first=True)

# # # (3) Spatial features
# # # mean_neighbor_risk_t

# # # (5) Data Cleaning
# # df = df.dropna()

# # # (6) Build feature matrix
# # features = ""

# # month_cols = [c for c in df.columns if c.startswith("month_")]
# # week_cols = [c for c in df.columns if c.startswith("week_")]
