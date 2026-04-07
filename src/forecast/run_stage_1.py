## run_stage_1.py
# imports
import click
import os
from pathlib import Path
from datetime import datetime
# from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import torch

import src.utils.general_helper as gh
import src.utils.evaluation_helper as eval
import src.utils.file_helper as fh
import src.utils.path_helper as ph
import src.utils.df_helper as dfh

from src.utils.split_helper import simple_time_split
from src.utils.file_helper import get_yaml_config

from src.core.session import session
from src.core.logger import create_logger
from src.H3_graph_building.prepare_static_graph import (create_edges_indexes, 
                                                        create_snapshots, 
                                                        create_data_from_snapshots)
from src.forecast.predict_by_ml_model import predict_by_ml_model, prepare_ml_data
from src.forecast.predict_by_gcn import predict_by_simple_bin_gcn
# from src.forecast.predict_by_light_gbm import predict_by_light_gbm
# from Road_accidents.src.forecast.predict_by_ml_model import predict_by_xgboost
# from src.forecast.predict_by_catboost import predict_by_catboost



# ------------------
# MAIN FUNCTION
# ------------------

@click.command()
@click.option("--config_name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def stage_1_prediction(config_name):  
    
    run_stage_1_prediction(config_name)

    return


def run_stage_1_prediction(config_name):
    # load env variables
    gh.load_env_vars()
    data_processed = os.getenv("PATH_PROCESSED")
    report= os.getenv("FOLDER_REPORT")

    # load values from config
    config = get_yaml_config(config_name)
    general_config = config.get("general_args", {})
    log_name = general_config["name_log_fore"]
    name_logfile = general_config["name_logfile_fore"]
    forecast_models = general_config["forecast_models"]

    period_col = general_config["period_col"]
    h3_col = general_config["h3_col"]
    target_col = general_config["target_col"]

    df_name = general_config["df_name"]
    data_folder = Path(general_config["save_folder"]) # , "")
    df_folder = Path(f"{data_processed}/{data_folder}")

    lr_config = config.get("log_reg", {})
    gcn_config = config.get("simple_bin_gcn", {})
    light_config = config.get("light_gbm", {})
    xgb_config = config.get("xgboost", {})
    cat_config = config.get("catboost", {})

    # --- create logger --
    logger = create_logger(name=log_name, file_name=name_logfile)

    # store importance objects/values in session 
    eval_folder = os.getenv("PATH_EVALUATED")
    now = "2026-03-27_14-55-26"      # datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    session.logger = logger
    session.df_name = df_name
    session.now = now
    session.save_folder = f"{eval_folder}/{now}"

    # --- load data ---
    f_path = f"{df_folder}/{df_name}.parquet"
    df = pd.read_parquet(f_path)

    df_sort, split_data, preprocessor = prepare_ml_data(df, general_config)

    feats_clean = [col for col in df_sort.columns if col not in [target_col, h3_col, period_col]]
    
    # print("head of df (loaded):\n", df.head(3))
    # data_list = torch.load(f"{feat_folder}/{build_time}_{data_name}.pt",
    #                        weights_only=False)   
    # # split data
    # train_data, test_data, val_data = simple_time_split(data_list, gnn_dict)

    # -------------------
    # 'SIMPLE' ML MODELS 
    # -------------------
    if "LogisticRegression" in forecast_models:

        # session.run_name = "log_reg" 
                    
        model_pipe = predict_by_ml_model(
                        split_data, 
                        "log_reg",
                        lr_config, 
                        preprocess=preprocessor
                        )

        eval.create_coef_df(model_pipe, save=True, data_viz=True)

    # ----------------------
    # 'BOOSTED' TREE MODELS 
    # ----------------------
    # Gradient Boosting (XGBoost, LightGBM, CatBoost) --> tsfresh
    if "LGBMClassifier" in forecast_models:

        # session.run_name = "light_gbm"

        model_pipe = predict_by_ml_model(
                        split_data, 
                        "light_gbm",
                        light_config, 
                        preprocess=preprocessor
                        )

        eval.create_tree_importance_df(model_pipe, save=True)

    if "XGBClassifier" in forecast_models:

        # session.run_name = "xgboost"

        model_pipe = predict_by_ml_model(
                                    split_data, 
                                    "xgboost",
                                    xgb_config, 
                                    preprocess=preprocessor
                                    )
        
        eval.create_tree_importance_df(model_pipe, save=True)

    if "CatBoostClassifier" in forecast_models:

        # session.run_name = "cat_boost"

        model_pipe = predict_by_ml_model(
                                    split_data, 
                                    "catboost",
                                    cat_config, 
                                    preprocess=preprocessor
                                    ) 
        
        eval.create_tree_importance_df(model_pipe, save=True)

    # ----------------------
    # TS-SPECIFIC MODELS 
    # ----------------------
    # ROCKET (RandOm Convolutional KErnel Transform) --> sktime


    # -----------------
    # GNN MODELS 
    # -----------------
    # = 1D Convolutional Neural Networks (1D-CNN)!?
    # used_gnn = False
    # gnn_timestamp = None

    if "SimpleBinaryGCN" in forecast_models:

        gnn_folder = os.getenv("FOLDER_GNN")
        graph_folder = Path(f"{gnn_folder}/graph")
        feat_folder = Path(f"{gnn_folder}/features")

        # graph_data = fh.load_dict(f"{graph_folder}/{timestamp}_graph_meta.json")
        # df, gnn_settings, folder, timestamp)
        now_gcn = session.now
        graph_meta, edge_index = create_edges_indexes(
                                                    df_sort, 
                                                    general_config,
                                                    gcn_config, 
                                                    graph_folder, 
                                                    now_gcn
                                                    )
        # assert not df_sort.duplicated([h3_col, period_col]).any()
        # assert df_sort[h3_col].notna().all()
        # assert df[period_col].notna().all()

        snapshots = create_snapshots(
                                df_sort, 
                                general_config, 
                                graph_meta
                                )
        
        snap_path = f"{feat_folder}/{now_gcn}_snapshots_base.pt"
        torch.save(snapshots, snap_path)
        logger.info("Snapshots saved as '%s'",
                    ph.shorten_path(snap_path))
        
        data_list = create_data_from_snapshots(
                                        snapshots, 
                                        edge_index,
                                        feat_folder,
                                        now_gcn
                                        )
        
        data_list = torch.load(f"{feat_folder}/{now_gcn}_data_base.pt",
                               weights_only=False)

        train_data, test_data, val_data = simple_time_split(data_list, 
                                                            general_config)

        data = {
            "feats": feats_clean, 
            "train": train_data,
            "test": test_data,
            "val": val_data
            }
        is_preprocess = (isinstance(preprocessor, ColumnTransformer) & 
                         len(preprocessor.transformers) > 0)
        
        weighted = gcn_config["weighted"]
        # gcn_results, gcn_meta = 
        predict_by_simple_bin_gcn(
                            data,
                            general_config, 
                            gcn_config,
                            preprocess=is_preprocess
                            )

        # gcn_results["file"] = df_name
        # results.append({"simple_bin_gcn": gcn_results})
        
        # now = session.now
        # weight_suf = "weighted" if weighted else "unweighted"
        # results_path = f"{report}/{now}_gcn_{weight_suf}_results.pt"
        # meta_path = f"{report}/{now}_gcn_{weight_suf}_meta.json"

        # torch.save(gcn_results, results_path)
        # fh.save_dict(gcn_meta, meta_path)
        # 
        # used_gnn = True
        # if gnn_timestamp is None:
        #     gnn_timestamp = now 

    
    # -----------------
    # RNN MODELS 
    # -----------------
    # RNN / LSTM / GRU (TensorFlow/Keras oder PyTorch)
    # attention-based / transformers!?


    # # evaluate performance of baseline run
    # phase_1_evaluation(data_dict)


    # # threshold_sweep_analysis to optimize initial results
    # threshold_sweep_analysis(data_dict, metric="f2")

if __name__ == "__main__":
    stage_1_prediction()



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


# def build_logreg_model():
#     class_weight = session.exp_params.get("class_weight", None)
#     l1_ratio = session.exp_params.get("l1_ratio", 0.0)
#     solver = session.exp_params.get("solver", "lbfgs")
#     random_state = session.exp_params.get("random_state", 42)
#     max_iter = session.exp_params.get("max_iter", 1000)

#     lr_model = LogisticRegression(
#         max_iter=max_iter,
#         l1_ratio=l1_ratio,
#         class_weight=class_weight,
#         random_state=random_state,
#         solver=solver,
#     )

#     return lr_model



# # def build_preprocessing():

# #     num_feats = session.prep_params.get("num_feats", None)

# #     preprocessor = ColumnTransformer(
# #         transformers=[("num", StandardScaler(), num_feats)], remainder="passthrough"
# #     )

# #     return preprocessor


# # def predict_by_logreg(train, test, val=None):
# #     # setup logger
# #     logger = session.logger

# #     #
# #     target_col = session.exp_params.get("target_final", None)

# #     # prepare pipeline
# #     pipe = Pipeline(
# #         [("preprocess", build_preprocessing()), ("model", build_logreg_model())]
# #     )

# #     # prepare data
# #     data_dict = {}
# #     data_list = [("train", train), ("test", test)]

# #     if val is not None:
# #         logger.info("Validation dataset is provided.")
# #         data_list.append(("val", val))

# #     for name, t in data_list:
# #         t_enc = target_encode_col(t)
# #         # cyclic_encode_col()
# #         t_clean = clean_prep_data(t_enc)
# #         logger.info("head of '%s_df':\n%s", name, t_clean.head())

# #         data_dict[f"X_{name}"] = t_clean.drop(columns=target_col).copy()
# #         data_dict[f"y_{name}"] = t_clean[target_col]

# #     # train model
# #     now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# #     X_train = data_dict["X_train"]
# #     y_train = data_dict["y_train"]

# #     pipe.fit(X_train, y_train)

# #     # make predictions
# #     X_test = data_dict["X_test"]

# #     data_dict["y_pred"] = pipe.predict(X_test)
# #     data_dict["y_proba"] = pipe.predict_proba(X_test)[:, 1]

# #     if val is not None:
# #         X_val = data_dict["X_val"]
# #         # data_dict["y_pred_val"] = pipe.predict(X_val)
# #         data_dict["y_proba_val"] = pipe.predict_proba(X_val)[:, 1]

# #     logger.info("Completed training and prediction.")

# #     session.exp_params["now"] = now

# #     feats = X_train.columns

# #     eval.create_coef_df(pipe, feats, save=True)

# #     return data_dict


# # def clean_prep_data(df):
# #     # setup logger
# #     logger = session.logger
# #     time_col = session.exp_params.get("time_col", None)

# #     #
# #     col_to_drop = [
# #         "year",
# #         "lag_3",  # löschen oder 'lag_2' ergänzen + 'rol_mean_3' löschen
# #         "was_zero",
# #         "n_accidents",
# #         time_col,
# #         #    group_col
# #     ]

# #     for p in ["month", "week"]:
# #         if p in df.columns:
# #             col_to_drop.append(p)

# #     df_clean = df.drop(columns=col_to_drop).copy()

# #     for col in df_clean.columns:
# #         if col.startswith("month_"):
# #             df_clean[col] = df[col].astype(int)

# #     logger.info("Distribution:\n%s", df_clean["risk_next"].value_counts(normalize=True))

# #     return df_clean

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
