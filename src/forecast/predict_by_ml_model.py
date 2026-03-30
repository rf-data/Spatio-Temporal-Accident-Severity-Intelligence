## predict_by_logreg.py
# imports
# import os
# from pathlib import Path
from datetime import datetime
# from typing import Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# import src.utils.general_helper as gh
# import src.utils.path_helper as ph

from src.core.session import session
from src.core.logger import ModelLogger
from src.core.ml_models import (build_logreg_model,
                                    build_xgboost_model,
                                    build_catboost_model,
                                    build_light_gbm_model)
from src.utils.split_helper import simple_time_split

# import json
# import time
# 


model_dict = {
    "xgboost": build_xgboost_model,
    "catboost": build_catboost_model,
    "light_gbm": build_light_gbm_model,
    "log_reg": build_logreg_model  
    }

# ------------------
# MAIN FUNCTION
# ------------------

def build_preprocessing(general_config):

    num_cols = general_config["num_cols"] # , None)
    bin_cols = general_config["bin_cols"] 

    preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("bin", "passthrough", bin_cols)
                ]
            )

    return preprocessor


def prepare_ml_data(df, general_config):
    # setup logger
    logger = session.logger

    # 
    period_col = general_config["period_col"]
    h3_col = general_config["h3_col"]
    target_col = general_config["target_col"]
    clip_values = general_config.get("clip_values", {})

    for col_name, clip_val in clip_values.items():
        df[col_name] =  df[col_name].clip(upper=clip_val)
        
    # --- split data ---
    df_sort = df.sort_values(period_col).reset_index(drop=True).copy()

    feats = df_sort.columns
    feats_clean = [col for col in feats 
                   if col not in [target_col, h3_col, period_col]] 
        
    # print("head of df_sort:\n", df_sort.head(3))
    train_data, test_data, val_data = simple_time_split(df_sort, general_config)
    
    split_data =  {}
    for name, data in [("train", train_data),
                       ("test", test_data),
                       ("val", val_data)]:
        
        if len(data) > 0:

            assert not ((data["has_accident"] == 1) & (data["zero_streak"] != 0)).any() 

            # data = pd.DataFrame(data, columns=)
            y = data[target_col]

            split_data[f"X_{name}"] = pd.DataFrame(data[feats_clean]) # .drop(columns=[target_col])
            split_data[f"y_{name}"] = y
            
            logger.info("[DEBUG] data head (%s):\n%s",
                    name,
                    data.head(3))
            logger.info("[DEBUG] 'target' values (%s):\n%s",
                    name,
                    y.value_counts())

        else: 
            split_data[f"X_{name}"] = []
            split_data[f"y_{name}"] = []
            logger.info("%s is empty",
                    name)

    # # --- preprocessing ---
    preprocessor = build_preprocessing(general_config)

    return df_sort, split_data, preprocessor


def predict_by_ml_model(
                    data: dict, 
                    model_name: str,
                    model_config: dict, 
                    preprocess: ColumnTransformer=None
                    ):
    # setup logger
    logger = session.logger
    logger.info("Start ML Training '%s')", 
                model_name)

    # extract data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"] 

    # calculate pos_weight
    weighted = model_config.get("weighted", None)
    pos_weight = None
    if weighted:
        pos = y_train.sum()
        neg = len(y_train) - pos

        pos_weight = float(neg / pos)

    # prepare pipeline
    model_fn = model_dict[model_name]
    model = model_fn(model_config, pos_weight)
    
    if preprocess:
        model_pipe = Pipeline(
            [("preprocess", preprocess), ("model", model)]
        )
        logger.info("Created pipeline (preprocessing + %s model)", 
                    model_name)
    
    else: 
        model_pipe = model
        logger.info("Created pipeline (%s model only)",
                    model_name)

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

    logger.info("Completed %s training and prediction.", 
                model_name)

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

    return model_pipe


if __name__ == "__main__":
    predict_by_ml_model()

    