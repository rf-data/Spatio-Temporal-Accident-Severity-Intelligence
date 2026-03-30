## predict_by_logreg.py
# imports
# import os
# from pathlib import Path
from datetime import datetime
# from typing import Callable
# from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
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

def build_light_gbm_model(light_config, pos_weight):

    n_estimators = int(light_config["n_estimators"])   # , None)
    learning_rate = float(light_config["learning_rate"])    # , 0.0)
    max_depth = int(light_config["max_depth"])   # , "lbfgs")
    num_leaves = int(light_config["num_leaves"])   # , 42)
    n_jobs = int(light_config["n_jobs"])     # , None)
    random_state = light_config["random_state"]

    model = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    scale_pos_weight=pos_weight, 
                    n_jobs=n_jobs,
                    random_state=random_state
                    )

    return model

# ------------------
# MAIN FUNCTION
# ------------------

def predict_by_light_gbm(
                    data: dict, 
                    light_config, 
                    preprocess: ColumnTransformer=None
                    ):
    # setup logger
    logger = session.logger
    logger.info("Start Training 'Light GBM'")

    # extract data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"] 

    # calculate pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos

    pos_weight = float(neg / pos)

    # prepare pipeline
    model = build_light_gbm_model(light_config, pos_weight)

    if preprocess:
        model_pipe = Pipeline(
            [("preprocess", preprocess), ("model", model)]
        )
        logger.info("Created pipeline (preprocessing + light_gbm_model)")
    
    else: 
        model_pipe = model
        logger.info("Created pipeline (light_gbm_model only)")

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

    logger.info("Completed Light_GBM training and prediction.")

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

    eval.create_tree_importance_df(model_pipe, save=True)

    return



if __name__ == "__main__":
    predict_by_light_gbm()
