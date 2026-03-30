## predict_by_logreg.py
# imports
# import os
# from pathlib import Path
from datetime import datetime
# from typing import Callable
# from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

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

def predict_by_catboost(
                    data: dict, 
                    cat_config, 
                    preprocess: ColumnTransformer=None
                    ):
    # setup logger
    logger = session.logger
    logger.info("Start Training 'XGBoost'")

    # extract data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"] 

    # calculate pos_weight
    # pos = y_train.sum()
    # neg = len(y_train) - pos

    # pos_weight = float(neg / pos)

    # prepare pipeline
    model = (cat_config)

    if preprocess:
        model_pipe = Pipeline(
            [("preprocess", preprocess), ("model", model)]
        )
        logger.info("Created pipeline (preprocessing + CatBoost_model)")
    
    else: 
        model_pipe = model
        logger.info("Created pipeline (CatBoost_model only)")

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
    
    logger.info("Completed XGBoost training and prediction.")

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
    predict_by_catboost()
