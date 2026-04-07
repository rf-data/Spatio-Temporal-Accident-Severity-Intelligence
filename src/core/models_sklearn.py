

# imports
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.core.ml_models import BaseModelAdapter

MODEL_DICT = {
    "xgboost": build_xgboost_model,
    "catboost": build_catboost_model,
    "light_gbm": build_light_gbm_model,
    "log_reg": build_logreg_model  
    }

class SklearnModelAdapter(BaseModelAdapter):

    def build(self, model_name, model_config):
        build_fn = MODEL_DICT[model_name] 

        return build_fn(**model_config)

    # def train(self, model, X_train, y_train):
    #     model.fit(X_train, y_train)
    #     return model

    # def predict(self, model, X_test):
    #     return model.predict(X_test)

    


# -----------------
# ML functions
# -----------------

def build_logreg_model(lr_config):      # , pos_weight=None

    class_weight = lr_config["class_weight"]   # , None)
    l1_ratio = float(lr_config["l1_ratio"])    # , 0.0)
    solver = lr_config["solver"]   # , "lbfgs")
    random_state = int(lr_config["random_state"])   # , 42)
    max_iter = int(lr_config["max_iter"])   # , 1000)
    param_C = float(lr_config["C"])     # , None)

    lr_model = LogisticRegression(
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        C=param_C,
        # penalty=penalty,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
    )

    return lr_model

# ----------------------------------------
# 'BOOSTED' TREE MODELS (CLASSIFICATION)
# ----------------------------------------

def build_catboost_model(cat_config):       # , pos_weight=None

    iterations = int(cat_config["iterations"])  
    learning_rate = float(cat_config["learning_rate"]) 
    early_stop = int(cat_config["early_stop"]) 
    depth = int(cat_config["depth"])  
    l2_leaf_reg = int(cat_config["l2_leaf_reg"])
    loss_fn = cat_config["loss_fn"]
    auto_class_weights = cat_config["auto_class_weights"]
    eval_metric = cat_config["eval_metric"]
    random_state = int(cat_config["random_state"])
    verbose = int(cat_config["verbose"])

    cat_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stop,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_fn,
        auto_class_weights=auto_class_weights,
        eval_metric=eval_metric,random_seed=random_state,
        verbose=verbose
    )

    return cat_model


def build_xgboost_model(xgb_config):        # , pos_weight=None

    n_estimators = int(xgb_config["n_estimators"])   # , None)
    learning_rate = float(xgb_config["learning_rate"])    # , 0.0)
    max_depth = int(xgb_config["max_depth"])   # , "lbfgs")
    min_child_weight = int(xgb_config["min_child_weight"])
    subsample = float(xgb_config["learning_rate"])
    colsample_bytree = float(xgb_config["colsample_bytree"])
    gamma = float(xgb_config["gamma"])
    reg_alpha = float(xgb_config["reg_alpha"])
    reg_lambda = float(xgb_config["reg_lambda"])
    objective = xgb_config["objective"]
    eval_metric = xgb_config["eval_metric"]
    n_jobs = int(xgb_config["n_jobs"])     # , None)
    random_state = int(xgb_config["random_state"])

    xgb_model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                objective=objective,
                eval_metric=eval_metric,
                random_state=random_state,
                n_jobs=n_jobs
                )

    return xgb_model


def build_light_gbm_model(light_config):        # , pos_weight

    n_estimators = int(light_config["n_estimators"])   # , None)
    learning_rate = float(light_config["learning_rate"])    # , 0.0)
    max_depth = int(light_config["max_depth"])   # , "lbfgs")
    num_leaves = int(light_config["num_leaves"])   # , 42)
    n_jobs = int(light_config["n_jobs"])     # , None)
    random_state = light_config["random_state"]

    light_model = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    # scale_pos_weight=pos_weight, 
                    n_jobs=n_jobs,
                    random_state=random_state
                    )

    return light_model

