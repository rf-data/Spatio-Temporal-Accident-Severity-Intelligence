## evaluate_stage_1.py
# imports
import click
import os
from pathlib import Path

import src.utils.general_helper as gh
# import src.utils.evaluation_helper as eval

from src.core.session import session
import src.core.ml_manager as man
from src.core.logger import create_logger 

from src.utils.file_helper import get_yaml_config


# ------------------
# MAIN FUNCTION
# ------------------

@click.command()
@click.option("--config_name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def stage_1_evaluation(config_name):  
    
    run_stage_1_evaluation(config_name)

    return


def run_stage_1_evaluation(config_name):
    # load env variables
    gh.load_env_vars()
    # data_evaluated = os.getenv("PATH_EVALUATED")

    # load values from config
    config = get_yaml_config(config_name)
    general_config = config.get("general_args", {})
    forecast_models = general_config["forecast_models"]
    log_name = general_config["name_log_eval"]
    name_logfile = general_config["name_logfile_eval"]
    timestamp = general_config["timestamp"]

    # --- create logger --
    logger = create_logger(name=log_name, file_name=name_logfile)
    session.logger = logger

    # --- create evaluation manager --
    eval_manager = man.EvaluationManager(timestamp)

    eval_manager.compare_roc_pr_curves(forecast_models)

    describe_metrics = []
    # create_classification_report(y_true, y_pred)
    # create_confusion_matrix(y_true, y_pred)
    # create_metrics(y_true, y_pred)
    # 

    thresh = []
    errors = []
    descript = []
    for model in forecast_models:
        eval_manager.compare_roc_pr_curves(model)

        thresh_result = eval_manager.optimize_threshold(
                                                    model, 
                                                    "f2", 
                                                    beta=2
                                                    )
        thresh_result["name"] = model
        thresh.append(thresh_result)

        desc_results = eval_manager.describe_run(
                                model, 
                                describe_metrics
                                )
        descript.append(desc_results)

        error_df = eval_manager.get_errors(model)
        errors.append(error_df)

    # store importance objects/values in session 
    # eval_folder = os.getenv("PATH_EVALUATED")
    # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
