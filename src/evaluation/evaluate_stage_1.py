## evaluate_stage_1.py
# imports
import click
import os
from pathlib import Path
import pandas as pd

import src.utils.general_helper as gh
import src.utils.df_helper as dfh
import src.utils.file_helper as fh
import src.utils.evaluation_helper as eval

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

    # load values from config
    config = get_yaml_config(config_name)
    general_config = config.get("general_args", {})
    forecast_models = general_config["forecast_models"]
    log_name = general_config["name_log_eval"]
    name_logfile = general_config["name_logfile_eval"]
    resolution = general_config["h3_col"]
    period = general_config["period"]

    eval_config = config.get("evaluation", {})
    timestamp = eval_config["timestamp"]
    data_eval = os.getenv("PATH_EVALUATED")
    save_folder = f"{data_eval}/{timestamp}"

    # --- create logger --
    logger = create_logger(name=log_name, file_name=name_logfile)
    session.logger = logger

    # --- create evaluation manager --
    eval_manager = man.EvaluationManager(timestamp)

    # eval_manager.compare_roc_pr_curves(forecast_models)

    describe_metrics = [eval.create_confusion_matrix,
                        eval.create_metrics]
    # (y_true, y_pred)
    # (y_true, y_pred)
    # (y_true, y_pred)
    # 

    thresh = []
    errors = []
    descript = []
    all_reports = {}
    for model in forecast_models:
        logger.info("Start evaluation of '%s' results", model)
        session.model_class = model

        eval_manager.compare_roc_pr_curves(model)

        thresh_result = eval_manager.optimize_threshold(
                                                    model, 
                                                    "f2", 
                                                    beta=2
                                                    )
        thresh_result["name"] = model
        thresh_result["time"] = timestamp
        thresh_result["period"] = period
        thresh_result["resolution"] = resolution

        thresh.append(thresh_result)        # List[dict]

        model_scores, class_report = eval_manager.describe_run(
                                model, 
                                describe_metrics,
                                eval.create_classification_report
                                )
        
        # model_scores.update({
        #                 "name": model,
        #                 "time": timestamp,
        #                 "period": period,
        #                 "resolution": resolution
        #                 })
        desc_df = pd.DataFrame([model_scores])
        desc_df[["name", 
                 "time", 
                 "period", 
                 "resolution"]] = (model,
                                timestamp,
                                period, 
                                resolution)

        descript.append(desc_df)      # List[dict]
        all_reports[model] = class_report

        error_df = eval_manager.get_errors(model)
        error_df["model"] = model
        error_df["time"] = timestamp
        error_df["period"] = period
        error_df["resolution"] = resolution

        errors.append(error_df)       # List[pd.DataFrame]

    # save results as DataFrame / parquet
    thresh_df = pd.concat(thresh)

    dfh.save_df_to_parquet(thresh_df, 
                           f"{timestamp}_thresh_df", 
                           save_folder, 
                           chunked=True)
    logger.info("Head of thresh_df (shape: %s):\n%s\n", 
                thresh_df.shape,
                thresh_df.head(5))
    
    descript_df = pd.concat(descript, 
                            axis=0, 
                            ignore_index=True)
    dfh.save_df_to_parquet(descript_df, 
                           f"{timestamp}_descript_df", 
                           save_folder, 
                           chunked=True)
    logger.info("Head of descript_df (shape: %s):\n%s\n", 
                descript_df.shape,
                descript_df.head(5).T)
    
    errors_df = pd.concat(errors)
    dfh.save_df_to_parquet(error_df, 
                           f"{timestamp}_error_df", 
                           save_folder, 
                           chunked=True)
    logger.info("Head of errors_df (shape: %s):\n%s\n", 
                errors_df.shape,
                errors_df.head(5))
    
    report_path = f"{save_folder}/{timestamp}_class_reports.json"
    fh.save_dict(all_reports, report_path)
    logger.info("'all_reports':\n%s'", 
                all_reports)
    # store importance objects/values in session 
    # eval_folder = os.getenv("PATH_EVALUATED")
    # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    stage_1_evaluation()

# forecast_stage_1