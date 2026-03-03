##
# imports
import os
from pathlib import Path

import src.utils.evaluation_helper as eval
import src.utils.general_helper as gh
import src.utils.file_helper as fh
import src.utils.path_helper as ph
from src.core.session import session


def phase_1_evaluation(predict_dict):
    # setup logger
    logger = session.logger
    # now = session.exp_params.get("now", None)
    run_name = session.exp_params.get("log_file", None)

    # create ClassReport, ConfMatrix, ClassMetrics
    y_true = predict_dict.get("y_test", None)
    y_pred = predict_dict.get("y_pred", None)
    y_proba = predict_dict.get("y_proba", None)

    report = eval.create_classification_report(y_true, y_pred)
    cm = eval.create_confusion_matrix(y_true, y_pred)
    metrics = eval.create_metrics(y_true, y_pred)

    # create ROC_AUC and PR_AUC
    eval.compile_roc_pr_auc(y_true, y_proba, data_viz=True)

    # save files locally
    path_cm = ph.create_save_path("ConfMatrix", "cm", "json")
    path_cr = ph.create_save_path("ClassReport", "cr", "json")
    path_metrics = ph.create_save_path("metrics", "metrics", "json")
    # path_fn_fp = Path(f"{folder}/fn_fp") # /{now}_{mode}_{version_run}_metrics.json")

    for path, file in zip([path_cm, path_cr, path_metrics], [cm, report, metrics]):
        if not path.exists():
            fh.save_dict(path, file)
        else:
            logger.error("File '%s' already exists. Hence, no overwrite", path)

    return
