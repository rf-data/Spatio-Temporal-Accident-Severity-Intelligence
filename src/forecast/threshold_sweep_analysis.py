##
# imports
# import numpy as np
# import pandas as pd
# from sklearn.base import clone
# from sklearn.metrics import (
#                              precision_recall_fscore_support,
#                              classification_report,
#                              confusion_matrix,
#                              roc_auc_score,
#                              average_precision_score,
#                              precision_score,
#                              recall_score,
#                              f1_score,
#                              fbeta_score
#                              )

# from core.mlflow_logger import get_experiment_logger
import src.utils.evaluation_helper as eval
import src.utils.path_helper as ph
import src.utils.file_helper as fh
import src.utils.thresh_sweep_helper as thresh

# import src.utils.split_helper as split

from src.core.session import session

#

# imports
# from sklearn.model_selection import (train_test_split,
#                                      GroupKFold,
#                                      GroupShuffleSplit)

# from core.mlflow_logger import get_experiment_logger
# import utils.preprocess_helper as pre

# def validation_split(model, X_train, y_train):
#     # random_state=session.parameters.get("random_state", 42)

#     X_train2, X_val, y_train2, y_val = train_test_split(
#                                             X_train,
#                                             y_train,
#                                             test_size=0.2,
#                                             random_state=random_state,
#                                             stratify=y_train
#                                                 )
#     # train model on validation dataset
#     model.fit(X_train2, y_train2)
#     y_proba_val = model.predict_proba(X_val)[:, 1]

#     return y_val, y_proba_val

#


def threshold_sweep_analysis(predict_dict, metric):
    # setup logger
    # exp_logger = get_experiment_logger()
    logger = session.logger

    #
    # X_train = predict_dict.get("X_train", None)
    # X_test = predict_dict.get("X_test", None)
    y_true = predict_dict.get("y_test", None)
    # y_train = predict_dict.get("y_train", None)
    # y_pred = predict_dict.get("y_pred", None)
    y_proba = predict_dict.get("y_proba", None)
    y_val = predict_dict.get("y_val", None)
    y_proba_val = predict_dict.get("y_proba_val", None)

    #
    # y_val, y_proba_val = validation_split(model, X_train, y_train)
    thresh_df = thresh.create_threshold_df(y_val, y_proba_val)
    best_row = thresh.find_optimal_thresh(thresh_df, metric)
    best_t = float(best_row["threshold"])

    thresh_path = ph.create_save_path("thresh_df", "thresh_df", "csv")
    thresh_df.to_csv(thresh_path)

    # final eval on test (fixed threshold!)
    # y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_t).astype(int)

    report = eval.create_classification_report(y_true, y_pred)
    cm = eval.create_confusion_matrix(y_true, y_pred)
    metrics = eval.create_metrics(y_true, y_pred)

    # create ROC_AUC and PR_AUC
    eval.compile_roc_pr_auc(y_true, y_proba, data_viz=True, suffix="thresh")

    # save files locally
    path_cm = ph.create_save_path("ConfMatrix", "cm_thresh", "json")
    path_cr = ph.create_save_path("ClassReport", "cr_thresh", "json")
    path_metrics = ph.create_save_path("metrics", "metrics_thresh", "json")
    # path_fn_fp = Path(f"{folder}/fn_fp") # /{now}_{mode}_{version_run}_metrics.json")

    for path, file in zip([path_cm, path_cr, path_metrics], [cm, report, metrics]):
        if not path.exists():
            fh.save_dict(path, file)
        else:
            logger.error("File '%s' already exists. Hence, no overwrite", path)

    return


"""
BEFORE
 Confusion Matrix created (run=LogReg_base_p1):
tn=(np.int64(217620), np.float64(0.43)) 
fn=(np.int64(25340), np.float64(0.05)) 
tp=(np.int64(114246), np.float64(0.226)) 
fp=(np.int64(148836), np.float64(0.294))
2026-02-20 15:08:03,350 [INFO] LOGREG_BASELINE_PHASE_1: Metrics compiled (run=LogReg_base_p1).
precision=0.4343 | recall=0.8185 | f1=0.5674 | f2=0.6954
2026-02-20 15:08:04,264 [INFO] LOGREG_BASELINE_PHASE_1: ROC-AUC: 0.7945
2026-02-20 15:08:04,275 [INFO] LOGREG_BASELINE_PHASE_1: PR-AUC: 0.6227

AFTER
 Metrics compiled (run=LogReg_base_p1).
precision=0.3525 | recall=0.9474 | f1=0.5138 | f2=0.7083
2026-02-20 15:08:12,554 [INFO] LOGREG_BASELINE_PHASE_1: ROC-AUC: 0.7945
2026-02-20 15:08:12,555 [INFO] LOGREG_BASELINE_PHASE_1: PR-AUC: 0.6227
Variables from .env.session loaded
"""
