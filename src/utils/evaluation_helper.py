##
# imports
# import json
import pandas as pd
import numpy as np
from pathlib import Path
import os

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    #  precision_score,
    #  recall_score,
    #  f1_score,
    fbeta_score,
)

from src.core.session import session
import src.utils.path_helper as ph
import src.utils.visualisation_helper as viz


def create_classification_report(y_true, y_pred):
    # setup logger
    # exp_logger = get_experiment_logger()
    logger = session.logger
    run_name = session.gen_params.get("log_file", None)

    # creation of ClassReport
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["no_accident", "accident"],
        zero_division=0,
        output_dict=True,
    )

    logger.info("Created Classification Report (run=%s)", run_name)
    # log ClassReport
    # exp_logger.log_text(
    #             "ClassReport.json",
    #             json.dumps(report, indent=2)
    #         )

    # exp_logger.log_metric("precision_escalation",
    #                   report["escalation"]["precision"])

    # exp_logger.log_metric("recall_escalation",
    #                   report["escalation"]["recall"])

    return report


def create_confusion_matrix(y_true, y_pred):
    # setup logger
    # exp_logger = get_experiment_logger()
    logger = session.logger
    run_name = session.gen_params.get("log_file", None)

    # create ConfMatrix
    tn, fp, fn, tp = confusion_matrix(  # tn, fp, fn, tp
        y_true, y_pred, labels=[False, True]
    ).ravel()

    all = np.sum([tn, fp, fn, tp])
    logger.info(
        "Confusion Matrix created (run=%s):\ntn=%s \nfn=%s \ntp=%s \nfp=%s",
        run_name,
        (tn, np.round(tn / all, 3)),
        (fn, np.round(fn / all, 3)),
        (tp, np.round(tp / all, 3)),
        (fp, np.round(fp / all, 3)),
    )
    # logger.info("false_positives: %s", fp)

    cm = {
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    # log values from ConfMatrix
    # exp_logger.log_text("ConfMatrix.json", json.dumps(cm, indent=2))
    # exp_logger.log_metric("true_negatives", tn)
    # exp_logger.log_metric("true_positives", tp)
    # exp_logger.log_metric("false_negatives", fn)
    # exp_logger.log_metric("false_positives", fp)

    return cm


def create_metrics(y_true, y_pred):
    # setup logger
    # exp_logger = get_experiment_logger()
    logger = session.logger
    run_name = session.gen_params.get("log_file", None)

    # compile Classification metrics
    precision, recall, f1, _ = (
        precision_recall_fscore_support(  # precision, recall, f1, support
            y_true, y_pred, average="binary", pos_label=True, zero_division=0
        )
    )

    f2 = fbeta_score(
        y_true, y_pred, beta=2, average="binary", pos_label=1, zero_division=0
    )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
    }

    logger.info(
        "Metrics compiled (run=%s).\nprecision=%.4f | recall=%.4f | f1=%.4f | f2=%.4f",
        run_name,
        precision,
        recall,
        f1,
        f2,
    )

    # log ClassMetrics
    # exp_logger.log_metric("precision", precision)
    # exp_logger.log_metric("recall", recall)
    # exp_logger.log_metric("f1", f1)
    # exp_logger.log_metric("f2", f2)

    return metrics


def compile_roc_pr_auc(y_test, y_proba, data_viz=False, suffix=False):
    # setup logger
    # exp_logger = get_experiment_logger()
    logger = session.logger

    roc_auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else None
    pr_auc = average_precision_score(y_test, y_proba)

    # logging
    # exp_logger.log_metric("ROC-AUC", roc_auc)
    # exp_logger.log_metric("Precision-Recall-AUC", pr_auc)

    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("PR-AUC: %.4f", pr_auc)

    if data_viz == True:

        add = f"_{suffix}" if suffix else ""

        pr_path = ph.create_save_path("plots", f"pr_curve{add}", "png")
        roc_path = ph.create_save_path("plots", f"roc_auc{add}", "png")

        viz.create_roc_auc(y_test, y_proba, roc_path)
        viz.create_pr_curve(y_test, y_proba, pr_path)

        # exp_logger.log_artifact(roc_path)
        # exp_logger.log_artifact(pr_path)

    return {"ROC_AUC": roc_auc, "PR_AUC": pr_auc}


def importance_by_permutation(pipe, data_dict, data_viz=False):
    # setup logger
    logger = session.logger

    n_perm = session.exp_params.get("n_perm", 5)
    random_state = session.exp_params.get("random_state", 42)
    scoring = session.exp_params.get("perm_score", None)

    #
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    # feats = pipe.named_steps["preprocess"].get_feature_names_out()
    feats = X_test.columns

    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=n_perm,
        random_state=random_state,
        scoring=scoring,
    )

    # print("n_feats:", len(feats))
    # print("n_perm:", len(r.importances_mean))

    perm_df = pd.DataFrame(
        {
            "feature": feats,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    logger.info("Created PermImportance_df.")
    logger.info(
        "'PermImportance_df' - sorted:\n%s", perm_df.sort_values("importance_mean")
    )

    if data_viz:
        viz.viz_permutation_importance(perm_df, save_path=True)

    # save perm_df
    df_path = ph.create_save_path("PermImport_df", "perm", "csv")
    perm_df.to_csv(df_path, index=False)
    logger.info("Saved permutation_importance_df in ...%s", ph.shorten_path(df_path))

    return perm_df


def create_coef_df(pipe, save=None, data_viz=None):
    # setup logger
    logger = session.logger

    # extract parameter
    model = pipe.named_steps["model"]
    classes = model.classes_
    coefs = model.coef_  # shape: (n_classes, n_features)
    # ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]

    coef_list = []
    feats = pipe.named_steps["preprocess"].get_feature_names_out()

    for i, cls in enumerate(classes):
        df_tmp = pd.DataFrame({"feature": feats, "coef": coefs[i], "class": cls})
        coef_list.append(df_tmp)

        logger.info("Created Coef_df (class=%s)", cls)

    coef_df = pd.concat(coef_list).sort_values(
        ["class", "coef"], ascending=[True, False]
    )

    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["importance"] = np.abs(coef_df["coef"])

    logger.info("Merged all coef_dfs, and added cols 'odds_ratio' and 'importance'.")
    # logger.info("'coef_df' - top5 feature_imortance per class:\n%s",
    #             coef_df\
    #             .sort_values("odds_ratio", ascending=False)\
    #             .groupby("class")
    #             #  .head(5))
    #              )

    if data_viz:
        coef_hm_path = ph.create_save_path("plots", "coef_heat", "png")

        viz.viz_odds_ratios(coef_df, top_k=5, save_path=True)
        viz.viz_odds_heatmap(coef_df, save_path=coef_hm_path)

    if save:
        df_path = ph.create_save_path("Coef_df", "coefs", "csv")
        coef_df.to_csv(df_path, index=False)
        logger.info("Saved coef_df in ...%s", ph.shorten_path(df_path))

    return coef_df

    # elif model:
    #     hi = ""
    #     coefs = model.coef_[0]

    # else:
    #     logger.info("Please, provide a trained model or a pipeline containing a model training")
    #     return

    # num_feats = session.parameters.get("num_feats")
    # cat_feats = session.parameters.get("cat_feats")

    # cat_feat_names = ohe.get_feature_names_out(cat_feats)
    # feat_names = (num_feats +
    #               list(cat_feat_names))
    # feats = df
