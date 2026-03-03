## visualisation_helper.py
# imports
from contextlib import contextmanager
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import seaborn as sns
import numpy as np


# === Display helpers ===
def show_overview(df, name):
    print(f"\n{'='*30}\n{name}\n{'='*30}")
    df.info()
    print("\nFirst rows:\n", df.head())


@contextmanager
def style_format_context(df, fmt_dict):
    """Context manager for temporary styled display (Jupyter)."""
    try:
        styled_df = df.style.format(fmt_dict)
        yield styled_df
    finally:
        pass


def format_and_display(df, fmt_dict):
    display(df.style.format(fmt_dict))


def create_geo_scatterplot(sample: pd.DataFrame):
    plot_path = os.getenv("PATH_PLOT")
    # "/home/robfra/0_Portfolio_Projekte/Road_accidents/data/plots/eda/scatter_geodata_norm.png"

    plt.scatter(sample["lon_norm"], sample["lat_norm"], s=1, alpha=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Accident locations (sample)")
    plt.savefig(plot_path)
    plt.show()

    return


def create_roc_auc(y_test, y_proba, roc_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – LogReg (baseline)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(roc_path, dpi=150)
    plt.close()

    return


def create_pr_curve(y_test, y_proba, pr_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – LogReg (baseline)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(pr_path, dpi=150)
    plt.close()

    return


def viz_odds_ratios(coef_df, top_k=5, save_path=None):

    classes = coef_df["class"].unique()

    for cls in classes:

        df_cls = coef_df[coef_df["class"] == cls].copy()

        # stärkste Effekte (weg von 1)
        df_cls["abs_log_or"] = np.abs(np.log(df_cls["odds_ratio"]))
        df_plot = df_cls.sort_values("abs_log_or", ascending=False).head(top_k)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x="odds_ratio", y="feature")

        plt.axvline(1, linestyle="--", color="black")
        plt.title(f"Top {top_k} Odds Ratios – Class: {cls}")
        plt.xlabel("Odds Ratio")
        plt.ylabel("")
        plt.tight_layout()

        if save_path:
            coef_path = ph.create_save_path("plots", f"coef_{cls}", "png")
            plt.savefig(save_path, dpi=150)

        plt.show()


def viz_odds_heatmap(coef_df, save_path=None):

    pivot = coef_df.pivot(index="feature", columns="class", values="coef")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, center=0, cmap="coolwarm")
    plt.title("Coefficient Heatmap (log-odds)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def viz_permutation_importance(perm_df, top_k=10, save_path=None):

    df_plot = perm_df.sort_values("importance_mean", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        df_plot["feature"],
        df_plot["importance_mean"],
        xerr=df_plot["importance_std"],
        align="center",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Decrease in Score")
    ax.set_title(f"Top {top_k} Permutation Importances")
    # plt.ylabel("")
    plt.tight_layout()

    if save_path:
        perm_path = ph.create_save_path("plots", f"PermImp", "png")
        plt.savefig(perm_path, dpi=300)

    plt.show()
