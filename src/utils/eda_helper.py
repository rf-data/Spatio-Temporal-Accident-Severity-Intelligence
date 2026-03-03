# --- eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# === Feature description ===
def describe_features_stats(
    df,
    num_feats=None,
    cat_feats=None,
    # na_threshold=0.5,
    # min_std=1e-6,
    verbose=True,
):
    """Computes descriptive statistics for numerical and categorical features."""
    if num_feats is None:
        num_feats = df.select_dtypes(include=[np.number]).columns.tolist()

    if cat_feats is None:
        cat_feats = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if verbose:
        print(f"📊 Numeric: {num_feats}\n🔤 Categorical: {cat_feats}")

    df_num = df[num_feats]
    # too_many_nans = (df_num.isna().mean() > na_threshold) if na_threshold is not None else False
    # low_variance = (df_num.std(skipna=True) < min_std) if min_std is not None else False
    # to_drop = too_many_nans | low_variance
    # df_num_filtered = df_num.loc[:, ~to_drop]

    # if verbose and to_drop.any():
    #     print(f"🧹 Dropped numeric features: {to_drop[to_drop].index.tolist()}")

    statistics = {
        "count_clean": lambda x: x.notna().sum(),
        "NaN_count": lambda x: x.isna().sum(),
        "min": np.nanmin,
        "q01": lambda x: x.quantile(0.01),
        "q05": lambda x: x.quantile(0.05),
        "q10": lambda x: x.quantile(0.10),
        "q25": lambda x: x.quantile(0.25),
        "mean": np.nanmean,
        "median": np.nanmedian,
        "q75": lambda x: x.quantile(0.75),
        "q90": lambda x: x.quantile(0.90),
        "q95": lambda x: x.quantile(0.95),
        "q99": lambda x: x.quantile(0.99),
        "max": np.nanmax,
        "std": np.nanstd,
        "skewness": lambda x: stats.skew(x.dropna()),
        "kurtosis": lambda x: stats.kurtosis(x.dropna()),
    }

    stats_num_dict = {
        stat: [func(df_num[col]) for col in df_num.columns]
        for stat, func in statistics.items()
    }

    stats_num = pd.DataFrame(stats_num_dict, index=df_num.columns)

    stats_cat = pd.DataFrame(
        index=cat_feats, columns=["n_unique", "NaN_count", "top", "top_freq"]
    )
    for col in cat_feats:
        stats_cat.loc[col, "n_unique"] = df[col].nunique(dropna=True)
        stats_cat.loc[col, "NaN_count"] = df[col].isna().sum()
        top_val = df[col].mode(dropna=True)
        if not top_val.empty:
            stats_cat.loc[col, "top"] = top_val[0]
            stats_cat.loc[col, "top_freq"] = df[col].value_counts(dropna=True).iloc[0]

    return stats_num, stats_cat


# === Feature visualisation ===
def visualize_features(num_data=None, cat_data=None):
    """Plots boxplots for numeric and barplots for categorical data."""
    if num_data is not None and isinstance(num_data, pd.DataFrame):
        num_feat = num_data.columns.tolist()
        num_data.plot(
            kind="box",
            subplots=True,
            layout=(int(np.ceil(len(num_feat) / 3)), 3),
            figsize=(15, 3 * int(np.ceil(len(num_feat) / 3))),
            title="Boxplots of Numeric Features",
        )
        plt.tight_layout()
        plt.show()

    if cat_data is not None and isinstance(cat_data, pd.DataFrame):
        for col in cat_data.columns:
            plt.figure(figsize=(8, 4))
            cat_data[col].value_counts(dropna=False).plot(kind="bar")
            plt.title(f"Categorical Distribution: {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def visualize_features_grouped(num_data=None, cat_data=None, suffixes=None):
    """
    Plots grouped numeric boxplots (e.g. _F, _M, _T suffixes) and categorical barplots.
    """
    if suffixes is None:
        suffixes = ["_F", "_M", "_T"]

    if num_data is not None and isinstance(num_data, pd.DataFrame):
        feature_groups = {}
        for col in num_data.columns:
            base = col
            for suffix in suffixes:
                if col.endswith(suffix):
                    base = col[:-2]
            feature_groups.setdefault(base, []).append(col)

        num_groups = len(feature_groups)
        ncols = 2
        nrows = int(np.ceil(num_groups / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten() if nrows > 1 else [axes]

        for ax, (base, cols) in zip(axes, feature_groups.items()):
            num_data[cols].plot(kind="box", ax=ax)
            ax.set_title(f"{base} ({'Grouped' if len(cols) > 1 else 'Single'})")

        for ax in axes[num_groups:]:
            ax.axis("off")

        plt.suptitle("Boxplots of Grouped Numeric Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    if cat_data is not None and isinstance(cat_data, pd.DataFrame):
        for col in cat_data.columns:
            plt.figure(figsize=(8, 4))
            cat_data[col].value_counts(dropna=False).plot(kind="bar")
            plt.title(f"Categorical Distribution: {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
