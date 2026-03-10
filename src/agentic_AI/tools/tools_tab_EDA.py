## tools_tab_EDA.py
# imports
import pandas as pd
import numpy as np

from src.core.tools_classes import tools_registry, add_tool, Observation
from src.agentic_AI.rules.eda_rules import recommend_from_distribution

##############################
# CONFIGURATION
##############################
KNOWN_FORMATS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S"
]

##############################
# GENERAL
#############################
@add_tool(tools_registry, 
      description="Create basic overview on data: shape, column names + dtypes",
      category="tabular",
      eda=True,
      default=True,
      cross_file=False
      )
def basic_overview(df: pd.DataFrame, config: dict | None = None):

    return Observation(
                tool_name="basic_overview",
                category="raw_tab_file_EDA",
                column="all",
                description="Create basic overview on data: shape, column names + dtypes",
                metrics={
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict()
                    },
                recommendation_hint={}
                )


@add_tool(tools_registry, 
      description="Evaluates column-wise if there any NaN values.",
      category="raw_tab_file_EDA",
      eda=True,
      default=True,
      cross_file=False)
def missing_analysis(df: pd.DataFrame, config: dict | None = None):
    threshold = config["missing_analysis"].get("threshold", 0.5)
    
    recs = {}
    for col in df.columns:
        nan_ratio = df[col].isna().mean() 
        if nan_ratio > threshold:
            recs[col] = nan_ratio
    
    
    hint= {"missing": recs} if recs else {}
        # recomm = f"""
        # There are columns having a nan_ratio above the allowed threshold ({threshold}):\n
        # {recs.keys()}
        # """

    return Observation(
                tool_name="missing_analysis",
                category="raw_tab_file_EDA",
                column="all",
                description="Evaluates column-wise if there any NaN values",
                metrics={
                    "nan_count": (df.isna()
                                  .sum()
                                  .sort_values(ascending=False)
                                  .to_dict()),
                    "nan_ratio": (df.isna()
                                  .mean()
                                  .sort_values(ascending=False)
                                  .to_dict())
                    },
                recommendation_hint=hint
                )


@add_tool(tools_registry, 
      description="Evaluates column-wise if there any np.inf values.",
      category="raw_tab_file_EDA",
      eda=True,
      default=True,
      cross_file=False)
def infinite_analysis(df: pd.DataFrame, config: dict | None = None):

    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        return Observation(
            tool_name="infinite_analysis",
            category="raw_tab_file_EDA",
            column="numeric",
            description="No numeric columns found.",
            metrics={"has_inf": False},
            recommendation_hint=None
        )
    
    has_inf = np.isinf(num_df).values.any()

    recs = {}
    for col in num_df.columns:
        inf_count = np.isinf(num_df[col]).values.sum()

        if inf_count > 0:
            recs[col] = int(inf_count)

    hint = {"infinite": recs} if recs else {}
    
    return Observation(
                tool_name="infinite_analysis",
                category="raw_tab_file_EDA",
                column="numeric",
                description="Evaluates numeric column if there any np.inf values.",
                metrics={
                    "has_inf": bool(has_inf),
                    "inf_count": recs
                    },
                recommendation_hint=hint
                )


@add_tool(tools_registry, 
      description="Checks if there any duplicates.",
      category="raw_tab_file_EDA",
      eda=True,
      default=True,
      cross_file=False)
def duplicate_analysis(df: pd.DataFrame, config: dict | None = None):
    threshold = config["duplicate_analysis"].get("threshold", None)
    grouping = config["duplicate_analysis"].get("grouping", [])

    # 
    n_dups = df.duplicated().sum()
    ratio_dups = df.duplicated().mean()

    n_dups_group = "not conducted / calculated" 
    ratio_dups_group = "not conducted / calculated" 

    if grouping:
        print("grouping:", grouping)
        n_dups_group = df.groupby(grouping).duplicated().sum()
        ratio_dups_group = df.groupby(grouping).duplicated().mean()

    recommendation = {}
    if ratio_dups > threshold: 
        recommendation["duplicate_ratio (general)"] = ratio_dups

    if isinstance(ratio_dups_group, (float, int)) and ratio_dups_group > threshold:
        recommendation["duplicate_ratio (grouped)"] = ratio_dups_group

    hint = {"duplicates": recommendation} if recommendation else {}

    return Observation(
                tool_name="duplicate_analysis",
                category="raw_tab_file_EDA",
                column="all",
                description="Checks if there any duplicates.",
                metrics={
                    "n_duplicates (general)": n_dups,
                    "duplicate_ratio (general)": ratio_dups,
                    "grouping": grouping,
                    "n_duplicates (grouped)": n_dups_group,
                    "duplicate_ratio (grouped)": ratio_dups_group
                    },
                recommendation_hint=hint
                )


@add_tool(tools_registry, 
      description="Compiles a descriptive statistics on numeric columns.",
      category="raw_tab_file_EDA", 
      eda=True,
      default=True,
      cross_file=False)
def numeric_summary(df: pd.DataFrame, config: dict | None = None):
    #
    perc = config["numeric_summary"].get("percentiles", [0.25, 0.5, 0.75])
    
    #
    numeric_df = df.select_dtypes(include=np.number)
    describe_dict = numeric_df.describe(percentiles=perc).to_dict()

    skew_col = {}
    kurt_col = {}
    for col in numeric_df.columns:
        skew_col[col] = numeric_df[col].skew(numeric_only=True, skipna=True)
        kurt_col[col] = numeric_df[col].kurtosis(numeric_only=True, skipna=True)

    recs = recommend_from_distribution(skew_col, 
                                       kurt_col, 
                                       config)

    hint = {"Skew_Kurt": recs} if recs else {}

    return Observation(
                tool_name="numeric_summary",
                category="raw_tab_file_EDA",
                column="numeric columns",
                description="Compiles a descriptive statistics on numeric columns.",
                metrics={
                    "df_describe": describe_dict,
                    "skewness": skew_col,
                    "kurtosis": kurt_col
                    },
                recommendation_hint=hint
                )


@add_tool(tools_registry, 
      description="""
      Compiles a descriptive statistics on categorical columns 
      (n_unique, cardinality, top_values).
      """,
      category="raw_tab_file_EDA",
      eda=True,
      default=True,
      cross_file=False)
def categorical_summary(df: pd.DataFrame, config: dict):
    #
    top_n =  config["categorical_summary"].get("top_n", 10)
    threshold = config["categorical_summary"].get("threshold", 50)

    # 
    cat_df = df.select_dtypes(include="object").copy()
    n_unique = {}
    cardinality = []
    top = {}
    for col in cat_df.columns:
        unique_count = df[col].nunique()

        n_unique[col] = unique_count
        if unique_count >= threshold:
            cardinality.append(col)  

        top[col] = df[col].value_counts().head(top_n).to_dict()
    
    hint = {"cardinality": cardinality} if cardinality else {}
    
    return Observation(
                tool_name="categorical_summary",
                category="raw_tab_file_EDA",
                column="categorical columns",
                description="""
                Compiles a descriptive statistics on categorical columns 
                (n_unique, cardinality, top_values).
                """,
                metrics={
                    "n_unique": n_unique,
                    "high_cardinality": cardinality,
                    "top_values": top
                    },
                recommendation_hint=hint
                )


def datetime_already_split(df):

    parts = ["year", "month","day","hour","minute", "second"
             "an","mois","jour", "heure", "seconde"
             "yr", "sec", "hrmn",
             "Jahr", "Monat", "Tag", "Stunde", "Sekunde"]

    found = [c for c in df.columns if c.lower() in parts]

    return {
        "detected_parts": found,
        "already_split": len(found) >= 2
    }


@add_tool(tools_registry, 
      description="Checks if column can be transformed into 'datetime'.",
      category="raw_tab_file_EDA",
      eda=True,
      default=True,
      cross_file=False)
def detect_datetime_candidates(df: pd.DataFrame, 
                               config: dict | None = None):
    candidates = {
        "single_col": [],
        "splitted_cols": []
        }

    split_dict = datetime_already_split(df)
    
    if split_dict["already_split"]:
        candidates["splitted_cols"] = split_dict["detected_parts"]

    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        if "date" in col.lower():
            candidates["single_col"].append(col)
            continue

        sample = df[col].dropna().astype(str).head(100)
        
        if sample.empty:
            continue
        
        for fmt in KNOWN_FORMATS:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            success_ratio = parsed.notna().mean()

            if success_ratio > 0.8:
                candidates["single"].append(col)      

    hint = {"dt_candidates": candidates} if candidates else {}
    
    return Observation(
                tool_name="detect_datetime_candidates",
                category="raw_tab_file_EDA",
                column="str- + obj-columns",
                description="Detects datetime-like columns based on parsing success ratio.",
                metrics={
                    "dt_candidates": candidates
                    },
                recommendation_hint=hint
                )

##############################
# MORE SPECIFIC
##############################
@add_tool(tools_registry, 
      description="Evaluates column-wise if zero-inflated.",
      category="raw_tab_file_EDA", 
      eda=True,
      default=True,
      cross_file=False)
def zero_inflation_analysis(df: pd.DataFrame, config: dict = None):
    #
    threshold = config["zero_inflation_analysis"].get("threshold", None)

    numeric_df = df.select_dtypes(include=np.number)

    result = {}
    for col in numeric_df.columns:
        zero_ratio = (numeric_df[col] == 0).mean()
        if zero_ratio > threshold:
            result[col] = zero_ratio

    hint = {"zero_inf": result} if result else {}

    return Observation(
                tool_name="zero_inflation_analysis",
                category="raw_tab_file_EDA",
                column="numeric columns",
                description="Evaluates column-wise if zero-inflated.",
                metrics={
                    "zero_inflated": result
                    },
                recommendation_hint=hint
                )
