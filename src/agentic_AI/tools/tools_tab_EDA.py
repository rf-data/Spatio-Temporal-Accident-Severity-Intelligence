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
##############################
@add_tool(tools_registry, 
      description="Create basic overview on data: shape, column names + dtypes",
      category="raw_tab_file_EDA")
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
                recommendation_hint=None
                )


@add_tool(tools_registry, 
      description="Evaluates column-wise if there any NaN values.",
      category="raw_tab_file_EDA")
def missing_analysis(df: pd.DataFrame, config: dict | None = None):
    threshold = config["missing_analysis"].get("threshold", 0.5)
    recomm = ""

    recs = {}
    for col in df.columns:
        nan_ratio = df[col].isna().mean() 
        if nan_ratio > threshold:
            recs[col] = nan_ratio
    
    if recs:
        recomm = f"""
        There are columns having a nan_ratio above the allowed threshold ({threshold}):\n
        {recs.keys()}
        """

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
                recommendation_hint={"missing": recomm}
                )


@add_tool(tools_registry, 
      description="Evaluates column-wise if there any np.inf values.",
      category="raw_tab_file_EDA")
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

    hint = {"infinite": recs} if recs else None
    
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
      category="raw_tab_file_EDA")
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

    recs = []
    if ratio_dups > threshold: 
        recs.append("High duplicates ratio (general)")

    if isinstance(ratio_dups_group, (float, int)) and ratio_dups_group > threshold:
        recs.append("High duplicates ratio (grouping)")

    return Observation(
                tool_name="duplicate_analyis",
                category="raw_tab_file_EDA",
                column="all",
                description="Checks if there any duplicates.",
                metrics={
                    "n_duplicates (general)": n_dups,
                    "duplicate_ratio (general)": ratio_dups,
                    "n_duplicates (grouped)": n_dups_group,
                    "duplicate_ratio (grouped)": ratio_dups_group
                    },
                recommendation_hint={"duplicates": recs}
                )


@add_tool(tools_registry, 
      description="Compiles a descriptive statistics on numeric columns.",
      category="raw_tab_file_EDA")
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

    recs = recommend_from_distribution(skew_col, kurt_col)

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
                recommendation_hint={"Skew_Kurt": recs}
                )


@add_tool(tools_registry, 
      description="""
      Compiles a descriptive statistics on categorical columns 
      (n_unique, cardinality, top_values).
      """,
      category="raw_tab_file_EDA")
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
    
    hint = {"cardinality": cardinality} if cardinality else None
    
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

@add_tool(tools_registry, 
      description="Checks if column can be transformed into 'datetime'.",
      category="raw_tab_file_EDA")
def detect_datetime_candidates(df: pd.DataFrame, config: dict | None = None):
    candidates = []


    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        if "date" in col.lower():
            candidates.append(col)
            continue

        sample = df[col].dropna().astype(str).head(100)
        
        if sample.empty:
            continue
        
        for fmt in KNOWN_FORMATS:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            success_ratio = parsed.notna().mean()

            if success_ratio > 0.8:
                candidates.append(col)      

    hint = {"dt_candidates": candidates} if candidates else None
    
    return Observation(
                tool_name="detect_datetime_candidates",
                category="raw_tab_file_EDA",
                column="str- + obj-columns",
                description="Detects datetime-like columns based on parsing success ratio.",
                metrics={
                    "datetime_candidates": candidates
                    },
                recommendation_hint=hint
                )

##############################
# MORE SPECIFIC
##############################
@add_tool(tools_registry, 
      description="Evaluates column-wise if zero-inflated.",
      category="raw_tab_file_EDA")
def zero_inflation_analysis(df: pd.DataFrame, config: dict = None):
    #
    threshold = config["zero_inflation_analysis"].get("threshold", None)

    numeric_df = df.select_dtypes(include=np.number)

    result = {}
    for col in numeric_df.columns:
        zero_ratio = (numeric_df[col] == 0).mean()
        if zero_ratio > threshold:
            result[col] = zero_ratio

    hint = {"zero_inf": result} if result else None 

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


@add_tool(tools_registry, 
      description="Pairwise schema comparison between DataFrames",
      category="raw_tab_cross_file_EDA")
def cross_file_schema_compare(dfs: dict[str, pd.DataFrame], config: dict | None = None):

    schema = {name: set(df.columns) for name, df in dfs.items()}
    names = list(schema.keys())
    
    comparison = {}

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]
            b = names[j]

            comparison[f"{a}_vs_{b}"] = {
                    "file_a": a,
                    "file_b": b,
                    "only_in_a": list(schema[a] - schema[b]),
                    "only_in_b": list(schema[b] - schema[a]),
                    "common": list(schema[a] & schema[b])
                    }

    return Observation(
                tool_name="cross_file_schema_compare",
                category="raw_tab_cross_file_EDA",
                column="all columns",
                description="Pairwise schema comparison between DataFrames",
                metrics={
                    "comparison": comparison
                    },
                recommendation_hint=None
                )


@add_tool(tools_registry, 
        description="""
        Evaluates df-wise if and how data could be merged 
        (common_values, type_conflicts, cardinality_mismatch)
        """,
        category="raw_tab_cross_file_EDA")
def merge_analysis(dfs: dict[str, pd.DataFrame], schema_compare: dict):

    # col_dict = {}
    # dtype_dict = {}

    results = {}
    for pair_name, pair_info in schema_compare.items():
        a = pair_info["file_a"]
        b = pair_info["file_b"]
        common_cols = pair_info["common"]

        df_a = dfs[a]
        df_b = dfs[b]

    type_conflicts = {}
    cardinality = {}

    for col in common_cols:
        # dtype comparison
        dtype_a = str(df_a[col].dtype)
        dtype_b = str(df_b[col].dtype)
        
        if dtype_a != dtype_b:
            type_conflicts[col] = {
                            a: dtype_a,
                            b: dtype_b
                            }
            
        # cardinality comparison
        nunique_a = df_a[col].nunique(dropna=True)
        nunique_b = df_b[col].nunique(dropna=True)

        if min(nunique_a, nunique_b) > 0:
            ratio = max(nunique_a, nunique_b) / min(nunique_a, nunique_b)
        else:
            ratio = None

        if ratio and ratio > 10:
            cardinality[col] = {
                a: nunique_a,
                b: nunique_b,
                "ratio": ratio
                }

        results[pair_name] = {
                    "file_a": a,
                    "file_b": b,
                    "common_columns": common_cols,
                    "type_conflicts": type_conflicts,
                    "cardinality_mismatch": cardinality
                    }

    return Observation(
                tool_name="merge_analysis",
                category="raw_tab_cross_file_EDA",
                column="all columns",
                description="""
                Evaluates df-wise if and how data could be merged 
                (common_values, type_conflicts, cardinality_mismatch)
                """,
                metrics={
                    "comparison": results
                    },
                recommendation_hint=None
                )

    