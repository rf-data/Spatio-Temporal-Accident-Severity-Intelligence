## tools_tab_EDA.py
# imports
import pandas as pd
import numpy as np

from src.core.tools_classes import tools_registry, add_tool, Observation
from src.agentic_AI.rules.eda_rules import recommend_from_distribution



@add_tool(tools_registry, 
      description="Pairwise schema comparison between DataFrames",
      category="raw_tab_cross_file_EDA",
      eda=True,
      default=True,
      cross_file=True)
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
                    "n_rows_a": len(a),
                    "n_rows_b": len(b),
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
        category="raw_tab_cross_file_EDA",
        eda=True,
        default=True,
        cross_file=True)
def merge_analysis(dfs: dict[str, pd.DataFrame], schema_compare: dict):

    # col_dict = {}
    # dtype_dict = {}

    results = {}
    for pair_name, pair_info in schema_compare.items():
        a = pair_info["file_a"]
        b = pair_info["file_b"]
        only_in_a = pair_info["only_in_a"]
        only_in_b = pair_info["only_in_b"]
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
                    "n_rows_a": len(a),
                    "n_rows_b": len(b),
                    "only_in_a": only_in_a,
                    "only_in_b": only_in_b,
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

    