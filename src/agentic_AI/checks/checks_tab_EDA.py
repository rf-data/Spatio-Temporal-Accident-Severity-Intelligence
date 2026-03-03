## checks_tab_EDA.py
# imports
import os
import pandas as pd

import src.agentic_AI.tools.tools_tab_EDA as tool
import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.agent_helper as ah
from src.core.checks_classes import checks_registry, add_check
from src.core.finding_classes import DiagnosticFinding
from src.agentic_AI.rules.eda_rules import (
    recommend_merge_strategy,
    filter_recommendations,
)

# # (1) load config + parse arguments
# gh.load_env_vars(name=".env.raw_data_agent")
# config_path = os.getenv("CONFIG_PATH")
# config = fh.load_config(config_path)


@add_check(
    checks_registry,
    description="Checks for aggregated consistency in ONE file.",
    category="raw_file_EDA",
)
def run_file_eda(dfs: dict[str, pd.DataFrame], config: dict | None = None):

    # load function arguments
    # miss_thresh = config["run_file_eda"].get("threshold_missing", None)
    # dup_thresh = config["run_file_eda"].get("threshold_duplicates", None)
    # group = config["run_file_eda"].get("threshold_duplicates", None)
    config_run_file = config["run_file_eda"]

    #
    overview = {}
    missing = {}
    infinite = {}
    duplicates = {}
    num_sum = {}
    cat_sum = {}
    zero_inf = {}
    dt_candidates = {}

    data_processing_strategy = {}

    for name, df in dfs.items():
        params = []

        base = tool.basic_overview(df)
        overview[name] = base
        if isinstance(base, dict) and isinstance(base.get("recommendation_hint"), dict):
            params.append(base)

        misses = tool.missing_analysis(df, config_run_file)
        missing[name] = misses
        if isinstance(misses, dict) and isinstance(
            misses.get("recommendation_hint"), dict
        ):
            params.append(misses)

        inf = tool.infinite_analysis(df)
        infinite[name] = inf
        if isinstance(inf, dict) and isinstance(inf.get("recommendation_hint"), dict):
            params.append(inf)

        dups = tool.duplicate_analysis(df, config_run_file)
        duplicates[name] = dups
        if isinstance(dups, dict) and isinstance(dups.get("recommendation_hint"), dict):
            params.append(dups)

        numeric = tool.numeric_summary(df, config_run_file)
        num_sum[name] = numeric
        if isinstance(numeric, dict) and isinstance(
            numeric.get("recommendation_hint"), dict
        ):
            params.append(numeric)

        categorical = tool.categorical_summary(df, config_run_file)
        cat_sum[name] = categorical
        if isinstance(categorical, dict) and isinstance(
            categorical.get("recommendation_hint"), dict
        ):
            params.append(categorical)

        zero = tool.zero_inflation_analysis(df, config_run_file)
        zero_inf[name] = zero
        if isinstance(zero, dict) and isinstance(zero.get("recommendation_hint"), dict):
            params.append(zero)

        cand = tool.detect_datetime_candidates(df)
        dt_candidates[name] = cand
        if isinstance(cand, dict) and isinstance(cand.get("recommendation_hint"), dict):
            params.append(cand)

        data_processing_strategy[name] = filter_recommendations(params, 
                                                                config_run_file)

    return DiagnosticFinding(
        check_name="run_file_eda",
        description="Checks for aggregated consistency in ONE file.",
        category="raw_file_EDA",
        column="all",
        metrics={
            "overview": overview,
            "missing": missing,
            "infinite": infinite,
            "duplicates": duplicates,
            "numeric": num_sum,
            "categorical": cat_sum,
            "zero_inflation": zero_inf,
            "detect_dt_candidates": dt_candidates
        },
        severity=None,
        recommendation_hint={"processing": data_processing_strategy}
    )


@add_check(
    checks_registry,
    description="Checks for aggregated consistency between ALL files.",
    category="raw_cross_file_EDA",
)
def run_cross_file_eda(dfs: dict[str, pd.DataFrame], config: dict | None = None):
    # further analysis
    x_file_comp = tool.cross_file_schema_compare(dfs)

    schema_compare = x_file_comp.metrics.get("comparison", None)
    merge_analysis = tool.merge_analysis(dfs, schema_compare)

    analysis_results = merge_analysis.metrics.get("comparison", None)
    merge_strategy = recommend_merge_strategy(analysis_results)

    return DiagnosticFinding(
        check_name="run_cross_file_eda",
        description="Checks for aggregated consistency in ONE file.",
        category="raw_cross_file_EDA",
        column="all",
        metrics={
            "merge_analysis": merge_analysis,
            "cross_file_schema_comparison": x_file_comp
        },
        severity=None,
        recommendation_hint={"merge": merge_strategy}
    )
