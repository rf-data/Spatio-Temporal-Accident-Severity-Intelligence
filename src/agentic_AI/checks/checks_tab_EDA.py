## checks_tab_EDA.py
# imports
import os
import pandas as pd

import src.agentic_AI.tools.tools_tab_EDA as tool
import src.agentic_AI.tools.tools_x_file_EDA as x_tool
import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.agent_helper as ah
from src.core.checks_classes import checks_registry, add_check
from src.core.finding_classes import DiagnosticFinding
from src.core.tools_classes import Observation

from src.agentic_AI.rules.eda_rules import filter_recommendations
#     recommend_merge_strategy,
#     ,
# )

# # (1) load config + parse arguments
# gh.load_env_vars(name=".env.raw_data_agent")
# config_path = os.getenv("CONFIG_PATH")
# config = fh.load_config(config_path)


@add_check(
    checks_registry,
    description="Checks for aggregated consistency in ONE file.",
    category="tabular",
    eda=True,
    default=True,
    cross_file=False
)
def run_file_eda(dfs: dict[str, pd.DataFrame], 
                 config: dict | None = None):

    # load function arguments
    # miss_thresh = config["run_file_eda"].get("threshold_missing", None)
    # dup_thresh = config["run_file_eda"].get("threshold_duplicates", None)
    # group = config["run_file_eda"].get("threshold_duplicates", None)
    # config_run_file = config["run_file_eda"]
    # print("config in 'run_file_eda':", config)
    #

    tools_config = config.get("tool_args", None)
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
        if isinstance(base, Observation) and isinstance(base.recommendation_hint, dict):
            params.append(base)

        misses = tool.missing_analysis(df, tools_config)
        missing[name] = misses
        if isinstance(misses, Observation) and isinstance(misses.recommendation_hint, dict):
            params.append(misses)

        inf = tool.infinite_analysis(df)
        infinite[name] = inf
        if isinstance(inf, Observation) and isinstance(inf.recommendation_hint, dict):
            params.append(inf)

        dups = tool.duplicate_analysis(df, tools_config)
        duplicates[name] = dups
        if isinstance(dups, Observation) and isinstance(dups.recommendation_hint, dict):
            params.append(dups)

        numeric = tool.numeric_summary(df, tools_config)
        num_sum[name] = numeric
        if isinstance(numeric, Observation) and isinstance(numeric.recommendation_hint, dict):
            params.append(numeric)

        categorical = tool.categorical_summary(df, tools_config)
        cat_sum[name] = categorical
        if isinstance(categorical, Observation) and isinstance(categorical.recommendation_hint, dict):
            params.append(categorical)

        zero = tool.zero_inflation_analysis(df, tools_config)
        zero_inf[name] = zero
        if isinstance(zero, Observation) and isinstance(zero.recommendation_hint, dict):
            params.append(zero)

        cand = tool.detect_datetime_candidates(df)
        dt_candidates[name] = cand
        if isinstance(cand, Observation) and isinstance(cand.recommendation_hint, dict):
            params.append(cand)

        data_processing_strategy[name] = filter_recommendations(params, 
                                                                tools_config)
        # for m in [overview, missing, infinite, duplicates,
        #         num_sum, cat_sum, zero_inf, dt_candidates]:
        #     print("[DEBUG]:\n", m)
    # for key, value in data_processing_strategy.items():
        # for 
        # print(f"[DEBUG] length data_processing_actions ('{key}'):", 
        #     len(value))
        
    return DiagnosticFinding(
        check_name="run_file_eda",
        description="Checks for aggregated consistency in ONE file.",
        category="raw_file_EDA",
        column="all",
        metrics={
            "overview": overview,       # dict[name_metric, values]
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

