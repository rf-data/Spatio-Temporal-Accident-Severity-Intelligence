## checks_tab_EDA.py
# imports
import os
import pandas as pd

import src.agentic_AI.tools.tools_x_file_EDA as x_tool
import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.agent_helper as ah
from src.core.checks_classes import checks_registry, add_check
from src.core.finding_classes import DiagnosticFinding
from src.core.tools_classes import Observation

from src.agentic_AI.rules.merge_rules import evaluate_merge_candidates

# (
#     # recommend_merge_strategy,
#     ,
# )

# # (1) load config + parse arguments
# gh.load_env_vars(name=".env.raw_data_agent")
# config_path = os.getenv("CONFIG_PATH")
# config = fh.load_config(config_path)



@add_check(
    checks_registry,
    description="Checks for aggregated consistency between ALL files.",
    category="tabular",
    eda=True,
    default=True,
    cross_file=True
)
def run_cross_file_eda(dfs: dict[str, pd.DataFrame], 
                       config: dict | None = None):
    
    dataset_config = config.get("dataset")
    # further analysis
    x_file_comp = x_tool.cross_file_schema_compare(dfs)

    schema_compare = x_file_comp.metrics.get("comparison")
    merge_analysis = x_tool.merge_analysis(dfs, schema_compare)

    analysis_results = merge_analysis.metrics.get("comparison")
    merge_strategy = evaluate_merge_candidates(dfs, 
                                               analysis_results,
                                               dataset_config)

    # for m in [merge_strategy, merge_analysis]:
    #     print("[DEBUG]:\n", m)

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
