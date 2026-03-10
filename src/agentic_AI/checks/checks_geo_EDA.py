## checks.py
# imports
# from src.agentic_AI.findings.audit_findings import (
#     Finding,
#     SeverityLevel,
#     resolve_severity,
# )

import os
import pandas as pd

import src.agentic_AI.tools.tools_geo_EDA as geo
from src.core.checks_classes import checks_registry, add_check
from src.core.finding_classes import DiagnosticFinding, ActionSchema
from src.core.tools_classes import Observation

from src.agentic_AI.rules.eda_rules import filter_recommendations
# --------------------
# CHECKS ON GEO-DATA
# --------------------

# check_spatial_consistency()
# check_cross_resolution_consistency()


@add_check(
    checks_registry,
    description="""
    Runs basic geo checks (e.g. 'geo_col_candidates exist?', 
    'how many duplicates?')
    """,
    category="geo",
    eda=True,
    default=True,
    cross_file=False
)
def run_basic_geo_check(dfs: dict[str, pd.DataFrame], 
                 config: dict | None = None):
    
    tools_config = config.get("tool_args", None)

    # duplicates = {}
    geo_candidates = {}
    geo_dups = {}
    data_processing_strategy = {}

    for name, df in dfs.items():
        params = []

        geo_cols = geo.detect_geo_columns(df, config)
        geo_candidates[name] = geo_cols

        if isinstance(geo_cols, Observation) and isinstance(geo_cols.recommendation_hint, dict):
            params.append(geo_cols)

        duplicates = geo.find_geo_dups(df, geo_cols, config)
        geo_dups[name] = duplicates
        if isinstance(duplicates, Observation) and isinstance(duplicates.recommendation_hint, dict):
            params.append(duplicates)


        # geo_cols = geo_obs.metrics.get("geo_col_candiates", [])
        # geo_candidates[name] = geo geo_cols

        # duplicates[name] = observations.metrics.get("duplicates", [])

        data_processing_strategy[name] = filter_recommendations(params, 
                                                                tools_config)
        # observations.recommendation_hint
        # recommend_geo_processing(observations, config)
    
    return DiagnosticFinding(
                        check_name="run_basic_geo_check",
                        description="""
                        Runs basic geo checks (e.g. 'geo_col_candidates exist?', 
                        'how many duplicates?')
                        """,
                        category="raw_basic_geo_EDA",
                        column="geo",
                        metrics={
                            "geo_cols": geo_candidates,
                            "geo_duplicates": geo_dups
                            },
                        severity=None,
                        recommendation_hint={"processing": data_processing_strategy}
                    )
        # col_dict = {
        #     "lat_candidates": [],
        #     "lon_candidates": [],
        #     "gps_candidates": [],
        #     "geo_duplicates": {}
        #     }