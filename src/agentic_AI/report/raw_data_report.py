## raw_data_report.py
# imports
# from pydantic import BaseModel
# from typing import Optional
# from datetime import datetime
import os
from pathlib import Path
import json
from collections import defaultdict

import src.utils.general_helper as gh
import src.utils.file_helper as fh
import src.utils.agent_helper as ah
from src.core.report_classes import PreparationSummary
from src.core.tools_classes import  Observation


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def pandas_to_sql_dtype(dtype_str: str):
    if "int" in dtype_str:
        return "INTEGER"
    if "float" in dtype_str:
        return "DOUBLE PRECISION"
    if "datetime" in dtype_str:
        return "TIMESTAMP"
    if "bool" in dtype_str:
        return "BOOLEAN"
    return "VARCHAR(255)"


def load_eda_summary(config):
    gh.load_env_vars()

    report_folder = os.getenv("FOLDER_REPORT")

    arg_dict = config.get("general_args", {})

    report_name = arg_dict.get("summary_file_name")
    report_path = Path(report_folder) / f"{report_name}.json"
    
    with open(report_path) as f:
        report = json.load(f)

    files = report.get("metadata", {}).get("files_analyzed", {})
    processing = report["processing"]

    return {
        "report": report, 
        "files": files,
        "processing": processing
        }

   
def build_sql_schema(dfs, merge_statement, table_name):

    # Beispiel: nimm erstes DF als Referenz
    _, df = next(iter(dfs.items()))

    sql_columns = {
            col: pandas_to_sql_dtype(str(df[col].dtype))
            for col in df.columns
        }

    primary_key = None

    if merge_statement:
        strategy = merge_statement.merge_strategy
        if strategy and strategy.strategy == "merge": 
            primary_key = strategy.join_key

            print("[MERGE] Automated join for %s", 
                  merge_statement.files)
        else:
            print("[NO MERGE] No automated join for %s (strategy=%s)\n%sreason: %s", 
                merge_statement.files,
                strategy.strategy,
                strategy.reason
                )

    return {
        "table_name": table_name,
        "columns": sql_columns,
        "primary_key": primary_key,
        "indexes": primary_key
        }


def build_feature_engineering_plan(data_processing_plan):

    plan = {
        "create_datetime_columns": [],
        "lag_rolling_features": [],
        "seasonality_features": [],
        "zero_streak_feature": [],
        "log_transform_candidates": [],
        "scaling_candidates": []
        }

    for _, actions in data_processing_plan.items():
        for act in actions:
            # print("[DEBUG] type + print", type(act), act)

            if act.action == "parse_datetime":
                plan["create_datetime_columns"].extend(act.target)
                plan["lag_rolling_features"].extend(act.target)
                plan["seasonality_features"].extend(act.target)

            if act.action == "handle_zero_inflation":
                plan["zero_streak_feature"].extend(act.target)

            if act.action == "handle_skewness":
                plan["log_transform_candidates"].extend(act.target)

            if act.action == "handle_kurtosis":
                plan["scaling_candidates"].extend(act.target)    

    # 🔹 Deduplizieren
    for k in plan:
        plan[k] = list(set(plan[k]))

    return plan


# ------------------------------
# MAIN FUNCTIONS
# ------------------------------
def generate_preparation_summary(dfs, findings, meta_data):
    
    metric_dict = defaultdict(dict)
    data_processing = {}    # file -> actions
    merge_dict = {}     # pair -> strategy
    schema_proposal = None

    table_name = meta_data.get("table_name", None)

    for i, f in enumerate(findings):
        # extract 'oberservation metrics'
        # metric_dict = extract_finding_metrics(f)

        # overview metric results
        metrics = f.metrics

        if not isinstance(metrics, dict):
            print("[INFO] dtype metrics:", type(metrics))
            print("[INFO] metrics:", metrics)
            # continue
        
        for tool_name, file_results in metrics.items():
            if isinstance(file_results, Observation):
                file_results = file_results.model_dump()
                
            for file_name, observation in file_results.items():
                metric_dict[str(file_name)][tool_name] = observation
                   
        # extract 'recommendations'
        hint = f.recommendation_hint
        if not isinstance(hint, dict):
            print("[INFO] dtype hint", type(hint))
            print("[INFO] hint:", hint)
            # continue
            
        # extract 'data processing strategy'
        proc = hint.get("processing", None)
        if proc:    # file -> actions
            for file_name, actions in proc.items():
                print("[DEBUG] file_name - len(actions):", file_name, len(actions))
                data_processing.setdefault(file_name, [])
                data_processing[file_name].extend(actions)

        # extract 'merge strategy'  
        merge_recomm = hint.get("merge", None)

        if merge_recomm:
            for pair_name, merge_statement in merge_recomm.items():
                merge_dict[pair_name] = merge_statement

            schema_proposal = build_sql_schema(
                                    dfs, 
                                    merge_statement, 
                                    table_name
                                    )

    print(f"[DEBUG] data_processing (dtype={type(data_processing)}):\n", 
          data_processing)

    feat_eng_plan = build_feature_engineering_plan(data_processing)

    return PreparationSummary(
                        metric_results=metric_dict,
                        processing=data_processing,
                        merge=merge_dict,
                        schema_proposal=schema_proposal,
                        feature_engineering=feat_eng_plan,
                        metadata=meta_data,
                        )


def build_and_save_summary(summary: PreparationSummary, 
                           name, 
                           json_only=True, 
                           separated=False,
                           base_path=None):

    now = summary.metadata.get("analysis_timestamp", "")
    sum_name = summary.metadata.get("summary_name", "")

    if base_path is None:
        # (1) load config
        gh.load_env_vars()
        base_path = os.getenv("FOLDER_REPORT")

    # convert summary to dict/json
    sum_dict = summary.model_dump()

    # JSON
    if separated: 
        metrics = sum_dict.get("metric_results", {})
        file_process = sum_dict.get("processing", {})
        merge = sum_dict.get("merge", {})
        schema_proposal = sum_dict.get("schema_proposal", {})
        feat_eng = sum_dict.get("feature_engineering", {})
        metadata = sum_dict.get("metadata", {})

        for name, sep in [("metrics", metrics), 
                          ("f_process", file_process),
                          ("merge", merge),
                          ("schema", schema_proposal),
                          ("FeatEng", feat_eng),
                          ("meta_data", metadata)]:

            sep_path = f"{base_path}/{now}_{sum_name}_EDA_{name}.json"
     
            with open(sep_path, "w") as f:
                json.dump(fh.make_json_safe(sep), f, indent=2, default=str)  

    else:
        json_path = f"{base_path}/{now}_{name}_EDA_summary.json"
     
        with open(json_path, "w") as f:
            json.dump(fh.make_json_safe(sum_dict), f, indent=2, default=str)

    if not json_only: 
        # Markdown
        md_path = f"{base_path}/{now}_{sum_name}_EDA_summary.md"

        md_content = build_markdown_summary(sum_dict)
        with open(md_path, "w") as f:
            f.write(md_content)

    return     

#######################################################################################
# ------------------------------
# UNDER CONSTRUCTION
# ------------------------------
def build_markdown_summary(summary:dict) -> str:

    metric_results = summary.get("metric_results")
    meta_data =  summary.get("metadata")
    files_analysed =meta_data.get("files_analyzed", "n.a.")
    timestamp = meta_data.get('analysis_timestamp', '')
    agent_version = meta_data.get('agent_version', 'tba')

    md = []
    md.append(f"# Raw Data EDA Report")
    md.append("")
    md.append(f"files analysed:")
    md.append(ah.to_md_safe(files_analysed))
    md.append(f"\nanalysis_timestamp:\t{timestamp}")
    md.append(f"agent_version:\t{agent_version}\n")

    # 🔹 Metrics
    for file_name, tool_dict in metric_results.items():
        md.append(f"## Tools observations -- {file_name}")
       
        for tool_name, result in tool_dict.items():
            
            md.append(f"### {tool_name}")
            if not isinstance(result, dict):
                print("dtype:", type(result))
                print(result)
                continue 

            for m in result.get("metrics"):

                if isinstance(m, dict):
                    for m_name , m_values in m.items():
                        md.append(f"name: {m_name.upper()}")
                        md.append(ah.to_md_safe(m_values))
                        md.append("")
                else:
                    md.append(f"name / result:{m}")
                    md.append("")
                    
                
                md.append("")
            md.append("")
            # \nresults:\n{(result)}")
            # if action.params:
            #     for arg, value in action.params.items():
            #         md.append(f"  param '{arg}' - value: {ah.to_md_safe(value)}")
        md.append("")


    # 🔹 Processing
    md.append("## Derived Actions -- 'Data Processing Plan'")
    for file_name, actions in summary.processing.items():
        md.append(f"### File: {file_name}")
        for action in actions:
            md.append(f"- {action}")
            if action.params:
                for arg, value in action.params.items():
                    md.append(f"  param '{arg}' - value: {ah.to_md_safe(value)}")
            md.append("")

    # 🔹 Merge
    md.append("## Derived Actions -- 'Merge Strategy'")
    for pair_name, details in summary.merge.items():

        join_keys = ah.to_md_safe(details.get('recommended_join_keys'))
        md.append(f"### {pair_name}")
        md.append(f"- Join Type: {details.get('join_type')}")
        md.append(f"- Join Keys: {join_keys}")
        md.append("")
        
    # 🔹 Feature Engineering
    md.append("## Feature Engineering")
    for plan_name, cols in summary.feature_engineering.items():
        md.append(f"- {plan_name}: {ah.to_md_safe(cols)}")

    # 🔹 SQL Schema
    if summary.sql_schema:
        md.append("\n## SQL Schema Proposal")
        md.append(f"Table: {summary.sql_schema.get('table_name')}")
        for col, dtype in summary.sql_schema.get("columns", {}).items():
            md.append(f"- {ah.to_md_safe(col)}: {dtype}")

    return "\n".join(md)



    # rename_columns = {...}
    # drop_columns = [...]
    # type_casts = {...}
    # datetime_parsing = {...}
    # imputation = {...}
    # outlier_handling = {...}
    # encoding =  {...}
    # scaling = {...}
    # zero_inflation = {...}

#         rec_process = check_find["recommendation_hint"].get("processing", None)
#         rec_merge = check_find["recommendation_hint"].get("merge", None)

#         if rec_merge:
#            preparation_summary["merge_strategy"] = rec_merge

#         if rec_process:
#             preparation_summary["data_processing"] = rec_process

#     [...]

#     return 


#     preparation_summary = {
#   "data_processing": {
#     "rename_columns": {...},
#     "drop_columns": [...],
#     "type_casts": {...},
#     "datetime_parsing": {...},
#     "imputation": {...},
#     "outlier_handling": {...},
#     "encoding": {...},
#     "scaling": {...},
#     "zero_inflation": {...},
#   },
# #   "merge_strategy": {...},   # pro file pair
#   "sql_schema": {...},       # optional, aber für dich sehr wertvoll
#   "scripts_to_generate": [...]  # Liste der Dateien, die erzeugt werden
# }



# def build_json_report(risk_summary, findings, llm_analysis=None):

#     return AuditReport(
#         generated_at=datetime.now(),
#         total_score=risk_summary.total_score,
#         risk_level=risk_summary.risk_level.value,
#         finding_count=risk_summary.finding_count,
#         findings=[f.model_dump() for f in findings],
#         top_findings=risk_summary.top_findings,
#         llm_analysis=llm_analysis.model_dump() if llm_analysis else None
#     )



# def generate_preparation_summary():

#     return 