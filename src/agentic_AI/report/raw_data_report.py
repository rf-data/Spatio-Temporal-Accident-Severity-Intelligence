## raw_data_report.py
# imports
# from pydantic import BaseModel
# from typing import Optional
# from datetime import datetime
import os
import json

import src.utils.general_helper as gh
import src.utils.agent_helper as ah
from src.core.report_classes import PreparationSummary

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


# def generate_sql_schema(final_columns, dtypes, merge_strategy):
   
def build_sql_schema(dfs, merge_strategy, table_name):

    # Beispiel: nimm erstes DF als Referenz
    name, df = next(iter(dfs.items()))

    sql_columns = {
            col: pandas_to_sql_dtype(str(df[col].dtype))
            for col in df.columns
        }

    primary_key = None

    if merge_strategy:
        for pair in merge_strategy.values():
            if pair["recommended_join_keys"]:
                primary_key = pair["recommended_join_keys"]
                break

    return {
        "table_name": table_name,
        "columns": sql_columns,
        "primary_key": primary_key,
        "indexes": primary_key
        }


def build_feature_engineering_plan(data_processing_plan):

    plan = {
        "create_datetime_columns": [],
        "zero_streak_feature": [],
        "lag_rolling_features": [],
        "seasonality_features": [],
        "log_transform_candidates": [],
        "scaling_candidates": []
        }

    for _, actions in data_processing_plan.items():
        for act in actions:
            if act.action == "parse_candidates":
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


def generate_preparation_summary(dfs, findings, meta_data):
        
    data_processing = {}   # file -> actions
    merge_strategy = {}    # pair -> strategy
    sql_schema = None

    table_name = meta_data.get("sql_table", None)
    for i, f in enumerate(findings):
        hint = f.recommendation_hint
        if not isinstance(hint, dict):
            continue
            
        proc = hint.get("processing", None)
        if proc:    # file -> actions
            for file_name, actions in proc.items():
                data_processing.setdefault(file_name, [])
                data_processing[file_name].extend(actions)
            
        merge = hint.get("merge", None)
        if merge:
            for pair_name, details in merge.items():
                
                # merge_strategy.setdefault(pair_name, {})
                merge_strategy[pair_name] = details

            schema = build_sql_schema(
                                    dfs, 
                                    merge, 
                                    table_name
                                    )
            # sql_schema.setdefault(pair_name, [])
            sql_schema = schema # [f"schema_{i}"] 
            print(f"created sql_schema 'schema_{i}'")

    feat_eng_plan = build_feature_engineering_plan(data_processing)
    
    return PreparationSummary(
                        processing=data_processing,
                        merge=merge_strategy,
                        sql_schema=sql_schema,
                        feature_engineering=feat_eng_plan,
                        metadata=meta_data,
                        )


def build_and_save_summary(summary: PreparationSummary, 
                           name, 
                           base_path=None):

    # timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
    now = summary.metadata.get("analysis_timestamp", "")
    name = summary.metadata.get("summary_name", "")

    if base_path is None:
        # (1) load config
        gh.load_env_vars()
        base_path = os.getenv("FOLDER_REPORT")
        
    json_path = f"{base_path}/{now}_{name}_EDA_summary.json"
    md_path = f"{base_path}/{now}_{name}_EDA_summary.md"

    # JSON
    with open(json_path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2, default=str)

    # Markdown
    md_content = build_markdown_summary(summary)
    with open(md_path, "w") as f:
        f.write(md_content)

    return {"json": json_path, "markdown": md_path}        


def build_markdown_summary(summary: PreparationSummary) -> str:

    files_analysed = summary.metadata.get("files_analyzed", "n.a.")
    timestamp = summary.metadata.get('analysis_timestamp', '')
    agent_version = summary.metadata.get('agent_version', 'tba')

    md = []
    md.append(f"# Raw Data EDA Report")
    md.append("")
    md.append(f"files analysed:")
    md.append(ah.to_md_safe(files_analysed))
    md.append(f"\nanalysis_timestamp:\t{timestamp}")
    md.append(f"\nagent_version:\t{agent_version}")

    # 🔹 Processing
    md.append("## Data Processing Plan")
    for file_name, actions in summary.processing.items():
        md.append(f"### File: {file_name}")
        for action in actions:
            md.append(f"- {action}")
            if action.params:
                for arg, value in action.params.items():
                    md.append(f"  param '{arg}' - value: {ah.to_md_safe(value)}")
            md.append("")

    # 🔹 Merge
    md.append("## Merge Strategy")
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