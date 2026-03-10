## agent_utils.py
# imports
from openai import OpenAI
import re
import importlib
import json
import os
from pathlib import Path
import argparse
import pkgutil


from src.agentic_AI.agent.constraints import ALLOWED_TABLES, FORBIDDEN_KEYWORDS


# --------------------
# DATA AUDITOR
# --------------------
def get_openai_client():
    client = OpenAI()

    return client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing report"
    )
    return parser.parse_args()


def guard_report_creation(config: dict, report: bool = True, force: bool = False):

    name_arg = "report" if report else "summary"

    r_folder = os.getenv("FOLDER_REPORT")
    r_name = config.get("general_args", None).get(f"{name_arg}_name", None)

    md_path = Path(r_folder) / f"audit_{r_name}.md"
    json_path = Path(r_folder) / f"audit_{r_name}.json"

    existing_files = [p for p in [md_path, json_path] if p.exists()]

    if existing_files and not force:
        files_str = ", ".join(p.name for p in existing_files)
        raise FileExistsError(
            f"Report already exists ({files_str}). " f"Use --force to overwrite."
        )

    return


def fill_registry_with_module(module_path: str):
    importlib.import_module(module_path)


def load_modules(package):

    print("Loading check/tool modules:")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        name = f"{package.__name__}.{module_name}"
        importlib.import_module(name)
        print(" -", name)

def validate_query(query: str) -> tuple[bool, str]:
    query_clean = query.strip().upper()

    # Only SELECT commands permitted
    if not query_clean.startswith("SELECT"):
        return False, "Only SELECT statements are allowed."

    # No multiple statements (e.g., separated by ';')
    if ";" in query_clean[:-1]:
        return False, "Multiple SQL statements are not allowed."

    # No forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in query_clean:
            return False, f"Forbidden keyword detected: {keyword}"

    # Check if accessed tables are in the allowed list
    if ALLOWED_TABLES:
        table_pattern = re.findall(r"FROM\s+([A-Z0-9_.]+)", query_clean)
        table_pattern += re.findall(r"JOIN\s+([A-Z0-9_.]+)", query_clean)

        for table in table_pattern:
            if table.lower() not in ALLOWED_TABLES:
                return False, f"Access to table '{table}' is not allowed."

    return True, "Query validated successfully."


def load_last_report(base_path="audit_reports"):

    files = [f for f in os.listdir(base_path) if f.endswith(".json")]

    if not files:
        return None

    latest = sorted(files)[-1]

    with open(os.path.join(base_path, latest)) as f:
        data = json.load(f)

    return data


#####
def _get_value(obj, value="recommendation_hint"):
    if isinstance(obj, dict):
        h = obj.get(value, None)
    else:
        h = getattr(obj, value, None) or {}
    return h


def to_md_safe(value):
    if value is None:
        return "n.a."
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    if hasattr(value, "keys"):
        return ", ".join(map(str, value))
    return str(value)


# if isinstance(h, dict) else {}


# def execute_check(check_name, check_fn, engine, audit_config):

#     table = ""
#     schema = ""
#     time_column=""
#     frequency=""
#     group_column=""
#     metric_column=""
#     aggregate_column=""

#     if check_name == "time_integrity":
#         findings = check_fn(
#                 engine,
#                 table,
#                 schema,
#                 time_column,
#                 frequency,
#                 config=audit_config
#                 )

#     else:
#         findings = check_fn(
#             engine,
#             table,
#             schema,
#             group_column,
#             metric_column,
#             aggregate_column,
#             config=audit_config
#             )

#     return findings


# def run_reflexive_agent(engine, goal, audit_config, risk_config, max_iterations=1):

#     state = AgentState(goal=goal)

#     # Initial planning
#     planned_checks = plan_checks(goal, CHECK_REGISTRY)

#     # Execution
#     iteration = 0

#     while iteration < max_iterations and planned_checks:

#         for check_name in planned_checks:

#             if check_name in state.executed_checks:
#                 continue

#             check_fn = CHECK_REGISTRY[check_name]

#             findings = execute_check(check_name, check_fn, engine, audit_config)

#             state.executed_checks.append(check_name)
#             state.findings.extend(findings)

#         # Reflexion
#         planned_checks = reflect_and_plan(goal, state, CHECK_REGISTRY)

#         iteration += 1

#     # Risk Scoring
#     state.risk_summary = score_findings(state.findings, risk_config)

#     # LLM interpretation if necessary
#     if state.risk_summary.risk_level.value in ["medium", "high", "critical"]:
#         state.llm_analysis = run_llm_reasoning(state.risk_summary)

#     return state

# if __name__ == "__main__":
#     run_reflexive_agent()


###################################################################################################################
# -----------------------------------------------------------------------------------------------------------------
###################################################################################################################

"""
🟢 Easy Win #1: Agent als Data Quality Auditor (nicht Cleaner)

Idee
Klassische Pipeline macht:
Cleaning
Typisierung
Feature Engineering
Agent analysiert Ergebnisse, nicht Rohdaten

Agent-Aufgaben
Findet:
auffällige Verteilungen
Zeitbrüche
räumliche Inkonsistenzen

Generiert:
„Data Quality Reports“
Hypothesen, keine Änderungen

📌 Wichtig:
Agent schreibt nichts zurück → nur Read-only.

Zeitbedarf
3–5 Tage MVP
sehr stabil
gut erklärbar
"""

# --------------------
# REASONING
# --------------------

dataset_attributes = ["spatial-temporal", "Road accidents"]

# --------------------
# AGENT_LOOP
# --------------------
