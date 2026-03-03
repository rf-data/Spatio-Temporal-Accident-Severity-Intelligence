## run_audit_agent.py
# imports
import os

# from agent_audit.findings.layer_finding import score_all_findings, gather_findings
# from agent_audit.layer_reasoning import run_llm_reasoning
from src.agentic_AI.report.auditor_report import build_json_report, save_report
from src.core.agent_classes import AuditAgent, AuditContext

import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.postgre_helper as post
import src.utils.agent_helper as ah
from src.utils.agent_helper import fill_registry_with_module
from src.core.checks_classes import checks_registry
from src.core.tools_classes import tools_registry

# -------------------
# MAIN
# -------------------

# A) Memory in die LLM-Reasoning einfließen lassen
# B) Confidence-Score für Drift einbauen
# C) Agent visualisieren (Graph-Architektur für Portfolio)
# D) System produktionsreif machen (Retry, Logging, Tests)
# Plugin-System für Tools
# YAML-basierte Tool-Parameter
# Multi-Datenquellen-Support
# Async-Tool-Execution
# Tool-Schema automatisch für OpenAI function-calling generieren


def run_audit():

    # (1) load config + parse arguments
    gh.load_env_vars()
    config_folder = os.getenv("CONFIG_PATH")
    config_path = Path(config_folder) / "audit_agent.yaml"
    config = fh.load_config(config_path)

    args = ah.parse_args()

    # (1.5) Check if report already exists
    ah.guard_report_creation(config, force=args.force)

    # (2) setup logger + engine
    # name= ""
    # logger = create_logger()

    # (3) setup_SQL_engine
    engine = post.get_engine(".env.audit_agent")

    # (4) load modules into 'tools_registry' + 'checks_registry'
    tools = "src.agent_audit.tools"  # = os.getenv("FOLDER_TOOLS")
    checks = "src.agent_audit.checks"  # os.getenv("FOLDER_CHECKS")

    for module in [
        f"{tools}.tools_general",
        f"{tools}.tools_ts",
        f"{checks}.checks_general",
        f"{checks}.checks_ts",
    ]:
        fill_registry_with_module(module)

    # (5) setup agentic context + agent
    context = AuditContext(engine, tools_registry, checks_registry)
    audit_agent = AuditAgent(context, config)

    # (6) define goal
    goal = "Audit weekly road accident aggregations (i.e. spatial-temporal analysis)."

    # (7) run agent
    state = audit_agent.run(goal)

    # (8) build report
    report = build_json_report(state.risk_summary, state.findings, state.llm_analysis)

    # (9) save report
    r_name = config["report_name"]
    save_report(report, r_name)

    # (10) comparison against previous version
    # previous = load_last_report()
    # comp_reports = compare_with_previous(report, previous)

    return


if __name__ == "__main__":
    run_audit()

#     findings = run_audit(engine, AUDIT_CONFIG)

# risk_summary = score_findings(findings, RISK_CONFIG)

# if risk_summary.risk_level.value in ["medium", "high", "critical"]:
#     llm_analysis = run_llm_reasoning(risk_summary)
# else:
#     llm_analysis = None
