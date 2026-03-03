## run_raw_data_agent.py
# imports
import os
from pathlib import Path

import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.agent_helper as ah
import src.agentic_AI.report.generate_raw_data_scripts as scr
import src.agentic_AI.report.raw_data_report as rep

from src.utils.agent_helper import fill_registry_with_module
from src.core.agent_classes import RawDataAgent, RawDataContext
from src.core.checks_classes import checks_registry
from src.core.tools_classes import tools_registry


# -------------------
# MAIN
# -------------------
def run_eda_agent():
    # (1) load config + parse arguments
    gh.load_env_vars()
    config_folder = os.getenv("CONFIG_PATH")
    config_path = Path(config_folder) / "raw_data_agent.yaml"
    config = fh.load_config(config_path)

    args = ah.parse_args()

    # (1.5) Check if report already exists
    ah.guard_report_creation(config, report=False, force=args.force)

    # (2) setup logger + engine
    # name= ""
    # logger = create_logger()

    # (3) load modules into 'tools_registry' + 'checks_registry'
    tools = "src.agentic_AI.tools"  # = os.getenv("FOLDER_TOOLS")
    checks = "src.agentic_AI.checks"

    for module in [f"{tools}.tools_tab_EDA", f"{checks}.checks_tab_EDA"]:
        fill_registry_with_module(module)

    # (4) setup agentic context + agent
    default_tools = ""
    default_checks = ""
    context = RawDataContext(tools_registry, checks_registry)
    agent = RawDataAgent(context, config)

    # (5) define goal
    goal = """
    Conduct an exploratory analysis on data of road accidents 
    (i.e. spatial-temporal analysis). Make recommendations what 
    to do regarding feature engineering and provide the respective
    scripts (as .py-files). 
    """

    # (6) load files + run agent
    agent.load_data()
    state = agent.run(goal)
    eda_params = agent.arguments

    # (7.1) build and save summary
    rep.build_and_save_summary(
                        state.preparation_summary, 
                        name=eda_params.summary_name
                        )
    
    # (7.2) build sripts
    clean_script = scr.generate_cleaning_script(state.preparation_summary)
    merge_script = scr.generate_merge_script(state.preparation_summary)
    sql_query = scr.generate_sql_query(state.preparation_summary)

    # (8) save files
    now = state.preparation_summary.metadata.get("analysis_timestamp", "")
    pre_name = state.preparation_summary.metadata.get("summary_name", "")

    for name, file in zip(["cleaning", "merging", "sql_query"],
                        [clean_script, merge_script, sql_query]):
        
        scr_name = f"{now}_{name}_{pre_name}"
        scr.save_script(file, scr_name)


if __name__ == "__main__":
    run_eda_agent()