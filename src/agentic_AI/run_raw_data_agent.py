## run_raw_data_agent.py
# imports
import os
from pathlib import Path

import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.agent_helper as ah

import src.agentic_AI.tools as tools_pkg
import src.agentic_AI.checks as checks_pkg

import src.agentic_AI.report.generate_raw_data_scripts as scr
import src.agentic_AI.report.raw_data_report as rep

# from src.utils.agent_helper import fill_registry_with_module
from src.core.agent_classes import RawDataAgent, RawDataContext
from src.core.checks_classes import checks_registry
from src.core.tools_classes import tools_registry


# -------------------
# MAIN
# -------------------

import click

@click.command()
# @click.option('--count', default=1, help='Number of greetings.')
@click.option("--name", prompt="Name of 'config_file'",
              help='The config_file to use.')
def run_eda_agent(name):
    # (1) load config + parse arguments
    gh.load_env_vars()

    config_folder = os.getenv("CONFIG_PATH")
    config_path = Path(config_folder) / f"{name}.yaml"
    config = fh.load_yaml_config(config_path)

    args = ah.parse_args()

    # (1.5) Check if report already exists
    ah.guard_report_creation(config, 
                             report=False, 
                             force=args.force)

    # (2) setup logger + engine
    # name= ""
    # logger = create_logger()

    # (3) load modules into 'tools_registry' + 'checks_registry'
    ah.load_modules(tools_pkg)
    ah.load_modules(checks_pkg)
    

    # (4) setup agentic context + agent
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
    print("Initiating agent")
    agent.load_data()
    state = agent.run(goal)
    eda_params = agent.arguments

    # (7) build and save summary
    print("Start building summary")
    rep.build_and_save_summary(
                        state.preparation_summary, 
                        name=eda_params.summary_name,
                        json_only=True
                        )
    
    # (8) extract merge + data processing strategy

    # (8.2) build sripts
    # clean_script = scr.generate_cleaning_script(state.preparation_summary)
    # merge_script = scr.generate_merge_script(state.preparation_summary)
    # sql_query = scr.generate_sql_query(state.preparation_summary)

    # (9) save scripts
    # now = state.preparation_summary.metadata.get("analysis_timestamp", "")
    # pre_name = state.preparation_summary.metadata.get("summary_name", "")

    # for name, file in zip(["cleaning", "merging", "sql_query"],
    #                     [clean_script, merge_script, sql_query]):
        
    #     scr_name = f"{now}_{name}_{pre_name}"
    #     scr.save_script(file, scr_name)


if __name__ == "__main__":
    run_eda_agent()