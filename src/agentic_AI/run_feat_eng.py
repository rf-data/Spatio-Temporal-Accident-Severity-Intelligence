## run_ts_processing.py
# import
import click
import os
from pathlib import Path

import src.utils.file_helper as fh
import src.utils.general_helper as gh
from src.agentic_AI.feature_engineering.run_time_processing import run_time_processing
from src.agentic_AI.feature_engineering.run_geo_processing import run_geo_processing
from src.agentic_AI.SQL_upload.load_to_sql import run_load_to_sql

# -------------------
# MAIN
# -------------------

@click.command()
# @click.option('--count', default=1, help='Number of greetings.')
@click.option("--name", prompt="Name of 'config_file'",
              help='The config_file to use.')
def run_feat_eng(name):
    # (1) load config + parse arguments
    gh.load_env_vars()

    config_folder = os.getenv("CONFIG_PATH")
    config_path = Path(config_folder) / f"{name}.yaml"
    config = fh.load_yaml_config(config_path)

    time_processing = config.get("time_processing", {})
    geo_processing = config.get("geo_processing", {})

    if time_processing.get("enabled"):
        run_time_processing(name)

    if geo_processing.get("enabled"):     
        run_geo_processing(name)

    # copy new features to SQL_DB
    run_load_to_sql(name)

    return 


if __name__ == "__main__":
    run_feat_eng()