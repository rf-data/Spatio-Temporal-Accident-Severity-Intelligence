## run_sql_upload.py
# import
import click

import src.utils.file_helper as fh

from src.agentic_AI.SQL_upload.harmonize_schema import run_harmonize_schema
from src.agentic_AI.SQL_upload.clean_from_json import run_clean_from_json
from src.agentic_AI.SQL_upload.load_to_sql import run_load_to_sql


@click.command()
@click.option("--name", required=True)
def run_sql_upload(name):

    # config = fh.get_yaml_config(name)

    run_harmonize_schema(name)
    run_clean_from_json(name)
    run_load_to_sql(name)

    return 

if __name__ == "__main__":
    run_sql_upload()
