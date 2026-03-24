## load_to_sql.py
# import
import pandas as pd
import io
from sqlalchemy import text
import os
from pathlib import Path
import click

import src.utils.file_helper as fh
import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.postgre_helper as post

from src.agentic_AI.report.raw_data_report import load_eda_summary

import logging


def normalize_integer_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            continue

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .round()
            .astype("Int64")
        )

    return df

def copy_from_upload(df, engine, schema, table):

    print("df length:\t", len(df))

    buffer = io.StringIO()

    df.to_csv(buffer, 
              index=False, 
              header=False, 
              sep="\t", 
              na_rep="\\N")
    
    buffer.seek(0)

    cols = ", ".join(df.columns)

    conn = engine.raw_connection()
    cur = conn.cursor()

    sql = f"""
        COPY {schema}.{table} ({cols})
        FROM STDIN 
        WITH (
            FORMAT text, 
            DELIMITER E'\t', 
            NULL '\\N'
            )
        """

    cur.copy_expert(sql, buffer)

    conn.commit()
    cur.close()
    conn.close()

    return 


# ------------------------------
# WRAPPER FUNCTION
# ------------------------------
@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def load_to_sql(name):
    run_load_to_sql(name)

    return 


# ------------------------------
# MAIN FUNCTION
# ------------------------------
def run_load_to_sql(name):
    # set SQL_logger to 'warning_level'
    # logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # (1) load config + parse arguments
    gh.load_env_vars()

    data_processed = os.getenv("PATH_PROCESSED")

    config = fh.get_yaml_config(name)

    arg_dict = config.get("general_args", {})

    data_folder = Path(arg_dict.get("data_folder"))
    file_suffix = arg_dict.get("file_suffix", ["clean"])
    df_folder = Path(f"{data_processed}/{data_folder}")

    # load report
    report_dict = load_eda_summary(config)
    files = report_dict["files"]

    # loading df_files
    feat_eng_config = config.get("Feat_Eng", {})
    schema = feat_eng_config.get("schema_name", None)
    schema_version = feat_eng_config.get("schema_version", None)
    table = feat_eng_config.get("table_name", None)
    sql_dtypes = feat_eng_config.get("sql_dtypes", {})
    col_to_keep = feat_eng_config.get("to_keep", [])
    prim_key = feat_eng_config.get("primary_key", None)
    int_cols = feat_eng_config.get("numeric", None)
    drop_exist_tbl = feat_eng_config.get("drop_existing_table", False)

    if table is None or schema is None or prim_key is None:
        raise ValueError(f"""
            No valid name for 'primary_key' ({prim_key}), 
            'table' ({table}) or 'schema' ({schema}).
            """)

    # determine schema and table if necessary
    engine = post.get_engine()

    intro = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
        """
    outro = ");"
    query = []

    for col, dtype in sql_dtypes.items():
        if col == prim_key:
            query.append(f"{col} {dtype} PRIMARY KEY,")
        else:
            query.append(f"{col} {dtype},")

    query_str = "\n".join(query)
        
    msg = intro + "\n" + query_str[:-1] + "\n" + outro

    with engine.begin() as conn:
        # create SQL schema
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))

        # 
        if drop_exist_tbl:
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table}"))

        # create SQL table
        conn.execute(text(msg))

        # create idx
        conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_id ON {schema}.{table}({prim_key});
                    """))

    
    # upload files to sql_db
    if isinstance(file_suffix, str):
        file_suffix = [file_suffix]

    for file in files:
        f_name = Path(file).stem

        df_list = []
        for f_suf in file_suffix: 
            f_path = f"{df_folder}/{str(f_name).strip()}_{f_suf}.parquet"
            print("Start loading:", ph.shorten_path(f_path))

            try:
                df = pd.read_parquet(f_path)
                df_list.append(df)
            except FileNotFoundError:
                print(f"File not found:", f_path)

        # merge dfs
        if len(df_list) == 1:
            df_merge = pd.DataFrame(df_list[0])

        elif len(df_list) >= 2:
            df_merge = df_list[0].merge(df_list[1], on="ID_accident", how="left")

            if len(df_list) > 2:
                for i in range(2, len(df_list)+1):
                    df_merge = df_merge.merge(
                                        df_list[i], 
                                        on="ID_accident", 
                                        how="left")

        else:
            print(f"Invalid count of dfs ({file}):", len(df_list))
            continue

        print("Completed df merge")

            # df_merge = "to_be_continued_soon"

            # print(f"[DTYPE_CHECK] ({ph.shorten_path(f_path)}):\n", df.dtypes)
            # print()
            # print(f"[NAN_CHECK] ({ph.shorten_path(f_path)}):\n", df.isna().sum())
            # print()
            # print(f"[NUNIQUE_CHECK] ({ph.shorten_path(f_path)}):\n", df.nunique())

        # filter out unnecessary columns
        cols_needed = [col for col in df_merge.columns if col in col_to_keep]
        df_filt = df_merge[cols_needed].copy()

        if int_cols:
            df_filt = normalize_integer_columns(df_filt, int_cols)

        df_filt["feature_set"] = schema_version
        copy_from_upload(df_filt, engine, schema, table)
        print("Loading completed:", ph.shorten_path(f_path))

        # check after SQL_upload
        query_1 = f"SELECT COUNT(*) FROM {schema}.{table};"
        query_2 = f"""
                SELECT COUNT(DISTINCT {prim_key})
                FROM {schema}.{table};
                """
        with engine.begin() as conn:
            count_all = conn.execute(text(query_1)).scalar()
            count_dist = conn.execute(text(query_2)).scalar()

        print("Validation after sql_upload")
        print("Rows:", count_all)
        print("Unique PK:", count_dist)

    # add 'GEOGRAPHY' col
    # if geo_col:
    #     msg_1 = f"""
    #         ALTER TABLE {schema}.{table}
    #         ADD COLUMN geom GEOGRAPHY(Point, 4326);
    #         """
        
    #     if len(geo_col) == 2:
    #         # conversion since lat + lon in Lambert 93 (EPSG:2154)
    #         msg_2 = f"""
    #             UPDATE {schema}.{table}
    #             SET geom = ST_SetSRID(ST_MakePoint({geo_col}),4326);
    #             """
    #     else:
    #         msg_2 = f"""
    #             UPDATE {schema}.{table}
    #             SET geom = ST_SetSRID(ST_MakePoint({geo_col}),4326);
    #             """

if __name__ == "__main__":
    load_to_sql()