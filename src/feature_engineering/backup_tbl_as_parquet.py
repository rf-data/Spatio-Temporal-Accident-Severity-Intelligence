# imports
import os
import pandas as pd
from sqlalchemy import text
from pathlib import Path
import gc
import pyarrow as pa
import pyarrow.parquet as pq

import src.utils.postgre_helper as post
import src.utils.general_helper as gh
import src.utils.path_helper as ph
from src.core.session import session
from src.core.logger import create_logger

engine = post.get_engine()
scheme = "accidents"

# load logger
log_name = "PARQUET_BACKUP"
name_logfile = "backup_tbl_as_parquet"
logger = create_logger(name=log_name, file_name=name_logfile)


def load_sql_table(table_name, chunked=False, chunk_size=500_000):
    # setup logger
    logger = session.logger

    # query
    query = f"""
    SELECT *
    FROM {scheme}.{table_name};
    """

    #
    logger.info(
        "Loading DB_table  %s %s.",
        table_name,
        f"chunked ({chunk_size} rows per chunk)" if chunked else "full",
    )

    if chunked:
        try:
            df_iter = pd.read_sql(text(query), engine, chunksize=chunk_size)
            return df_iter

        except Exception as e:
            logger.error("Error loading table %s: %s", table_name, str(e))
            return

    else:
        try:
            df = pd.read_sql(text(query), engine)
            return df

        except Exception as e:
            logger.error("Error loading table %s: %s", table_name, str(e))
            return


def save_chunkwise(df, file_path):
    # setup logger
    logger = session.logger

    writer = None

    for i, chunk in enumerate(df):
        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(f"{file_path}.parquet", table.schema)

        writer.write_table(table)

        logger.info("Saved chunk %s to %s", i + 1, ph.shorten_path(file_path))
        del chunk
        del table
        gc.collect()

    if writer:
        writer.close()

    return


def save_df_to_parquet(df, f_name, freq, res, chunked=False):
    # setup logger
    logger = session.logger

    output_folder = os.getenv("PATH_PROCESSED")
    # Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    inflate = session.inflate

    file_path = (
        Path(output_folder)
        / f"h3_dfs/{freq}{'_ZeroInf' if inflate else ''}_h3_res{res}"
    )

    if chunked:
        save_chunkwise(df, file_path)

    else:
        df.to_parquet(f"{file_path}.parquet", index=False)

    logger.info(
        "Saved %s → %s (%s)",
        f_name,
        ph.shorten_path(file_path),
        "chunked" if chunked else "full",
    )
    return


def backup_tbl_as_parquet():
    # load env variables
    gh.load_env_vars()

    # load logger
    log_name = "PARQUET_BACKUP"
    name_logfile = "backup_tbl_as_parquet"
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger
    session.resolution = [4, 5, 6, 7]
    session.freq = ["week"]  # "week"
    session.inflate = True

    for res in session.resolution:
        for freq in session.freq:
            name = f"h3_res{res}_{freq}_zeroinf"
            df = load_sql_table(name, chunked=True)

            if df is not None:
                save_df_to_parquet(df, name, freq, res, chunked=True)

            del df
            gc.collect()

    return


def drop_table():
    # setup logger
    # logger = session.logger

    #
    session.resolution = [4, 5, 6, 7]
    session.freq = ["month"]  # "week"
    session.inflate = False

    #
    for res in session.resolution:
        for freq in session.freq:
            name = f"h3_res{res}_{freq}_zeroinf"
            query = f"DROP TABLE IF EXISTS {scheme}.{name};"

            with engine.connect() as conn:
                try:
                    conn.execute(text(query))
                    conn.commit()
                    logger.info("Dropped table %s.%s", scheme, name)

                except Exception as e:
                    logger.error("Error dropping table %s: %s", name, str(e))

    return


if __name__ == "__main__":
    drop_table()
    # backup_tbl_as_parquet()

# # list all tables available in the scheme (here: 'accidents')
# SELECT table_name
# FROM information_schema.tables
# WHERE table_schema = 'accidents';

"""
 characteristics
 h3_res8_month_zeroinf
"""
