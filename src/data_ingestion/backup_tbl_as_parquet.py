# imports
import os
import pandas as pd
from sqlalchemy import text
from pathlib import Path
import gc
import pyarrow as pa
import pyarrow.parquet as pq

import src.postgre.postgre_helper as post
import src.utils.general_helper as gh
import src.utils.path_helper as ph
from src.core.session import session
from src.core.logger import create_logger

engine = post.get_engine()
scheme = "accidents"

def load_sql_table(table_name, 
                   chunked=False,
                   chunk_size=500_000
                   ): 
    # setup logger
    logger = session.logger

    # query
    query = f"""
    SELECT *
    FROM {scheme}.{table_name}
    WHERE year >= 2018;
    """

    # 
    logger.info("Loading DB_table  %s %s.",  
                table_name, 
                f"chunked ({chunk_size} rows per chunk)" if chunked else "full")
    
    if chunked:
        df_iter = pd.read_sql(text(query), 
                              engine, 
                              chunksize=chunk_size)
        return df_iter

    else:
        df = pd.read_sql(text(query), 
                         engine)
        return df


def save_chunkwise(df, file_path):
    # setup logger
    logger = session.logger
    
    writer = None

    for i, chunk in enumerate(df):
        table = pa.Table.from_pandas(chunk)
        
        if writer is None:
            writer = pq.ParquetWriter(f"{file_path}_to2018.parquet", table.schema)
        
        writer.write_table(table)
        
        logger.info("Saved chunk %s to %s",
                    i+1,
                    ph.shorten_path(file_path))
        del chunk
        del table
        gc.collect()

    if writer:
        writer.close()

    return 


def save_df_to_parquet(df, 
                       f_name, 
                       chunked=False):
    # setup logger
    logger = session.logger
    
    output_folder = os.getenv("PATH_PROCESSED")
    # Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    file_path = Path(output_folder) / f"h3_dfs/{f_name}"

    if chunked:
        save_chunkwise(df, file_path)
        
    else:
        df.to_parquet(f"{file_path}.parquet", index=False)


    logger.info("Saved %s → %s (%s)", 
                f_name, 
                ph.shorten_path(file_path), 
                "chunked" if chunked else "full")


def backup_tbl_as_parquet():
    # load env variables
    gh.load_env_vars()

    # load logger
    log_name = "PARQUET_BACKUP"
    name_logfile = "backup_tbl_as_parquet"
    logger = create_logger(name=log_name,
                            file_name=name_logfile)

    session.logger = logger

    for res in [8]:
        name = f"h3_res{res}_month_zeroinf"
        df = load_sql_table(name, chunked=True)
        save_df_to_parquet(df, name, chunked=True)

        del df
        gc.collect()

    return 

if __name__ == "__main__":
    backup_tbl_as_parquet()


