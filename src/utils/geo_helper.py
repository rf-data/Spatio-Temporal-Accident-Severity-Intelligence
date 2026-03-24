## geo_helper.py
# imports
from sqlalchemy import text
import pandas as pd
import os

import src.utils.file_helper as fh
import src.utils.postgre_helper as post
from src.core.session import session


def zero_inflate_data(value, 
                      freq, 
                      src_tbl,
                      as_table=False, 
                      replace=False):
  # setup logger
  logger = session.logger
  logger.info("Start zero-inflating data from table '%s' (H3_RES=%s; freq=%s)", 
              src_tbl, 
              value, 
              freq)

  # load variable from session
  scheme = session.scheme

  # (1) determine all 'active' h3 cells (>= 1 accident)
  active = f"""
      CREATE TEMP TABLE active_h3 AS
      SELECT DISTINCT h3_res{value} AS h3_index
      FROM {scheme}.{src_tbl}
      WHERE h3_res{value} IS NOT NULL;
      """

  # (2) get time_idx (range of time_bins)
  time_range = f"""
      CREATE TEMP TABLE {freq} AS
      SELECT generate_series(
          (SELECT MIN(date_trunc('{freq}', datetime)) FROM {scheme}.{src_tbl}),
          (SELECT MAX(date_trunc('{freq}', datetime)) FROM {scheme}.{src_tbl}),
          interval '1 {freq}'
      ) AS {freq}_start;
      """
  
  # (3) cartesian product (active h3 x time_bins)
  cartesian = f"""
      CREATE TEMP TABLE grid AS
      SELECT
          h.h3_index,
          m.{freq}_start
      FROM active_h3 h
      CROSS JOIN {freq} m;
      """
  
  # (4) left join with actual data
  table_src = f"h3_res{value}_{freq}"
  table_new = f"h3_res{value}_{freq}_ZeroInf"

  msg_join = []

  if replace:
    msg_join.append(f"DROP TABLE IF EXISTS {scheme}.{table_new};")

  if as_table:
    msg_join.append(f"CREATE TABLE {scheme}.{table_new} AS")
  else:
    msg_join.append(f"CREATE TEMP TABLE {table_new} AS")

  msg_join.append(f"""
      SELECT
          g.h3_index,
          g.{freq}_start,
          COALESCE(c.n_accidents, 0) AS n_accidents
      FROM grid g
      LEFT JOIN {scheme}.{table_src} c
      ON g.h3_index = c.h3_index
      AND g.{freq}_start = c.{freq}_start;
      """)
  
  l_join = "\n".join(msg_join)
  
  # 
  get_table = f"SELECT * FROM {table_new}"

  # setup engine
  engine = post.get_engine()

  with engine.begin() as conn:
    conn.execute(text(active))
    conn.execute(text(time_range))
    conn.execute(text(cartesian))
    conn.execute(text(l_join))   

    if not as_table:
      logger.info("Finished zero-inflation and start creating parquet file for '%s.%s'",
                  scheme,
                  table_new)
      
      result = conn.execute(text(get_table))
      df = pd.DataFrame(result.fetchall(), columns=result.keys())

      f_name = f"h3_dfs/h3_res{value}_{freq}_ZeroInf"
        fh.save_df_to_parquet(df, 
                       f_name, 
                       chunked=True)
      
    else:
      conn.close()
      logger.info("Finished zero-inflation and created new table '%s.%s'",
                  scheme,
                  table_new) 
      
      # check
      post.check_table(engine, scheme, table_new)

  return


def fill_h3_columns(df_in):
  # setup logger
  logger = session.logger

  # setup engine
  engine = post.get_engine()
  p_key = session.p_key
  h3_values = session.h3_values

  cols = [f"h3_res{val}" for val in h3_values]
  cols.append(p_key)

  df_in = df_in.rename(columns={"id": "id_accid"}) 
  df = df_in[cols].copy()
  
  tmp_table = "tmp_h3"
  cols_tmp = [f"h3_res{val} TEXT" for val in h3_values]
  tmp_query = post.generate_tmp_tbl(tmp_table , cols_tmp)

  dst_tbl = "accidents.characteristics"
  update_query = post.generate_update_query(tmp_table, dst_tbl, cols)

  with engine.begin() as conn:
    conn.execute(text(tmp_query))
    logger.info("Created temporary table (name = %s)", 
                tmp_table)
    
    post.copy_data_to_pg(df, cols, conn, tmp_table)

    conn.execute(text(update_query))
    logger.info("Updated '%s' with table (name = %s)", 
                dst_tbl, 
                tmp_table)  
  return  


def add_h3_cols(h3_values):
  # setup logger
  logger = session.logger

  # setup SQL_engine 
  engine = post.get_engine()

  scheme = os.getenv("TABLE_SCHEME")    # "accidents"
  table_name = os.getenv("TABLE_NAME")  # "characteristics"
  table = f"{scheme}.{table_name}"

  for val in h3_values:
    h3_col = f"""
    ALTER TABLE {table}
    ADD COLUMN IF NOT EXISTS h3_res{val} TEXT;
    """

    with engine.begin() as conn:
        conn.execute(text(h3_col))
        logger.info("Added column 'h3_res%s' to table '%s'", 
                    val, 
                    table)

  return


def create_crosstable(value, 
                      freq,  
                      replace=False,
                      also_parquet=False):
  # setup logger
  logger = session.logger

  # setup SQL_engine 
  engine = post.get_engine()

  # 
  scheme = session.scheme
  src_tbl = session.src_table
  table_new = f"h3_res{value}_{freq}"

  queries = []

  if replace:
    queries.append(f"DROP TABLE IF EXISTS {scheme}.{table_new};")

  queries.append(f"""
    CREATE TABLE IF NOT EXISTS {scheme}.{table_new} AS
    SELECT
      h3_res{value} AS h3_index,
      date_trunc('{freq}', datetime)::date AS {freq}_start,
      COUNT(*) AS n_accidents
    FROM {scheme}.{src_tbl}
    WHERE h3_res{value} IS NOT NULL
    GROUP BY 1, 2;
  """)

  msg = "\n".join(queries)
  # 
  with engine.begin() as conn:
    conn.execute(text(msg))

    logger.info("Created new cross_table '%s' in '%s'", 
              table_new,
              scheme)
  
  # check
  check_table(engine, scheme, table_new)
    
  if also_parquet:  
    df = pd.read_sql(f"SELECT * FROM {scheme}.{table_new}", 
                       engine,
                       chunksize=500_000)

    save_df_to_parquet(df, 
                       table_new, 
                       chunked=True)
    
    # logger.info("Saved CrossTable also as parquet file ('%s' in ...%s).",
    #             f_name,
    #             ph.shorten_path(f_path))


  return 
