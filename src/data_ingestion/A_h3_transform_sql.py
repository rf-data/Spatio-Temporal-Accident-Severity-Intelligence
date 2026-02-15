# H3 Index pro Unfall erzeugen
import pandas as pd
import h3
import io
import os
from pathlib import Path
from sqlalchemy import text
from typing import List

from src.core.session import session
from src.core.logger import create_logger
import src.utils.general_helper as gh
import src.postgre.postgre_helper as post
from configuration.H3_ZeroInflate_4to9_month import config

# Deine nächste Aktion
# Mach bitte genau diese drei Outputs (damit wir res & timebin finalisieren):
# H3_RES=7,8,9 jeweils:
# Anzahl unique H3-Zellen
# Median/95%/99% von accident_count auf (h3,tbin) (ohne Grid reicht)

# Mit data_full:
# Anteil accident_count==0
# data_full["accident_count"].describe()
# Sag mir: willst du daily oder hourly als erstes (ich empfehle daily, dann hourly als Phase 2).
# Jetzt brauchst du zeitliche Binning-Strategie.

# Beispiele:
# Variante	  Sinnvoll?
# year	      zu grob
# month	      gut
# week	      sehr gut
# weekday	    gut
# hour-bin 	  sehr gut
# (z.B. 6h)

#------------------
# HELPER FUNCTION
#------------------

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
  tmp_query = generate_tmp_tbl(tmp_table , cols_tmp)

  dst_tbl = "accidents.characteristics"
  update_query = generate_update_query(tmp_table, dst_tbl, cols)

  with engine.begin() as conn:
    conn.execute(text(tmp_query))
    logger.info("Created temporary table (name = %s)", 
                tmp_table)
    
    copy_data_to_pg(df, cols, conn, tmp_table)

    conn.execute(text(update_query))
    logger.info("Updated '%s' with table (name = %s)", 
                dst_tbl, 
                tmp_table)  
  return  


def generate_update_query(src_table,
                   dst_table, 
                   cols,  
                   p_key=None):
  # setup logger
  logger = session.logger

  # 
  cols_pre = [f"{col} = t.{col}," for col in cols]
  cols_new = " ".join(cols_pre)[:-1]

  if p_key is None:
    p_key = session.p_key

  # if src_table == "tmp":
  #   h3_values = session.h3_values
  #   src_table_name = session.tmp_tbl

  #   cols_tmp = [f"h3_res{val} TEXT," for val in h3_values]

  #   tmp_tbl_query = create_tmp_tbl(src_table_name, 
  #                                  cols_tmp,
  #                                   p_key=None)

  # else:
  #   src_table_name = src_table
      
  update = f"""
    UPDATE {dst_table} c
    SET
        {cols_new}
    FROM {src_table} t
    WHERE c.{p_key} = t.{p_key};
  """

  # with engine.begin() as conn:
  # conn.execute(text(update))

  return update


def generate_tmp_tbl(name, cols, p_key=None):
  # setup logger
  logger = session.logger

  # # # setup DB_engine
  # engine = post.get_engine()

  # check dtype
  if isinstance(cols, list):
    col_string = ", ".join(cols)  # [:-1]
  elif isinstance(cols, str):
    col_string = cols
  else:
    logger.error("'cols' is neither str nor list.\t--> %s", 
                   type(cols))
    raise TypeError

  if p_key is None:
    p_key = session.p_key

  tmp_tbl = f"""
    CREATE TEMP TABLE {name} (
      {p_key} BIGINT,
      {col_string}
      );
    """
  
  # with engine.begin() as conn:
  #   conn.execute(text(tmp_tbl))
  #   logger.info("Created temporary table (name = %s)", 
  #               name)
    
  return tmp_tbl
    

def copy_data_to_pg(df, cols, conn, table=None):
  # setup logger
  logger = session.logger
  h3_values = session.h3_values

  # 
  # if table is None:
  #   scheme = os.getenv("TABLE_SCHEME")    # "accidents"
  #   table_name = os.getenv("TABLE_NAME")  # "characteristics"
  #   table = f"{scheme}.{table_name}"

  logger.info("Copying data to DB_table '%s' \nshape = %s \ncolumns = %s",
              table, 
              df.shape,
              df.columns)
  
  buffer = io.StringIO()
  df.to_csv(
        buffer,
        index=False,
        header=False,
        sep="\t",
        na_rep="\\N"
    )
  buffer.seek(0)

  # 
  cur = conn.connection.cursor()

  if isinstance(cols, str):
    col_str = cols
  
  elif isinstance(cols, List):
    col_str = ", ".join(cols)
    
  else:
    logger.error("'cols' is neither str nor list.\t--> %s", 
                   type(cols))
    raise TypeError
    
  cur.copy_expert(
        f"""
        COPY {table} (
            {col_str}
        )
        FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')
        """,
        buffer
    )
  
  # conn.commit()
  # cur.close()
  # conn.close()
  
  return conn


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


def create_crosstable(value, freq, replace=False):
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
    CREATE TABLE {scheme}.{table_new} AS
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
  
  return 


def create_h3_grid(df_in, h3_value: int | List):
  # setup logger
  logger = session.logger
  logger.info("Start creating h3 geo_grid (H=%s)", h3_value)

  # (1) Nur valide Geo
  df = df_in[df_in["lat_norm"].notna() & df_in["lon_norm"].notna()].copy()

  # (2) 
  # vektorisieren oder via itertuples() beschleunigen
  if isinstance(h3_value, int):
    h3_value = [h3_value]

  for val in h3_value:
    df[f"h3_res{val}"] = df.apply(lambda r: h3.latlng_to_cell(r["lat_norm"], r["lon_norm"], val), axis=1)

  return df

  # (9) Was kommt als erstes Modell?
  # Modelle:
  # Poisson / Negative Binomial
  # XGBoost / LightGBM
  # Temporal features + Context features

  # --> LightGBM / XGBoost auf log1p(count) oder direkt count (mit Poisson objective, wenn verfügbar)


def check_table(engine, schema, table_name):
  # setup logger
  logger = session.logger

  # 
  from sqlalchemy import inspect
  
  inspector = inspect(engine)
  tables = inspector.get_table_names(schema=schema)

  if table_name not in tables:
    logger.error(f"❌ Table {schema}.{table_name} does NOT exist.")
    return
    
  df = pd.read_sql(
        f"SELECT COUNT(*) as n FROM {schema}.{table_name}",
        engine
    )
  logger.info(f"✅ Table {schema}.{table_name} exists.")
  logger.info(f"Rows: {df.loc[0, 'n']}")


def zero_inflate_data(value, freq, src_tbl, replace=False):
  # setup logger
  logger = session.logger
  logger.info("Start zero-inflating in table '%s' (H3_RES=%s; freq=%s)", 
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
  table_new = f"h3_res{value}_{freq}_ZeroInf"

  msg_join = []

  if replace:
    msg_join.append(f"DROP TABLE IF EXISTS {scheme}.{table_new};")
    
  msg_join.append(f"""
      CREATE TABLE IF NOT EXISTS {scheme}.{table_new} AS 
      SELECT
          g.h3_index,
          g.{freq}_start,
          COALESCE(c.n_accidents, 0) AS n_accidents
      FROM grid g
      LEFT JOIN {scheme}.h3_res{value}_{freq} c
      ON g.h3_index = c.h3_index
      AND g.{freq}_start = c.{freq}_start;
      """)
  l_join = "\n".join(msg_join)

  # setup engine
  engine = post.get_engine()

  with engine.begin() as conn:
    conn.execute(text(active))
    conn.execute(text(time_range))
    conn.execute(text(cartesian))
    conn.execute(text(l_join))   

  logger.info("Finished zero-inflation and created new table '%s.%s'",
              scheme,
              table_new) 
  
  # check
  check_table(engine, scheme, table_new)

  return

#------------------
# MAIN FUNCTION
#------------------

def h3_transformation():
  # load env variables
  gh.load_env_vars()

  session.load_config(config)
  log_name = session.log_name # "ETL_CHARACTERISTICS"
  name_logfile = session.log_file # "etl_characteristics"

  # load logger
  logger = create_logger(name=log_name,
                           file_name=name_logfile)

  session.logger = logger

  # (0) load data + reduced to necessary columns 
  # cols = session.cols_needed
  
  # f_path = f"/home/robfra/0_Portfolio_Projekte/Road_accidents/data/data_processed/df_character_norm.csv"

  # df_pre = pd.read_csv(f_path,
  #                   low_memory=False)
  # df = df_pre[cols].copy()

  # (1) apply geo_grid (h3)
  h3_values = session.h3_values

  # if h3_values is None:
  #   logger.error("'H3_values' is None.")
  #   raise ValueError
  
  # df_h3 = create_h3_grid(df, h3_values)

  # # save dfs
  # folder = os.getenv("PATH_PROCESSED")
  # df_path = f"{folder}/df_h3.csv"

  # df_h3.to_csv(df_path)
  # logger.info("df_h3 saved")

  # # (2) add cols in SQL_db
  # # folder = os.getenv("PATH_PROCESSED")
  # # h3_path = f"{folder}/df_h3.csv"
  # # df_h3 = pd.read_csv(h3_path)
  # add_h3_cols(h3_values)

  # logger.info("Shape of df_h3: %s\n", df_h3.shape)
  # fill_h3_columns(df_h3)

  freqence = session.freq 
  if not isinstance(freqence, list):
    freqence = [freqence]
    logger.info("Converted freqence to list:\t%s (dtype = %s)", 
                freqence, 
                type(freqence))
  
  src_tbl=session.src_table
  inflate = session.inflate

  for value in h3_values:
    for freq in freqence:
      if inflate:
        zero_inflate_data(value, freq, src_tbl, replace=False)
      else:
        create_crosstable(value, freq, replace=True) 
      

if __name__ == "__main__":
  h3_transformation()


  # 
  # msg = f"""
  #   CREATE TABLE {scheme}.h3_res{value}_{freq}_full AS
  #   SELECT
  #       h.h3_res{value},
  #       t.{freq}
  #   FROM all_h3 h
  #   CROSS JOIN all_{days} t;
  # """

# def create_time_index(freq):
#   # setup logger
#   logger = session.logger

#   # setup SQL_engine 
#   engine = post.get_engine()

#   # query
#   idx = f"""
#     CREATE TEMP TABLE all_{freq} AS
#     SELECT generate_series(
#         (SELECT MIN(date(datetime)) FROM {table}),
#         (SELECT MAX(date(datetime)) FROM {table}),
#         interval '1 {freq}'
#     )::date AS {freq};
#     """
  
#   # 
#   with engine.begin() as conn:
#     conn.execute(text(idx))

#   return 

# def load_sql_data(query):
#   engine = post.create_engine("postgresql+psycopg2://...")
#   df = pd.read_sql(query, engine)
#   return df


# def join_counts():
#   add = f"""
#     ALTER TABLE accidents.h3_res8_daily_full
#     ADD COLUMN n_accidents INTEGER DEFAULT 0;
#     """
  
#   fill = f"""
#     UPDATE accidents.h3_res8_daily_full f
#     SET n_accidents = c.n_accidents
#     FROM accidents.h3_res8_daily c
#     WHERE f.h3_res8 = c.h3_res8
#       AND f.day = c.day;
#     """

#   # 
#   with engine.begin() as conn:
#     conn.execute(text(idx))

#   return 
  

# def create_aggregation_tbl(name):
#   # setup logger
#   logger = session.logger

#   # setup SQL_engine 
#   engine = post.get_engine()

#   # 
#   scheme = os.getenv("TABLE_SCHEME")    # "accidents"
#   table = f"{scheme}.{name}"

#   tbl_dict = {
#       "h3_index": "TEXT",
#       "year": "SMALLINT",
#       "month": "SMALLINT",
#       "n_accidents": "INTEGER",
#       "PRIMARY KEY": "(h3_index, year, month)"
#       }
  
#   cols = [f"{key} {value}," for key, value in tbl_dict.items()]
#   cols_joined = ", ".join(cols)

#   creation = f"""
#     CREATE TABLE {table} (
#       {cols_joined}
#     );
#     """
  
#   return 

  # # (1) define time axis
  # time_axis = """
  # CREATE TEMP TABLE all_days AS
  # SELECT generate_series(
  #     (SELECT MIN(date(datetime)) FROM accidents.characteristics),
  #     (SELECT MAX(date(datetime)) FROM accidents.characteristics),
  #     interval '1 day'
  # )::date AS day;
  # """
  
  # # (2)
  # h3_value = ""

  

  # h3_x_tbl = f"""
  # CREATE TABLE accidents.h3_accidents_res{h3_value} (
  #     h3_res{h3_value} TEXT,
  #     year SMALLINT,
  #     n_accidents INTEGER,
  #     PRIMARY KEY (h3_res{h3_value}, year)
  # );
  # """

  # h3_x_idx = f"""
  # CREATE INDEX idx_h3_res{h3_value}
  # ON accidents.h3_accidents_res{h3_value} (h3_res{h3_value});
  # """
  
  # h3_d_tbl = f"""
  # CREATE TEMP TABLE all_h3 AS
  # SELECT DISTINCT h3_res{h3_value}
  # FROM accidents.h3_res{h3_value}_daily;
  # """



# def extract_h3_cells(value, freq):
#   # setup logger
#   logger = session.logger

#   # setup SQL_engine 
#   engine = post.get_engine()

#   hi = f"""
#     CREATE TEMP TABLE all_h3_{freq} AS
#     SELECT DISTINCT h3_res{value}
#     FROM {scheme}.h3_res{value}_{freq};
#     """

#   # 
#   with engine.begin() as conn:
#     conn.execute(text(idx))

#   return 


  # #   df = load_sql_data(h3_query)
  # f_path = f"/home/robfra/0_Portfolio_Projekte/Road_accidents/data/data_processed/df_character_norm.csv"
  # df = pd.read_csv(f_path,
  #                   low_memory=False)

  # # (2) creating time_bin_column
  # df_tbin = time_binning(df, freq="D")

  # # (3) creating h3_grid
  # # if h3_test == True: 
  # #   h3_values = session.h3_values
  # #   h3_grid_test(h3_values)
  # grid_dict = {}
  # for h3_value in [7]:  # , 8, 9
  #   logger.info("Using H3_resolution=%s", h3_value)
  #   df_h3 = create_h3(df_tbin, h3_value)
  #   grid_dict[f"RES_{h3_value}"] = df_h3

  # for name, df in grid_dict.items():
  #   # (4) compile accidents per 'h3 x tbin'
  #   logger.info("Getting accident's incidence (%s)",
  #               name)
  #   df_2 = get_accidents_incidence(df)

  #   # (5) Before_Sparsity_check on dfs
  #   logger.info("BEFORE INFLATION -- Sparsity check on %s", 
  #               name)
  #   sparsity_check(df_2)

  #   # (6) Zero Inflation
  #   df_zero = zero_inflation(df, df_2)

  #   # (7) After_Sparsity_check on dfs
  #   logger.info("AFTER INFLATION -- Sparsity check on %s", 
  #               name)
  #   sparsity_check(df_zero)
    
  
#   
# ,id,
# year,month,day,
# hour,time_clean, weekday,is_weekend,
# light conditions,localisation,intersection type,
# weather,collision type,
# commune,department,
# lat_norm,lon_norm
  # (5) Feature-Aggregation (aus deinen vorhandenen Spalten)
  # # (A) Zeitfeatures direkt aus tbin
  # g["weekday"] = g["tbin"].dt.weekday
  # g["is_weekend"] = (g["weekday"] >= 5).astype(int)
  # g["month"] = g["tbin"].dt.month
  # g["year"] = g["tbin"].dt.year
"""
RES = 7:
rows:        595_390
zero_share:  0.0

count    595390.000000
mean          1.059479
std           0.283725
min           1.000000
25%           1.000000
50%           1.000000
75%           1.000000
max           9.000000
  
RES = 8:
rows:        622_801
zero_share:  0.0
  
count    622801.000000
mean          1.012848
std           0.116323
min           1.000000
25%           1.000000
50%           1.000000
75%           1.000000
max           4.000000

RES = 9:
rows:        628_393
zero_share:  0.0

count    628393.000000
mean          1.003835
std           0.062628
min           1.000000
25%           1.000000
50%           1.000000
75%           1.000000
max           4.000000

"""