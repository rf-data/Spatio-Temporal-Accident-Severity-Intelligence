# H3 Index pro Unfall erzeugen
import pandas as pd
import h3
import os
from pathlib import Path

from src.core.session import session
from src.core.logger import create_logger
import src.utils.general_helper as gh
import src.postgre.postgre_helper as post
from configuration.H3_6to9_month_from_charact import config

#------------------
# HELPER FUNCTION
#------------------

def load_sql_data(query):

  engine = post.create_engine("postgresql+psycopg2://...")
  df = pd.read_sql(query, engine)

  return df


def create_h3(df_in, h3_value):
  # setup logger
  logger = session.logger
  logger.info("Start creating h3 geo_grid (H=%s)", h3_value)

  # (1) Nur valide Geo
  df = df_in[df_in["lat_norm"].notna() & df_in["lon_norm"].notna()].copy()

  # (2) 
  # vektorisieren oder via itertuples() beschleunigen
  # H3_RES = 8
  df["h3"] = df.apply(lambda r: h3.latlng_to_cell(r["lat_norm"], r["lon_norm"], h3_value), axis=1)

  return df


def time_binning(df_in, freq="D"):
  # setup logger
  logger = session.logger

  if "datetime" not in df_in.columns:
    logger.error("No 'datetime' column in df")
    raise KeyError
   
  logger.info("Start creating time bins (freq=%s)", freq)

  # ensure correct dtype + filter NaN's
  df_in["dt"] = pd.to_datetime(df_in["datetime"], errors="coerce")
  df = df_in[df_in["dt"].notna()].copy()

  # apply binning
  df["tbin"] = df["dt"].dt.floor(freq)

  # Alternative: Stunden-Bin
  # df["tbin"] = df["dt"].dt.floor("H")

  return df


def get_accidents_incidence(df):
  # setup logger
  logger = session.logger
  logger.info("Start compiling accidents per 'h3 x tbin'")

  # (4) Zielvariable bauen: Unfallhäufigkeit pro H3×Zeit)
  g = (
      df.groupby(["h3", "tbin"], as_index=False)
        .agg(accident_count=("id", "size"))
      )

  feat = ohe_cat_feats(df)

  # Merge Features + Target
  data = g.merge(feat, on=["h3", "tbin"], how="left")

  return data

def ohe_cat_feats(df):
  # (B) Kontextfeatures aus Original-DF aggregieren
  cat_cols = session.cat_feats

  tmp = df[["h3", "tbin"] + cat_cols].copy()

  # One-hot für jede Kategorie und dann Mittelwert => Anteil pro Klasse
  tmp_oh = pd.get_dummies(tmp, columns=cat_cols, dummy_na=True)

  feat = (
      tmp_oh.groupby(["h3", "tbin"], as_index=False)
            .mean(numeric_only=True)
          )
  
  return feat

def sparsity_check(df):
  # setup logger
  logger = session.logger
  logger.info("Checking sparsity")

  # (6) Sparsity-Check (entscheidet res & timebin)
  zero_share = (df["accident_count"] == 0).mean()
  desc = df["accident_count"].describe()

  logger.info("rows:\t%s", len(df))
  logger.info("zero_share:\t%s", zero_share)    
  logger.info(
    "Accident count stats | mean=%.4f | min=%d | max=%d | unique=%d",
    desc["mean"],
    desc["min"],
    desc["max"],
    df["accident_count"].nunique()
    )

  return 


def zero_inflation(df, data):
  # (7) „Zero Inflation“ korrekt machen
  h3_cells = pd.Index(df["h3"].unique(), name="h3")
  time_bins = pd.date_range(df["tbin"].min(), df["tbin"].max(), freq="D", name="tbin")

  grid = (
      pd.MultiIndex.from_product([h3_cells, time_bins])
        .to_frame(index=False)
  )

  data_full = grid.merge(data, on=["h3", "tbin"], how="left")
  data_full["accident_count"] = data_full["accident_count"].fillna(0).astype("int64")

  # Zeitfeatures nachziehen
  data_full["weekday"] = data_full["tbin"].dt.weekday
  data_full["is_weekend"] = (data_full["weekday"] >= 5).astype(int)
  data_full["month"] = data_full["tbin"].dt.month
  data_full["year"] = data_full["tbin"].dt.year

  return data_full 


def time_aware_split():
  # (8) Train/Test Split ohne Leakage (Time split)
  cut = data_full["tbin"].quantile(0.8)                   # oder fix: "2019-01-01" etc.
  train = data_full[data_full["tbin"] <= cut].copy()
  test  = data_full[data_full["tbin"] > cut].copy()

  return 
  # (9) Was kommt als Modell als erstes?
  # Modelle:
  # Poisson / Negative Binomial
  # XGBoost / LightGBM
  # Temporal features + Context features

  # --> LightGBM / XGBoost auf log1p(count) oder direkt count (mit Poisson objective, wenn verfügbar)



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

  # (1) load data + filter 
  cols = ["id",
          "lat_norm", 
          "lon_norm", 
          "datetime", 
          "light conditions", 
          "intersection type", 
          "weather", 
          "collision type"]
  
  f_path = f"/home/robfra/0_Portfolio_Projekte/Road_accidents/data/data_processed/df_character_norm.csv"

  df_pre = pd.read_csv(f_path,
                    low_memory=False)
  df = df_pre[cols].copy()

  # (2) creating time_bin_column
  df_tbin = time_binning(df, freq="D")

  # (3) creating h3_grid
  # if h3_test == True: 
  #   h3_values = session.h3_values
  #   h3_grid_test(h3_values)
  grid_dict = {}
  for h3_value in [7]:  # , 8, 9
    logger.info("Using H3_resolution=%s", h3_value)
    df_h3 = create_h3(df_tbin, h3_value)
    grid_dict[f"RES_{h3_value}"] = df_h3

  for name, df in grid_dict.items():
    # (4) compile accidents per 'h3 x tbin'
    logger.info("Getting accident's incidence (%s)",
                name)
    df_2 = get_accidents_incidence(df)

    # (5) Before_Sparsity_check on dfs
    logger.info("BEFORE INFLATION -- Sparsity check on %s", 
                name)
    sparsity_check(df_2)

    # (6) Zero Inflation
    df_zero = zero_inflation(df, df_2)

    # (7) After_Sparsity_check on dfs
    logger.info("AFTER INFLATION -- Sparsity check on %s", 
                name)
    sparsity_check(df_zero)
    
    # save dfs
    folder = os.getenv("PATH_PROCESSED")
    df_path = f"{folder}/df_h3_{name}.csv"

    df_zero.to_csv(df_path)
    logger.info("df saved (%s)", name)

  return

if __name__ == "__main__":
  h3_transformation()
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