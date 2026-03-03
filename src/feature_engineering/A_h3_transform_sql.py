# H3 Index pro Unfall erzeugen
import pandas as pd
import os

# from pathlib import Path
# from sqlalchemy import text

from src.core.session import session
from src.core.logger import create_logger
import src.utils.general_helper as gh
import src.utils.geo_helper as geo

# import src.postgre.postgre_helper as post
# import src.utils.path_helper as ph
from configuration.H3_ZeroInflate_4to7_weekly import config

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

# ------------------
# HELPER FUNCTION
# ------------------


# (9) Was kommt als erstes Modell?
# Modelle:
# LogReg
# Poisson / Negative Binomial
# XGBoost / LightGBM
# Temporal features + Context features

# --> LightGBM / XGBoost auf log1p(count) oder direkt count (mit Poisson objective, wenn verfügbar)

# ------------------
# MAIN FUNCTION
# ------------------


def h3_transformation():
    # load env variables
    gh.load_env_vars()

    # load configurations
    session.load_config(config)
    log_name = session.log_name  # "ETL_CHARACTERISTICS"
    name_logfile = session.log_file  # "etl_characteristics"

    # extract configurations from session
    cols = session.cols_needed
    src_tbl = session.src_table
    inflate = session.inflate
    freqence = session.frequence
    h3_values = session.h3_values

    # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    # (0) load data + reduced to necessary columns
    folder = os.getenv("PATH_PROCESSED")
    f_path = f"{folder}/df_character_norm.csv"

    df_pre = pd.read_csv(f_path, low_memory=False)
    df = df_pre[cols].copy()

    # (1) apply geo_grid (h3)
    if h3_values is None:
        logger.error("'H3_values' is None.")
        raise ValueError

    df_h3 = geo.create_h3_grid(df, h3_values)

    # save dfs
    folder = os.getenv("PATH_PROCESSED")
    df_path = f"{folder}/df_h3_pre-weekly.csv"

    df_h3.to_csv(df_path)
    logger.info("df_h3 saved")

    # (2) add cols in SQL_DB
    geo.add_h3_cols(h3_values)

    logger.info("Shape of df_h3: %s\n", df_h3.shape)
    geo.fill_h3_columns(df_h3)

    if not isinstance(freqence, list):
        freqence = [freqence]
        logger.info(
            "Converted freqence to list:\t%s (dtype = %s)", freqence, type(freqence)
        )

    for value in h3_values:
        for freq in freqence:
            geo.create_crosstable(value, freq, replace=False, also_parquet=True)

            if inflate:
                geo.zero_inflate_data(
                    value, freq, src_tbl, as_table=True, replace=False
                )


if __name__ == "__main__":
    h3_transformation()
