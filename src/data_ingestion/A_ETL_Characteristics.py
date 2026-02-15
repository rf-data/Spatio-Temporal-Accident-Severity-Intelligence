## 
# imports
# import numpy as np
import pandas as pd
import os
from pathlib import Path

import src.utils.file_helper as fh 
import src.utils.cleaning_helper as ch
import src.utils.FeatEng_helper as feat
import src.utils.general_helper as gh
from src.core.session import session
from src.core.logger import create_logger
from configuration.ETL_characteristics import config

#------------------
# HELPER FUNCTION
#------------------

#------------------
# MAIN FUNCTION
#------------------

def etl_characteristics():
    # load env variables
    gh.load_env_vars()

    session.load_config(config)
    log_name = session.log_name # "ETL_CHARACTERISTICS"
    name_logfile = session.log_file # "etl_characteristics"

    # load logger
    logger = create_logger(name=log_name,
                           file_name=name_logfile)

    session.logger = logger

    # (1) load dataset from csv files 
    folder = os.getenv("FOLDER_CHARACT")
    fold_charac = Path(folder)
    dfs_charac = fh.load_files_from_folder(fold_charac)

    dfs_new = []
    for i, df_in in enumerate(dfs_charac):
        logger.info("Processing df_%s of %s", i+1, len(dfs_charac)) 

        df_1= ch.rename_cols(df_in)
        df_2 = ch.correct_charact_cols(df_1)
        df_3 = feat.add_time_cols(df_2)

        df_4 = feat.remove_domtom(df_3)

        dfs_new.append(df_4)

    logger.info("Start merging dfs")
    charac_merge = pd.concat(dfs_new)

    charac_norm = feat.lat_long_normalisation(charac_merge)

    # save dfs
    fold_proc = os.getenv("PATH_PROCESSED")
    merge_path = f"{fold_proc}/df_character_merged.csv"
    norm_path = f"{fold_proc}/df_character_norm.csv"

    try:
        charac_norm.to_csv(norm_path)
    except Exception as e:
        logger.error("Error while saving 'charac_norm':\n%s", e)
    
    try:
        charac_merge.to_csv(merge_path)
    except Exception as e:
        logger.error("Error while saving 'charac_merge':\n%s", e)

    return 


if __name__ == "__main__":
    etl_characteristics()
