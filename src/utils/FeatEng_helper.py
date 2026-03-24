# import
import numpy as np
import pandas as pd

from src.core.session import session
import src.utils.cleaning_helper as ch

"""
    if encode_col == "month":
        total_pa = 12
    elif encode_col == "week":
        total_pa = 52
    else:
        logger.error("Enter invalid value for 'encode_time': %s",
                     encode_col)
        raise ValueError("Enter invalid value for 'encode_time'")
"""


def remove_domtom(df, dep_col):

    print("[BEFORE]", df.shape)

    DOM_DEPARTMENTS = ["971", "972", "973", 
                       "974", "975", "976",
                       "977", "978", "986",
                       "987", "988"]

    # df["region_type"] = np.where(
    #     df["department"].astype(str).isin(DOM_DEPARTMENTS), "dom", "metropole"
    # )

    df_red = df[~df[dep_col].astype(str).isin(DOM_DEPARTMENTS)].copy()
    print("[AFTER]", df_red.shape)

    return df_red