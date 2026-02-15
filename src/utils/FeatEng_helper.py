# import 
import numpy as np
import pandas as pd

from src.core.session import session
import src.utils.cleaning_helper as ch
import src.utils.visualisation_helper as viz


def check_geo_valid(df, viz=True):
    # setup logger
    logger = session.logger

    # (2C) Feature Engineering 'geo' 
    geo_valid = (df["latitude"].notna() & 
                 df["longitude"].notna())
    logger.info("ratio 'geo_valid' (unnormalized)\t", 
          geo_valid.mean().round(4))

    geo_valid_new = None

    if ("lat_norm" in df.columns) and ("lon_norm" in df.columns):
        geo_valid_new = (df["lat_norm"].notna() & 
                         df["lon_norm"].notna())
        
        logger.info("\nratio 'geo_valid' (normalized)\t", 
            geo_valid_new.mean().round(4))

    if viz:
        geo_data = (geo_valid_new 
                    if geo_valid_new is not None 
                    else geo_valid)
        
        sample = df.loc[geo_data].sample(20_000,
                                        random_state=42)
        
        viz.create_geo_scatterplot(sample)

    return 


def lat_long_normalisation(df_in):
    # setup logger
    logger = session.logger

    logger.info("Start normalizing 'latitude' and 'longitude'")
    df = df_in.copy()

    def normalize_lat_lon(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return pd.NA, pd.NA

        # already plausible
        if 40 <= lat <= 52 and -6 <= lon <= 11:
            return lat, lon

        # try scaled versions
        for scale in (1e5, 1e6):
            lat_s = lat / scale
            lon_s = lon / scale
            if 40 <= lat_s <= 52 and -6 <= lon_s <= 11:
                return lat_s, lon_s

        return pd.NA, pd.NA
    
    norm = df.apply(
            lambda r: normalize_lat_lon(r["latitude"], 
                                        r["longitude"]),
            axis=1,
            result_type="expand"
            )

    df[["lat_norm", "lon_norm"]] = norm
    check_geo_valid(df)

    return df


def remove_domtom(df):
     
    DOM_DEPARTMENTS = {"971", "972", "973", "974", "976"}
        
    df["region_type"] = np.where(
                        df["department"].astype(str)\
                            .isin(DOM_DEPARTMENTS),
                        "dom",
                        "metropole"
                    )

    df_red =  df[df["region_type"]=="metropole"].copy()
    
    return df_red


def add_time_cols(df_in):
    df = df_in.copy()

    # (2B) Feature Engineering 'time'     
    df["time_clean"] = df["time (hr:mn)"].apply(
                                            ch.parse_time_hhmm
                                            )

    df["datetime"] = pd.to_datetime(
                            df[["year", "month", "day"]]\
                                .astype(str)\
                                .agg("-".join, axis=1)
                            + " "
                            + df["time_clean"],
                            errors="coerce"
                        )
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5

    return df

