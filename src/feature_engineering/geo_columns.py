## geo_columns
# import
import pandas as pd
import numpy as np
from pyproj import Transformer
from collections import defaultdict
import h3
from typing import List

import src.utils.visualisation_helper as viz



def add_h3_col(df_in, 
               h3_value: int | List, 
               lat_col: str, 
               lon_col: str):
    # setup logger
    # logger = session.logger
    # if logger:
    #     logger.info("Start creating h3 geo_grid (H=%s)", h3_value)

    # (1) only valid geo data
    df_in[lat_col] = pd.to_numeric(df_in[lat_col], 
                                   errors="coerce")
    df_in[lon_col] = pd.to_numeric(df_in[lon_col], 
                                   errors="coerce")
    df = df_in[df_in[lat_col].notna() & df_in[lat_col].notna()].copy()

    # (2) 
    if isinstance(h3_value, int):
        h3_value = [h3_value]

    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()

    max_res = max(h3_value)

    base_cells = [
        h3.latlng_to_cell(la, lo, max_res)
        for la, lo in zip(lat, lon)
    ]

    df[f"h3_res{max_res}"] = base_cells

    for res in h3_value:
        if res != max_res:
            df[f"h3_res{res}"] = [
                        h3.cell_to_parent(c, res)
                        for c in base_cells
                        ]

    return df

# def check_geo_valid(df, data_viz=True):
#     # setup logger
#     # logger = session.logger

#     # (2C) Feature Engineering 'geo'
#     geo_valid = df["latitude"].notna() & df["longitude"].notna()
#     # logger.info("ratio 'geo_valid' (unnormalized)\t", geo_valid.mean().round(4))

#     geo_valid_new = None

#     if ("lat_norm" in df.columns) and ("lon_norm" in df.columns):
#         geo_valid_new = df["lat_norm"].notna() & df["lon_norm"].notna()

#         logger.info("\nratio 'geo_valid' (normalized)\t", geo_valid_new.mean().round(4))

#     if data_viz:
#         geo_data = geo_valid_new if geo_valid_new is not None else geo_valid

#         sample = df.loc[geo_data].sample(20_000, random_state=42)

#         viz.create_geo_scatterplot(sample)

#     return


# def lat_long_normalisation(df_in):
#     # setup logger
#     # logger = session.logger

#     # logger.info("Start normalizing 'latitude' and 'longitude'")
#     df = df_in.copy()

#     def normalize_lat_lon(lat, lon):
#         if pd.isna(lat) or pd.isna(lon):
#             return pd.NA, pd.NA

#         # already plausible
#         if 40 <= lat <= 52 and -6 <= lon <= 11:
#             return lat, lon

#         # try scaled versions
#         for scale in (1e5, 1e6):
#             lat_s = lat / scale
#             lon_s = lon / scale
#             if 40 <= lat_s <= 52 and -6 <= lon_s <= 11:
#                 return lat_s, lon_s

#         return pd.NA, pd.NA

#     norm = df.apply(
#         lambda r: normalize_lat_lon(r["latitude"], r["longitude"]),
#         axis=1,
#         result_type="expand",
#     )

#     df[["lat_norm", "lon_norm"]] = norm
#     check_geo_valid(df)

#     return df


def generate_profile_from_df(df, checks: dict):

    observation_dict = {}
    for name, fun in checks.items():
        observation_dict[name] = fun(df)        # !!!! Observation object

    return observation_dict


def classify_coord_row( 
                    lat_in: pd.Series, 
                    lon_in: pd.Series, 
                    config: dict
                    ) -> str:
    
    lat = pd.to_numeric(lat_in, errors="coerce")
    lon = pd.to_numeric(lon_in, errors="coerce")
    
    rules = config.get("coordinate_detection", {}).get("rules", {})
    wgs_rules = rules.get("wgs84", {})
    lambert_rules = rules.get("lambert93", {})

    lon_min = wgs_rules.get("lon_min")
    lon_max = wgs_rules.get("lon_max")
    lat_min = wgs_rules.get("lat_min")
    lat_max = wgs_rules.get("lat_max")

    x_min = lambert_rules.get("x_min")
    x_max = lambert_rules.get("x_max")
    y_min = lambert_rules.get("y_min") 
    y_max = lambert_rules.get("y_max")

    # for col in df.columns: 
    if pd.isna(lat) or pd.isna(lon):
        return "missing"

    # if lat == 0:
    #     return "invalid"

    if (lat_min is None) or (lat_max is None) or (lon_min is None) or (lon_max is None):
        print("Check min and max values for 'latitude' + 'longitude' from config for None")
        print("WGS_RULES")
        print("lon_min:", lon_min, type(lon_min))
        print("lon_max:", lon_max, type(lon_max))
        print("lat_min:", lat_min, type(lat_min))
        print("lat_max:", lat_max, type(lat_max))
    
    # plausible WGS84 for metropolitan France + Corsica
    if (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max):
        return "wgs84"

    if (y_min is None) or (y_max is None) or (x_min is None) or (x_max is None):
        print("Check min and max values for 'x' + 'y' from config for None")
        print("LAMBERTRULES")
        print("x_min:", x_min, type(x_min))
        print("x_max:", x_max, type(x_max))
        print("y_min:", y_min, type(y_min))
        print("y_max:", y_max, type(y_max))
        
    # plausible Lambert-like projected coords in your data
    if (y_min <= lat <= y_max) and (x_min <= lon <= x_max):
        return "lambert93_like"

    # possible swapped WGS84
    if (lat_min <= lon <= lat_max) and (lon_min <= lat <= lon_max):
        return "swapped_wgs84"

    # possible swapped lambert93
    if (y_min <= lon <= y_max) and (x_min <= lat <= x_max):
        return "swapped_lambert93"
    
    return "invalid" 


def classify_coordinates(df, config, lat_col="lat", lon_col="long"):
    out = df.copy()
    out["coord_class"] = [
        classify_coord_row(lat, lon, config)
        for lat, lon in zip(out[lat_col], out[lon_col])
    ]
    return out


def generate_geo_quality_report(df,
                                json_report=False, 
                                sampling=False,
                                lat_col=None, 
                                lon_col=None):

    counts = df["coord_class"].value_counts(dropna=False)
    total = len(df)

    report = pd.DataFrame({
        "total": total, 
        "count": counts,
        "share": counts / total
    })

    report["share_percent"] = (report["share"] * 100).round(2)
    report_sort = report.sort_values("count", ascending=False)

    summary = {}
    if json_report:
        for cls, n in counts.items():
            summary[cls] = {
                "count": int(n),
                "share": round(n / total, 4)
            }

        summary["total_rows"] = total

    if sampling:
        if not lat_col or not lon_col:
            raise ValueError("""
                    Check if 'lat_col' or 'lon_col' is None.
                    {lat_col}\t{lon_col}
                             """)
           
        samples = sample_by_class(df, lat_col, lon_col)

    else:
        samples = None

    return report_sort, summary, samples


def sample_by_class(df, lat_col, lon_col, n=5):

    samples = {}

    for cls in df["coord_class"].unique():

        samples[cls] = df.loc[
            df["coord_class"] == cls,
            [lat_col, lon_col]
        ].head(n)

    return samples


def convert_lambert_to_wgs84(df, 
                             lat_col, 
                             lon_col, 
                             new_col_suffix, 
                             handle_invalid: str | None=None):
    
    lat_convert = f"{lat_col}_{new_col_suffix}" if new_col_suffix else lat_col
    lon_convert = f"{lon_col}_{new_col_suffix}" if new_col_suffix else lon_col

    if df["coord_class"].nunique() == 1 and df["coord_class"].iloc[0] == "missing":
        print("Dataset contains no usable coordinates")

    out = df.copy()

    print("[BEFORE]", out.shape)

    mask = out["coord_class"] == "lambert93_like"
    if mask.any():
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        lon_lat = out.loc[mask, [lon_col, lat_col]].apply(
            lambda r: transformer.transform(r[lon_col], r[lat_col]),
            axis=1
        )

        out.loc[mask, lon_convert] = [x[0] for x in lon_lat]
        out.loc[mask, lat_convert] = [x[1] for x in lon_lat]

    mask_swap_lamb = out["coord_class"] == "swapped_lambert93"
    if mask_swap_lamb.any():
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        lon_lat = out.loc[mask, [lon_col, lat_col]].apply(
            lambda r: transformer.transform(r[lon_col], r[lat_col]),
            axis=1
        )

        out.loc[mask, lon_convert] = [x[1] for x in lon_lat]
        out.loc[mask, lat_convert] = [x[0] for x in lon_lat]

    mask_wgs = out["coord_class"] == "wgs84"
    if mask_wgs.any():
        out.loc[mask_wgs, lon_convert] = out.loc[mask_wgs, lon_col]
        out.loc[mask_wgs, lat_convert] = out.loc[mask_wgs, lat_col]

    mask_swap = out["coord_class"] == "swapped_wgs84"
    if mask_swap.any():
        out.loc[mask_swap, lon_convert] = out.loc[mask_swap, lat_col]
        out.loc[mask_swap, lat_convert] = out.loc[mask_swap, lon_col]

    mask_missing = out["coord_class"] == "missing"
    if mask_missing.any():
        out.loc[mask_missing, [lon_convert, lat_convert]] = np.nan
    
    if handle_invalid == "drop": 
        out_drop = out[out["coord_class"] != "invalid"].copy()
        print("[AFTER -- DROPPED] ", out_drop.shape)

        return out_drop

    if handle_invalid == "nan":
        mask_inv = out["coord_class"] == "invalid"
        if mask_inv.any():
            out.loc[mask_inv, [lon_convert, lat_convert]] = np.nan

    print("[AFTER -- UNDROPPED]", out.shape)
        
    return out


def h3_column(res):

    return 