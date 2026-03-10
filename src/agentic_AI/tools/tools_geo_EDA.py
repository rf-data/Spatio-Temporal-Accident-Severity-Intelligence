## tools.py
# imports
from sqlalchemy import text
from sklearn.cluster import DBSCAN
import numpy as np


# from src.audit.tools.tools_general import FORBIDDEN_KEYWORDS, ALLOWED_TABLES, ToolResult
from src.core.tools_classes import (tools_registry, 
                                    add_tool, 
                                    Observation)

# import src.utils.postgre_helper as post
# from src.core.session import session


# -----------------------
# GEO TOOL FUNCTIONS
# -----------------------
@add_tool(tools_registry, 
        description="""
        Evaluates if df has columns named 'gps', 'longitude' and/or 'latitude' 
        (incl. abbreviations).
        """,
        category="geo",
        eda=True,
        default=True,
        cross_file=False)
def detect_geo_columns(df, config=None):

    lat_candidates = []
    lon_candidates = []
    gps_candidates = []
            

    for col in df.columns:

        name = col.lower()

        if "lat" in name:
            lat_candidates.append(col)

        if "lon" in name or "long" in name:
            lon_candidates.append(col)

        if "gps" in name:
            gps_candidates.append(col)

    hint = {"geo_col_candidates": lat_candidates +
                                   lon_candidates +
                                   gps_candidates
                                   }

    return Observation(
                tool_name="detect_geo_columns",
                category="geo",
                column="geo",
                description="""
                Evaluates if df has columns named 'gps', 'longitude' and/or 'latitude' 
                (incl. abbreviations)
                """,
                metrics={
                    "lat_candidates": lat_candidates,
                    "lon_candidates": lon_candidates,
                    "gps_candidates": gps_candidates
                    },
                recommendation_hint=hint
                )
    

@add_tool(tools_registry, 
        description="Evaluates if potential geo_col candidates have dupliacte rows.",
        category="geo",
        eda=True,
        default=True,
        cross_file=False)
def find_geo_dups(df, geo_candidates, config=None):

    # geo_candidates = detect_geo_columns(df)
    lat = geo_candidates.metrics.get("lat_candidates", [])
    lon = geo_candidates.metrics.get("lon_candidates", [])
    gps = geo_candidates.metrics.get("gps_candidates", [])
    hint = geo_candidates.recommendation_hint or {}

    duplicates = {}
    if lat and lon:
        for lon_col in lon:
            for lat_col in lat:

                dup = df.duplicated(subset=[lat_col, lon_col]).sum()

                duplicates["lat_lon"] = {
                                    "duplicates": int(dup),
                                    "ratio": float(dup / len(df))
                                        }

    if gps:
        dup_gps = df.duplicated(subset=gps).sum()
        duplicates["gps"] = {
                    "duplicates": int(dup_gps),
                    "ratio": float(dup_gps / len(df))
                    }
    if duplicates:
        hint["geo_duplicates"] = duplicates

    # col_dict["geo_duplicates"] = geo_dups
    
    geo_col_candidates = lat + lon + gps

    return Observation(
                tool_name="geo_duplicate_analysis",
                category="geo",
                column="geo",
                description="""
                Evaluates if df has columns named 'gps', 'longitude' and/or 'latitude' 
                (incl. abbreviations)
                """,
                metrics={
                    "duplicates": duplicates
                    },
                recommendation_hint=hint
                )


# geo_dtype
# geo_range
# spatial_duplicates

# --> 
#(A) Typnormalisierung
# Viele Geo-Datensätze haben:

# lat  -> string
# long -> string

# oder

# lat  -> comma decimal

# Minimalregel:
# df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
# df["long"] = pd.to_numeric(df["long"], errors="coerce")
# (B) Range-Check (sehr wichtig)

# Viele Datensätze enthalten Fehler wie:

# lat = 4800000
# lon = 2000000

# Check:
# lat_valid = df["lat"].between(-90, 90)
# lon_valid = df["long"].between(-180, 180)

# Report:
# invalid_geo = (~lat_valid | ~lon_valid).sum()

# Das gehört definitiv in deine file-EDA checks.

# (C) Precision normalisieren (optional)
# Bei Merge oder Clustering können Float-Precision-Probleme auftreten.

# Dann:
# df["lat"] = df["lat"].round(6)
# df["long"] = df["long"].round(6)
    

# def detect_spatial_clusters(df, lat_col="lat", lon_col="long", eps=0.0005):
#     coords = df[[lat_col, lon_col]].dropna().values

#     clustering = DBSCAN(eps=eps, min_samples=2).fit(coords)

#     labels = clustering.labels_
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#     return {
#         "clusters_detected": int(n_clusters),
#         "cluster_points": int((labels != -1).sum())
#     }

# @add_tool(
#     tools_registry,
#     description="Compares metrics for two given H3_resolutions.",
#     category="geo_data",
# )
# def cross_resolution_check(engine, table_low, table_high, schema):
#     """SKIZZE !!!!
#     Idee:
#     Gruppiere res_low
#     Aggregiere
#     Vergleiche mit res_high

#     Hier musst du definieren:
#     wie parent_index berechnet wird
#     welche H3-Level verglichen werden

#     Das wird dein Signature-Tool.
#     """
#     query = text(f"""
#         SELECT 
#             l.parent_index,
#             SUM(l.n_accidents) AS sum_children,
#             h.n_accidents AS parent_value,
#             SUM(l.n_accidents) - h.n_accidents AS difference
#         FROM {schema}.{table_low} l
#         JOIN {schema}.{table_high} h
#         ON l.parent_index = h.h3_index
#         GROUP BY l.parent_index, h.n_accidents
#     """)
