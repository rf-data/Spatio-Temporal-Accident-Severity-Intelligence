# imports
from sqlalchemy import text, inspect
import numpy as np
import pandas as pd
import os
import geopandas as gpd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping
import h3
import folium
import json
import branca.colormap as cm

import src.utils.postgre_helper as post
import src.utils.general_helper as gh
import src.utils.path_helper as ph
from src.core.session import session
from src.core.logger import create_logger
from configuration.H3_risk_maps_v2 import config

# mean(count per time)


# ------------------
# HELPER FUNCTION
# ------------------


def check_table(engine, schema, table_name):
    # setup logger
    logger = session.logger

    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)

    if table_name not in tables:
        logger.error(f"❌ Table {schema}.{table_name} does NOT exist.")
        return

    df = pd.read_sql(f"SELECT COUNT(*) as n FROM {schema}.{table_name}", engine)
    logger.info(f"✅ Table {schema}.{table_name} exists.")
    logger.info(f"Rows: {df.loc[0, 'n']}")

    return


def general_check():
    # setup DB_engine
    engine = post.get_engine()

    h3_values = session.h3_values

    for res in h3_values:
        check_table(engine, "accidents", f"h3_res{res}_month")

    return


def h3_to_polygon(h3_index):
    if pd.isna(h3_index):
        return None
    coords = h3.cell_to_boundary(str(h3_index))
    coords_lonlat = [(lon, lat) for lat, lon in coords]
    return Polygon(coords_lonlat)


def h3_to_geojson_feature(h3_index, value, resolution):
    polygon = h3_to_polygon(h3_index)

    return {
        "type": "Feature",
        "geometry": mapping(polygon),
        "properties": {"h3_index": h3_index, "value": value, "resolution": resolution},
    }


def load_single_h3_data(res, freq):
    # setup logger
    logger = session.logger

    # setup DB_engine
    engine = post.get_engine()

    #
    table = f"h3_res{res}_{freq}"

    query = f"""
        SELECT 
            h3_index, 
            {freq}_start, 
            n_accidents,
            {res} AS resolution
        FROM accidents.{table};
    """
    logger.info("Loading H3 data for resolution (%s) and frequency (%s)", res, freq)
    df = pd.read_sql(text(query), engine)

    return df


def load_create_all_h3_df(res_list, freq_list):
    # setup logger
    logger = session.logger

    if not isinstance(res_list, list):
        logger.info("Converted 'res_list'(dtype=%s) to list.", type(res_list))
        res_list = [res_list]

    if not isinstance(freq_list, list):
        logger.info("Converted 'freq_list'(dtype=%s) to list.", type(freq_list))
        freq_list = [freq_list]

    dfs = []
    for res in res_list:
        for freq in freq_list:
            df = load_single_h3_data(res, freq)
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    check_h3_df(df_all)

    # save df
    folder = os.getenv("PATH_PROCESSED")
    df_path = f"{folder}/df_h3_all_res_freq.csv"
    # df_all.to_csv(df_path)
    logger.info("df_h3_all_res_freq saved to %s", ph.shorten_path(df_path))

    return df_all


def check_h3_df(df):
    # setup logger
    logger = session.logger

    logger.info(
        "Checking H3 DataFrame\n ---- INFO ----\n%s\n ---- DESCRIBE ----\n%s\n",
        df.info(),
        df.describe(),
    )

    logger.info(
        "Further Checks:\n --- VALUE COUNTS ('resolution') ----\n%s\n\nN_UNIQUE ('H3-INDEX') = %s\n",
        df["resolution"].value_counts(),
        df["h3_index"].nunique(),
    )

    return


#
def h3_heatmap_plt(df, res_list):  #  freq_list):
    # setup logger
    logger = session.logger

    if not isinstance(res_list, list):
        logger.info("Converted 'res_list'(dtype=%s) to list.", type(res_list))
        res_list = [res_list]

    plot_folder = os.getenv("PATH_PLOT")
    heatmap_folder = f"{plot_folder}/risk_heatmaps"
    ph.ensure_dir(heatmap_folder)

    freq = session.freq

    # df_res_dict = {}
    for res in res_list:

        fig, ax = plt.subplots(figsize=(6, 6))

        df_res = (
            df[df["resolution"] == res]
            .groupby("h3_index")["n_accidents"]
            .sum()
            .reset_index()
            .copy()
        )

        # adding cols 'geometry' and 'log_count'
        unique_cells = df_res["h3_index"].unique()

        poly_dict = {
            h: Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)])
            for h in unique_cells
        }
        df_res["geometry"] = df_res["h3_index"].map(poly_dict)
        df_res["log_count"] = np.log1p(df_res["n_accidents"])

        for scale in ["n_accidents", "log_count"]:
            vmax = df_res[scale].max().round(3)

            logger.info(
                "Creating heatmap 'road_accidents' (res = %s, freq = %s, vmax = %s)",
                res,
                freq,
                vmax,
            )

            # (2) Plot
            gdf = gpd.GeoDataFrame(df_res, geometry="geometry", crs="EPSG:4326")

            gdf.plot(
                column=scale,
                cmap="Reds",
                legend=True,
                edgecolor="none",
                ax=ax,
                vmin=0,
                vmax=vmax,
            )

            scale_name = "linear" if scale == "n_accidents" else "log"
            ax.set_title(
                f"H3 Road Accidents Heatmap \n(res={res}, freq={freq}, scale={scale_name})"
            )
            ax.set_axis_off()

            inflate = session.inflate
            plot_name = f"{heatmap_folder}/h3_accident_heatmap_res{res}_{freq}{'_ZeroInfl' if inflate else ''}_{scale_name}.png"
            fig.savefig(plot_name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Heatmap saved to %s", ph.shorten_path(plot_name))

    return


def h3_heatmap_fol(df_in, res_list, center=[48.5, 2.2], zoom_start=4):
    # setup logger
    logger = session.logger
    freq = session.freq

    if not isinstance(res_list, list):
        logger.info("Converted 'res_list'(dtype=%s) to list.", type(res_list))
        res_list = [res_list]

    # --- Global log scale ---
    df_all = df_in.copy()
    df_all["log_count"] = np.log1p(df_all["n_accidents"])

    global_max = df_all["log_count"].max()

    # --- Base map ---
    map = folium.Map(
        location=center,
        tiles="Cartodbpositron",  # "Cartodb dark_matter"
        zoom_start=zoom_start,
    )

    #
    for res in res_list:
        feats = create_feats_from_df(df_all, res)
        geojson = {"type": "FeatureCollection", "features": feats}

        layer = folium.FeatureGroup(name=f"Resolution {res}")

        geo_json, color_map = customise_geojson(geojson, global_max)
        geo_json.add_to(layer)
        color_map.add_to(map)

        layer.add_to(map)

    folium.LayerControl(collapsed=False).add_to(map)

    plot_folder = os.getenv("PATH_PLOT")
    heatmaps = f"{plot_folder}/risk_heatmaps"
    ph.ensure_dir(heatmaps)

    resolutions = "_".join(str(r) for r in res_list)
    plot_name = f"{heatmaps}/h3_acci_heatmap_fol_res{resolutions}_{freq}.html"

    map.save(plot_name)
    logger.info("Folium map saved to %s", ph.shorten_path(plot_name))

    return map


def customise_geojson(geojson, global_max):
    # setup logger
    logger = session.logger

    #
    colormap = cm.linear.YlOrRd_09.scale(0, global_max)
    colormap.caption = "Log(Accidents + 1)"

    geo = folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"]["value"]),
            "color": None,
            "weight": 0,
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["h3_index", "value"], aliases=["H3 Index:", "Log Accidents:"]
        ),
    )

    return geo, colormap


def create_feats_from_df(df, res):
    # setup logger
    logger = session.logger

    #
    df_res = (
        df[df["resolution"] == res]
        .groupby("h3_index")["n_accidents"]
        .sum()
        .reset_index()
        .copy()
    )

    df_res["log_count"] = np.log1p(df_res["n_accidents"])

    features = []

    for _, row in df_res.iterrows():
        feature = h3_to_geojson_feature(row["h3_index"], row["log_count"], res)
        features.append(feature)

    return features


def create_h3_heatmap():
    # load env variables
    gh.load_env_vars()

    session.load_config(config)
    log_name = session.log_name
    name_logfile = session.log_file

    # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    #
    res_list = session.h3_values
    freq_list = session.freq

    all_h3_df = load_create_all_h3_df(res_list, freq_list)

    #
    # if "plt" in session.plotting:
    #     h3_heatmap_plt(all_h3_df, res_list)

    if "folium" in session.plotting:
        h3_heatmap_fol(all_h3_df, res_list)

    return


if __name__ == "__main__":
    create_h3_heatmap()
