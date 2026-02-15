# imports
from sqlalchemy import text
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import h3

import src.postgre.postgre_helper as post
import src.utils.general_helper as gh
import src.utils.path_helper as ph
from src.core.session import session
from src.core.logger import create_logger
from configuration.H3_risk_maps import config


# mean(count per time)


#------------------
# HELPER FUNCTION
#------------------

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
    df_all.to_csv(df_path)
    logger.info("df_h3_all_res_freq saved to %s", df_path)

    return df_all


def check_h3_df(df):
    # setup logger
    logger = session.logger

    logger.info("Checking H3 DataFrame\n ---- INFO ----\n%s\n ---- DESCRIBE ----\n%s\n", 
                df.info(), 
                df.describe())
    
    logger.info("Further Checks:\n --- VALUE COUNTS ('resolution') ----\n%s\n\nN_UNIQUE ('H3-INDEX') = %s\n", 
                df["resolution"].value_counts(),
                df["h3_index"].nunique())

    return 

# 
def h3_heatmap_plt(df, res_list): #  freq_list):
    # setup logger
    logger = session.logger

    if not isinstance(res_list, list):
        logger.info("Converted 'res_list'(dtype=%s) to list.", type(res_list))
        res_list = [res_list]

    # (1)
    # fig, axes = plt.subplots(
    #     1,
    #     len(res_list),
    #     figsize=(6 * len(res_list), 6),
    #     constrained_layout=True
    #     )

    # if len(res_list) == 1:
    #     axes = [axes]
    
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

            logger.info("Creating heatmap 'road_accidents' (res = %s, freq = %s, vmax = %s)", 
                        res,
                        freq,
                        vmax)

            # (2) Plot 
            gdf = gpd.GeoDataFrame(df_res, 
                                geometry="geometry", 
                                crs="EPSG:4326")

            gdf.plot(
                column=scale,
                cmap="Reds",
                legend=True,
                edgecolor="none",
                ax=ax,
                vmin=0,
                vmax=vmax
                )

            scale_name = "linear" if scale == "n_accidents" else "log"
            ax.set_title(f"H3 Road Accidents Heatmap \n(res={res}, freq={freq}, scale={scale_name})")
            ax.set_axis_off()

            inflate = session.inflate
            plot_name = f"{heatmap_folder}/h3_accident_heatmap_res{res}_{freq}{'_ZeroInfl' if inflate else ''}_{scale_name}.png"
            fig.savefig(
                    plot_name,
                    dpi=300,
                    bbox_inches="tight"
                    )
            plt.close(fig)
            logger.info("Heatmap saved to %s", ph.shorten_path(plot_name))

    # all_heatmap_path = f"{heatmap_folder}/h3_accident_heatmap_res4to9.png"

    # plt.savefig(all_heatmap_path, dpi=300, bbox_inches="tight")
    # logger.info("Heatmap saved to %s", ph.shorten_path(all_heatmap_path))
    
    # if verbose:
    #     plt.show()
    # else:
    #     plt.close(fig)
    


    # return

def h3_heatmap_fol(df, res_list):


    # This example uses heatmaps to visualize the density of volcanoes
    # # which is more in some parts of the world compared to others.

    # from folium import plugins

    # map = folium.Map(location=[15, 30], tiles="Cartodb dark_matter", zoom_start=2)

    # heat_data = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

    # heat_data
    # plugins.HeatMap(heat_data).add_to(map)

    # map
    return 

def retrieve_base_stat(value, conn):
    # load variable from session
    inflate = session.inflate

    # query    
    base_stat = f"""
        SELECT
            COUNT(*) AS rows,
            MIN(n_accidents) AS MIN,
            AVG(n_accidents) AS MEAN,
            VAR_SAMP(n_accidents) AS VAR, 
            STDDEV_SAMP(n_accidents) AS STD, 
            VAR_SAMP(n_accidents) / STDDEV_SAMP(n_accidents) AS RATIO_VAR_MEAN,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY n_accidents) AS Q25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_accidents) AS MEDIAN, 
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY n_accidents) AS Q75,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY n_accidents) AS Q90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY n_accidents) AS Q95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY n_accidents) AS Q99,
            MAX(n_accidents) AS MAX,
            SUM(CASE WHEN n_accidents = 0 THEN 1 ELSE 0 END) * 1.0 AS ZERO_COUNT, 
            SUM(CASE WHEN n_accidents = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS ZERO_SHARE
        FROM accidents.h3_res{value}_month{"_ZeroInf" if inflate else ""};
        """

    result = conn.execute(text(base_stat))
    df = pd.DataFrame(result.fetchall())

    return df

# retrieve basic statistics from SQL tables
def create_base_stat_df(h3_values, freq):
    # setup logger
    logger = session.logger

    # setup DB_engine
    engine = post.get_engine()
    
    h3_dict = {}
    with engine.begin() as conn:
        for val in h3_values:
            df = retrieve_base_stat(val, conn)
            
            df["res"] = val
            df["freq"] = freq
            h3_dict[f"res_{val}"] = df

    h3_stat_merge = pd.concat(h3_dict.values())
    describe_save_h3_df(h3_stat_merge, 
                        f_name="df_h3_base_stats",
                        idx_new=["res", "freq"])

    return
    

def describe_save_h3_df(df, f_name, idx_new=None):
    # setup logger
    logger = session.logger

    # 
    if idx_new is not None:
        df.set_index(idx_new, inplace=True)

    for col in ["mean_count", "var_count", 
                "std_count", "var_mean_ratio", 
                "zero_count", "zero_share"]:
        df[col] = df[col].astype(float).round(3)

    logger.info("h3_stat_merge -- INFO ---\n%s\n", df.info()) 
    logger.info("h3_stat_merge -- OVERVIEW ---\n%s\n", df.T)
    
    folder = os.getenv("PATH_PROCESSED")
    df_path = f"{folder}/{f_name}.csv"
    df.to_csv(df_path)
    logger.info("Saved df '%s' to %s", 
                f_name,
                ph.shorten_path(df_path)) 

    return

def create_h3_heatmap():
    # load env variables
    gh.load_env_vars()

    session.load_config(config)
    log_name = session.log_name # "ETL_CHARACTERISTICS"
    name_logfile = session.log_file # "etl_characteristics"
    
    h3_values = session.h3_values
    res_list = session.resolution
    freq_list = session.freq

    # load logger
    logger = create_logger(name=log_name,
                            file_name=name_logfile)

    session.logger = logger

    # compute base statistics for each H3 resolution and frequency
    create_base_stat_df(h3_values, freq_list)

    # 
    all_h3_df = load_create_all_h3_df(res_list, freq_list)

    # 
    if "plt" in session.plotting:
        plot_h3_heatmap_plt(all_h3_df, res_list)    

    if "fol" in session.plotting:
        plot_h3_heatmap_plt(all_h3_df, res_list) 

    return 

if __name__ == "__main__":
    create_h3_heatmap()


    # res_list = session.resolution
    # h3_path = "/home/robfra/0_Portfolio_Projekte/Road_accidents/data/data_processed/df_h3_all_res_freq.csv"
    # df = pd.read_csv(h3_path)
    # plot_h3_heatmap(df, res_list)   # , freq_list)


"""
📐 Wie groß sind H3-Zellen?

H3 ist global definiert. Die mittlere Zellfläche pro Resolution ist bekannt.

Die ungefähren mittleren Flächen:

Resolution	Ø Fläche (km²)	Ø Kantenlänge (km)
4	~1770 km²	~25 km
5	~252 km²	~9.4 km
6	~36 km²	~3.6 km
7	~5.1 km²	~1.4 km
8	~0.73 km²	~0.53 km
9	~0.10 km²	~0.20 km

Das sind Mittelwerte (wegen icosahedral distortion leicht variierend).
"""

"""
# 1️⃣ Matplotlib + h3 (schnelle EDA)
# 2️⃣ Folium oder Plotly (interaktive Karten)

📦 Module-Übersicht
🔹 h3-py (unbedingt)

Um Polygone zu generieren:
h3.cell_to_boundary(h3_index)

🔹 GeoPandas
Für:
GeoDataFrame
CRS Handling
einfache Plot-Funktion

import geopandas as gpd
from shapely.geometry import Polygon

🔹 Matplotlib
Für:
statische Heatmaps
schnelle Kontrolle
gdf.plot(column="n_accidents", cmap="Reds", legend=True)

🔹 Folium (Leaflet)
Für:
interaktive Karten
Hover-Info
Präsentation
Sehr geeignet für Portfolio.

🔹 Plotly (optional)
Für:
moderne interaktive Darstellung
Integration in Streamlit
"""