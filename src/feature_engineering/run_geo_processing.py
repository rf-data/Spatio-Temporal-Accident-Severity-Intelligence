## run_geo_processing.py
# import
import click
import json
import os
from pathlib import Path

from src.utils.file_helper import (load_files_from_folder, 
                                   save_df_to_parquet, 
                                   save_dict, 
                                   get_yaml_config)
import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.FeatEng_helper as feat

from src.agentic_AI.feature_engineering.geo_columns import (classify_coordinates,
                                                            generate_geo_quality_report,
                                                            add_h3_col, 
                                                            convert_lambert_to_wgs84)

# from src.agentic_AI.tools.tools_tab_EDA import missing_analysis, numeric_summary
from src.agentic_AI.report.raw_data_report import load_eda_summary


def save_geo_output(df, profile, f_name, folder):
    # save df_converted
    # df_path = f"{folder}/{f_name}_converted"

    save_df_to_parquet(df, f_name, folder, chunked=True)
    # print("Saved 'df':", f_name)

    # save df profile
    if profile:
        
        prof_path = Path(f"{folder}/{f_name}_profile.json")
        save_dict(profile, prof_path)
        print("Saved 'profile':", ph.shorten_path(prof_path))
    
    print()
    return 


@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def geo_preprocessing(name):

    run_geo_processing(name)


def run_geo_processing(name):    
    # (1) load config + parse arguments
    gh.load_env_vars()

    # data_raw = os.getenv("PATH_RAW")
    data_processed = os.getenv("PATH_PROCESSED")

    config = get_yaml_config(name)
    arg_dict = config.get("general_args", {})
    selected_years = arg_dict.get("years", None)
    data_folder = Path(arg_dict.get("data_folder", {}))
    folder = Path(f"{data_processed}/{data_folder}")

    # df = load_raw_data(config)
    report = load_eda_summary(config)
    df_names = report.get("files", [])

    df_dict = load_files_from_folder(
                                folder,
                                df_names,
                                "harmonized", 
                                f_type="parquet"
                                    )
    
    process_dict = config.get("geo_processing", {})
    
    domtom_filter = process_dict.get("domtom_filter", False)
    prim_key = process_dict.get("prim_key", False)
    lat_col = process_dict.get("lat_col", None)
    lon_col = process_dict.get("lon_col", None)
    dept_col = process_dict.get("dept_col", "")
    handle_invalid = process_dict.get("handle_invalid_coords", False)
    h3_values = process_dict.get("h3_values", [])
    lat_lon_suffix = "convert"

    # checks = {
    #     "missing": missing_analysis, 
    #     "num_sum": numeric_summary
    #     }

    conv_dict = {}
    for df_name, df in df_dict.items():

        file = df_name.split(".")[0]
        file_parts = file.split("_")

        if selected_years:
            if not any(year in file_parts for year in selected_years):
                print(f"Skip file '{file}'")
                continue

        print("Start processing:", file)
        print("df_head:\n", df.head(5))

        if domtom_filter:
            print(f"[{file}] Removing accidents from DOMTOM")
            df_clean = feat.remove_domtom(df, dept_col)

        else:
            df_clean = df.copy()

        if (lat_col is None) or (lon_col is None): 
            print("Check columns 'latitude' and 'longitude' if 'None'\n", 
                  lat_col, 
                  lon_col)

        # print("Generating profile from df:", df_name)
        profile = None
        # geo.generate_profile_from_df(
        #                                     df_clean,
        #                                     checks
        #                                 )
            
        print(f"[{file}] Classifing coordinates")
        df_class = classify_coordinates(
            df_clean,
            process_dict,
            lat_col,
            lon_col
        )

        geo_report, _, samples = generate_geo_quality_report(
                                                        df_class, 
                                                        sampling=True,
                                                         lat_col=lat_col,
                                                         lon_col=lon_col
                                                         )
        print(geo_report)

        if samples:
            for k, v in samples.items():
                print("\nCLASS:", k)
                print(v)

        print(f"[{file}] Converting coordinates to 'wgs84' system")
        df_conv = convert_lambert_to_wgs84(
            df_class,
            lat_col,
            lon_col,
            new_col_suffix=lat_lon_suffix,
            handle_invalid=handle_invalid
        )

        conv_dict[df_name] = df_conv
        # "df": df,
        #                 "profile": profile,
        #                 "df_conv":
        # df = filter_invalid_coordinates(df, config)

        file_new = f"{file}_converted" 
        save_geo_output(df_conv, profile, file_new, folder)    #, config)

    # df_dict = load_files_from_folder(
    #                                 folder,
    #                                 df_names,
    #                                 "converted", 
    #                                 f_type="parquet"
    #                                 )

    lat_col_new = f"{lat_col}_{lat_lon_suffix}"
    lon_col_new = f"{lon_col}_{lat_lon_suffix}"
    
    for df_name, df in conv_dict.items():

        print("Adding h3_columns to:\t", df_name)

        df_red = df[[prim_key, lat_col_new, lon_col_new]]
        df_h3 = add_h3_col(df_red, 
                           h3_values, 
                           lat_col_new, 
                           lon_col_new)
        
        print("[DEBUG] df head:", df_h3.head(3))
        file_h3 = f"{df_name.split(".")[0]}_h3"

        save_geo_output(df=df_h3, 
                        profile=None,
                        f_name=file_h3,
                        folder=folder)

if __name__ == "__main__":
    geo_preprocessing()