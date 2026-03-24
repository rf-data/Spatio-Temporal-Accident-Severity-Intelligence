# imports
import click
import os
from pathlib import Path
from typing import List
# from datetime import datetime
# from sqlalchemy import text
from collections import defaultdict
import numpy as np
import pandas as pd

from src.utils.file_helper import get_yaml_config
from src.agentic_AI.report.raw_data_report import load_eda_summary

import src.utils.general_helper as gh
import src.feature_engineering.time_columns as time
import src.utils.file_helper as fh
import src.utils.df_helper as dfh
import src.utils.path_helper as ph
import src.utils.eda_helper as eda
import src.utils.evaluation_helper as eval

# from src.core.session import session
# from src.core.logger import create_logger
# import src.utils.postgre_helper as post
# import src.utils.path_helper as ph
# from configuration.H3_evaluate_res import config




def h3_tbin_evaluation(df, config):

    res_col = config.get("res", "h3_index")
    
    # files = Path(folder).glob("h3_res*_*.parquet")

    # res_col = f"h3res_{res}"
    # results = []

    # for name, df in df_dict:
    # print("DEBUG:", res, freq)

    # res_value = df[[res]].unique()
    # freq_value = df[[freq]].unique()

    base = eval.retrieve_base_stat(df, config)

    base["entropy"] = eval.calculate_entropy(df, config)
    base["gini"] = eval.calculate_gini(df, config)

    time_stab_corr, nan_stab, time_stab_diff = eval.estimate_time_stability(df, config)
    base["time_stability (correlation)"] = time_stab_corr
    base["nan_stability"] = nan_stab
    base["time_stability (difference)"] = time_stab_diff

    active_results = eval.compute_active_bins_per_cell(df, config)
    base.update(active_results)

    median_results = eval.compute_median_non_zero_per_cell(df, config)
    base.update(median_results)

    intense_results = eval.compute_event_intensity(df, config)
    base.update(intense_results)
    
    base["n_unique_cells"] = df[res_col].nunique()
    base["mean_per_cell"] = df.groupby(res_col)["n_accidents"].sum().mean()
                
        # results.append(base)

    return base 

# pd.concat(results)





def evaluate_h3_time_results(results: List):

    results_df = pd.DataFrame(results)

    clean_results = results_df[(results_df["non_zero_share"] > 0.05) & 
                               (results_df["rows"] > 1000)]
    
    print(f"Comparison shape 'result_df' vs. 'clean_results:\n{results_df.shape}\t{clean_results.shape}")

    if clean_results.empty:
        return {
            "success": False, 
            "results": results_df,
            "results_list": results
            }

    df_scored = eval.compute_score(clean_results)

    df_ranked = df_scored.sort_values("score", ascending=False)

    top_configs = df_ranked.head(10)
    
    return {
        "success": True, 
        "top": top_configs,
        "rank": df_ranked,
        "score": df_scored.to_dict('records')
        }


@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def find_best_h3_time(name):   
    # (1) load config + parse arguments
    gh.load_env_vars()
    
    data_processed = os.getenv("PATH_PROCESSED")
    report_folder = os.getenv("FOLDER_REPORT")

    config = get_yaml_config(name)
    arg_dict = config.get("general_args", {})
    data_folder = Path(arg_dict.get("data_folder", {}))
    file_suffix = arg_dict.get("file_suffix", ["clean"])
    df_folder = Path(f"{data_processed}/{data_folder}")

    res_range = config.get("geo_processing", {}).get("h3_values", [])

    time_processing = config.get("time_processing", {})
    time_col_new = time_processing.get("time_col_new", "datetime")
    freq_range = time_processing.get("freq_values", [])

    h3_time_dict = config.get("find_h3_time", {}) 
    cols_needed = h3_time_dict.get("necessary_cols", [])
    target_col = h3_time_dict.get("target_col", "n_accidents")

    # load report and df_dict 
    report = load_eda_summary(arg_dict)
    files = report.get("files", [])

    all_results = {}
    score_list = []
    for file in files:
        f_name = Path(file).stem

        df_list = []
        for f_suf in file_suffix:
            f_path = f"{report_folder}/{str(f_name).strip()}_{f_suf}.json"
            
            print("Start loading:", ph.shorten_path(f_path))

            try:
                df = pd.read_parquet(f_path)
                df_list.append(df)

                print("[DEBUG] df shape:", df.shape)
                # print("[DEBUG] NaN count in df:\n", df.isna().sum())
            except FileNotFoundError:
                print("File not found:", f_path)

        # merge dfs
        if len(df_list) != 2:
            print(f"Invalid count of dfs ({file}):", len(df_list))
            continue

        df_merge = df_list[0].merge(df_list[1], on="ID_accident", how="inner")
        print("[DEBUG] geo unique:", df_list[0]["ID_accident"].nunique())
        print("[DEBUG] time unique:", df_list[1]["ID_accident"].nunique())
        print("[DEBUG] merged unique:", df_merge["ID_accident"].nunique())

        df_red = df_merge[cols_needed].copy()

        print("[DEBUG] df_red shape:\t", df_red.shape)

        print("Completed df red")

        for freq in freq_range:
            freq_safe = time.check_translate_freq(freq)

            df_red = df_red.dropna(subset=[time_col_new])
            df_red[time_col_new] = pd.to_datetime(df_red[time_col_new], errors="coerce")
            
            print("[DEBUG] invalid datetime:", df_red[time_col_new].isna().sum())
            print(f"[DEBUG] freq | freq_safe:\t{freq} | {freq_safe}")
            
            if freq_safe in ["2W", "3W"]:
                df_red = time.create_n_weekly(df_red, time_col_new)
            
            else: 
                df_red[freq] = df_red[time_col_new].dt.to_period(freq_safe).dt.start_time

        print("[DEBUG] NaN count in df_red:")
        eda.count_nan_by_column(df_red)
        
        # before = len(df_red)
        # after = df_red.dropna(subset=["datetime"]).shape[0]
        # nan_sum = df_red["datetime"].isna().sum()
        # nan_mean = df_red["datetime"].isna().mean()

        # print("[DEBUG] checks on df red")
        # print(f"number and ratio of 'NaT':\t{nan_sum}\t{nan_mean}")
        # print("week bins:\n", df_red["datetime"].dt.to_period("W").head())

        # print(f"data loss by binning (before vs. after): {before}\t{after}")

        # results = {}
        results_list = []
        file_results = defaultdict(dict)

        for res in res_range:
            res_col = f"h3_res{res}"

            for freq in freq_range:
                df_tmp = df_red.copy()

                freq_col = "time_bin"

                config_optimise = {
                            "res": res_col,
                            "freq": freq,
                            "time_col_new": time_col_new,
                            "freq_col": freq_col,
                            "target_col": target_col
                            }

                print(f"{"="*15} PROCESSING: df={f_name} | res={res} | freq={freq} {"="*15}\n")
                
                # 1. time binning
                if freq in ["two_weekly", "three_weekly"]:
                    df_tmp["time_bin"] = df_tmp[freq]

                else:
                    df_tmp = time.apply_time_binning(df_tmp, config_optimise)

                # 2. drop NaT and NaN 

                df_tmp = df_tmp.dropna(subset=[freq_col, res_col])

                print("[DEBUG] df_tmp before agg:")
                print("[DEBUG] shape:\t", df_tmp.shape)
                # print("[DEBUG] head:\n", df_tmp.head(3), "\n")
                print("[DEBUG] unique h3:", df_tmp[res_col].nunique())
                print("[DEBUG] unique time:", df_tmp[freq_col].nunique())

                # 3. aggregation
                df_agg = dfh.aggregate_single(df_tmp, config_optimise)

                original = df_red["ID_accident"].nunique()
                tmp = df_tmp["ID_accident"].nunique()
                aggregated = df_agg["n_accidents"].sum()

                print("Check 'sum sanity':")
                print(f"original: {original}")
                print(f"after NaN_drop (df_tmp; total | diff): {tmp} | {original - tmp}")
                print(f"aggregated (df_agg): {aggregated}\n")
                # print("[DEBUG] df_agg shape:\t", df_agg.shape)
                # print("[DEBUG] df_agg head:\n", df_agg.head(3), "\n")

                # 4. inflation
                df_full = dfh.inflate_df(df_agg, config_optimise)
                
                print("[DEBUG] df_full shape:\t", df_full.shape)
                # print("[DEBUG] df_full head:\n", df_full.head(3), "\n")
                print("[DEBUG] non-zero after inflate:", (df_full["n_accidents"] > 0).sum())
                # df_filt = df_red[(df["resolution"] == res) &
                #                  df["resolution"] == freq]

                # 5. Evaluation
                print(f"Evaluating {res_col} - {freq}\n")
                metrics = h3_tbin_evaluation(df_full, config_optimise)

                results_list.append({
                                "freq": freq,
                                "res": res,
                                **metrics
                                })

                file_results[freq][res_col] = metrics

        eval_dict = evaluate_h3_time_results(results_list)

        if not eval_dict["success"]: # is None:
            print(f"'Eval_dict' of '{f_name}' is empty. Results_list will be added as 'eval_raw'.")
            eval_raw = eval_dict.get("results_list", [])

            file_results["eval_raw"] = eval_raw

            eval_key = f"eval_raw_{f_name.split('_')[1]}"
            score_list.append({eval_key: eval_raw})
            print(f"Head of 'results_df' ({f_name}):\n", eval_dict["results"])

        else:
            scores = eval_dict.get("score", [])

            file_results["scores"] = scores

            score_key = f"scores_{f_name.split('_')[1]}"
            score_list.append({score_key: scores})
            print(f"Head of 'score_df' ({f_name}):\n", eval_dict["top"])

        report_annual_path = Path(f"{report_folder}/best_h3_tbin_{f_name}.json")
        fh.save_dict(file_results, report_annual_path)

        all_results[f_name] = results_list

        print("Completed evaluation")
        
    # # 
    # all_years_eval = evaluate_h3_time_results(score_list)

    # if not all_years_eval["success"]:
    #     print("'all_years_eval' is empty. Report will be created with 'eval_complete_raw'.")
    #     eval_complete_raw = all_years_eval.get("results_list", [])

    #     all_results["eval_complete_raw"] = eval_complete_raw

    #     # eval_complete_key = f"eval_complete_raw_{f_name.split('_')[1]}"
    #     # score_list.append({eval_complete_key: eval_complete_raw})
    #     print(f"Head of 'results_df' ({f_name}):\n", all_years_eval["results"])


    # else:
    #     all_scores = all_years_eval.get("score", [])
    #     all_results["scores"] = all_scores
    #     print(f"Head of 'score_df' ({f_name}):\n", all_years_eval["top"])

    # generate JSON report from results
    report_path = Path(f"{report_folder}/best_h3_tbin_complete.json")
    fh.save_dict(score_list, report_path)

    return 


if __name__ == "__main__":
    find_best_h3_time()
 

        # # change df format
        # df_long = melt_h3(df_red, res_range, freq_range)
        # # print("[DEBUG] df_long shape:", df_long.shape)

        # df_time = melt_time(df_long, freq_range)
        # # print("[DEBUG] df_time shape:", df_time.shape)

        # df_agg = aggregate_all(df_time)
        # # print("[DEBUG] df_agg shape:", df_agg.shape)
        # # print("[DEBUG] sum sanity:", df_agg["n_accidents"].sum())
        

        # # print("Completed df melting to long format")

        # file_results = defaultdict(dict)

        # for (res, freq), df_sub in df_agg.groupby(["resolution", "freq"]):

        #     # print("[DEBUG] res:", res)
        #     # print("[DEBUG] freq:", freq)
        #     print("[DEBUG] df_sub shape:\n", df_sub.shape)
            
        #     # zero-inflating df
        #     df_infl = inflate_time_h3(df_sub, time_col="time_bin", h3_col="h3_index")

        #     print("[DEBUG] df_infl shape:\n", df_infl.shape)

    #         result = h3_tbin_evaluation(
    #                                 df_infl, 
    #                                 "h3_index", 
    #                                 "time_bin"
    #                                 )
    #         results_list.append({
    #                     "df_name": f_name,
    #                     "freq": freq,
    #                     "res": res,
    #                     **result
    #                     })

    #         file_results[freq][res] = result

    #     # h3_tbin_results[freq_col][res_col] = result
    #     report_annual_path = Path(f"{report_folder}/best_h3_tbin_{f_name}.json")
    #     fh.save_dict(file_results, report_annual_path)
                
    # #         # print("[DEBUG] length of 'result' / 'results_list':", len(result), len(results_list))
        
    #     print("Completed evaluation")

    #     all_results[f_name] = results_list

    # # generate JSON report from results
    # report_path = Path(f"{report_folder}/best_h3_tbin_complete.json")
    # fh.save_dict(all_results, report_path)

    # return 






    # # load env variables
    # gh.load_env_vars()

    # session.load_config(config)
    # log_name = session.log_name
    # name_logfile = session.log_file

    # # # load logger
    # logger = create_logger(name=log_name, file_name=name_logfile)

    # session.logger = logger

    # # setup DB_engine
    # engine = post.get_engine()

    # #
    # h3_values = session.h3_values
    # frequence = [session.freq]
    # inflate = session.inflate

    # h3_dict = {}
    # with engine.begin() as conn:
    #     for val in h3_values:
    #         for freq in frequence:
    #             table = f"accidents.h3_res{val}_{freq}{'_zeroinf' if inflate else ''}"

    #             df = retrieve_base_stat(table, conn)

    #             df["entropy"] = calculate_entropy(table, conn)
    #             df["gini"] = calculate_gini(table, conn)
    #             df["time_stability"] = estimate_time_stability(table, freq, conn)

    #             df["res"] = val
    #             df["freq"] = freq
    #             h3_dict[f"res_{val}"] = df

    # h3_stat_merge = pd.concat(h3_dict.values(), ignore_index=True)

    # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # describe_save_h3_df(h3_stat_merge, f_name="df_h3_eval", idx_new=["res", "freq"])

    # logger.info("Completed H3 resolution evaluation at %s", now)

    # return



# def describe_save_h3_df(df, f_name, idx_new=None):
#     # setup logger
#     logger = session.logger

#     #
#     if idx_new is not None:
#         df.set_index(idx_new, inplace=True)

#     # for col in ["mean", "var",
#     #             "std", "ratio_var_mean",
#     #              "zero_share",
#     #             "entropy", "gini",
#     #             "time_stability"]:
#     #     df[col] = df[col].astype(float).round(3)

#     for col in ["zero_count", "rows", "min", "median", "max"]:
#         df[col] = df[col].astype(int)

#     pd.set_option("display.float_format", "{:.3f}".format)

#     logger.info("h3_stat_merge -- INFO ---\n%s\n", df.info())
#     logger.info("h3_stat_merge -- OVERVIEW ---\n%s\n", df.T)

#     folder = os.getenv("PATH_PROCESSED")
#     df_path = f"{folder}/{f_name}.csv"
#     # df.to_csv(df_path)
#     logger.info("Saved df '%s' to %s", f_name, ph.shorten_path(df_path))

#     return
