# ,id,lat_norm,lon_norm,datetime,light conditions,
# intersection type,weather,collision type,h3_res4,h3_res5,h3_res6,h3_res7

config = {
    "general_parameters": {
        "log_name": "LOGREG_BASELINE_PHASE_1 (CYCLIC)",
        "log_file": "LogReg_base_p1",
        "run_name": "",
    },
    "preparation_parameters": {
        "df_name": "df_h3_pre-weekly",
        "df_prep_name": "lr_prep_p1_ZeroInf_cyclic",
        "frequency": "month",
        "inflate": True,
        "h3_idx": "h3_res5",
        "cols_needed": [
            "id",
            "lat_norm",
            "lon_norm",
            "datetime",
            # "light conditions",
            # "intersection type",
            # "weather",
            # "collision type"
        ],
        "num_feats": ["h3_res5_te"],
    },
    "sql_parameters": {
        "tmp_tbl": "tmp_h3",
        "p_key": "id_accid",
        "scheme": "accidents",
        "src_table": "characteristics",
    },
    "experiment_parameters": {
        "split_method": "simple",
        "n_splits": None,
        "train_size": float(0.2),
        "class_weight": "balanced",
        "l1_ratio": float(0.0),
        "solver": "liblinear",
        "random_state": int(42),
        "max_iter": int(1000),
        "cyclic_encode": True,
        "encode_time": "month",
        "encode_space": "h3_res5",
        "max_lag": int(3),
        "features": [
            "lag_1",
            "lag_2",
            "lag_3",
            # "lag_12",
            # "roll_mean_3",
            # "roll_mean_12",
            # "roll_sum_3",
            # "roll_sum_12",
            "month",
            "year",
            "zero_streak",
        ],
        "q75": 1,
        "q95": 3,
        "period_dict": {"week": "W", "month": "M"},
        "target_init": "n_accidents",
        "target_final": "risk_next",
        "time_col": "period",
        "classification": "binary",  # multi_class
        # "split_col_names": [
        #         "weath_",
        #         "inter_",
        #         "coll_",
        #         "light_"
        #         ]
    },
}
# "year",
#                      "month",
#                      "weekday",
#                      "week"]:
# "h3_values": [4, 5, 6, 7, 8, 9],
#     "cols_needed": [
#             "id",
#             "lat_norm",
#             "lon_norm",
#             "datetime",
#             "light conditions",
#             "intersection type",
#             "weather",
#             "collision type"
#             ],
#
