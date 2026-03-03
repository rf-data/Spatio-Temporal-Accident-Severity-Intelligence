# ,id,lat_norm,lon_norm,datetime,light conditions,
# intersection type,weather,collision type,h3_res4,h3_res5,h3_res6,h3_res7

config = {
    "general_parameters": {
        "log_name": "Phase2A_BASE_LOGREG (INFLUENCE ON 'COLL_TYPE')",
        "log_file": "phase_2A_base_LogReg",
    },
    "preparation_parameters": {
        "df_name": "df_h3_pre-weekly",
        "df_prep_name": "df_p2_base_coll",
        "cols_needed": [
            "datetime",
            "light conditions",
            "intersection type",
            "weather",
            "collision type",
        ],
        "num_feats": [],
        "cat_feats": ["weather", "light conditions", "intersection type"],
    },
    "sql_parameters": {},
    "experiment_parameters": {
        "test_size": float(0.2),
        # "class_weight": "balanced",
        # "multi_class": "multinomial", # or "ovr"
        # "l1_ratio": float(0.0),
        "solver": "lbfgs",
        "random_state": int(42),
        "max_iter": int(500),
        "target_col": "collision type",
        "encode_time": "week",
        "features": ["week", "weekday", "hour"],
        "n_perm": int(5),
        "perm_score": "neg_log_loss",
        # "target_final": "risk_next",
        # "classification": "binary",     # multi_class
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
