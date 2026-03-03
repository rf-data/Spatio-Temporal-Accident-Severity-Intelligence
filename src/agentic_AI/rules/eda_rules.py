## eda_rules.py
# imports
import src.utils.agent_helper as ah

from src.core.finding_classes import ActionSchema

# class ActionSchema(BaseModel):
#     action: str
#     target: List[str] | None = None
#     params: dict | None = None

#     model_config = {
#         "extra": "forbid"
#         }

def recommend_merge_strategy(merge_analysis: dict):

    recommendations = {}

    for pair_name, analysis in merge_analysis.items():

        common_cols = analysis["common_columns"]
        type_conflicts = analysis["type_conflicts"]
        cardinality = analysis["cardinality_mismatch"]

        # valid join candidates
        valid_keys = [
            col
            for col in common_cols
            if col not in type_conflicts and col not in cardinality
        ]

        # determine join type
        if len(valid_keys) == 0:
            join_type = None
        elif len(cardinality) > 0:
            join_type = "left"
        else:
            join_type = "inner"

        # pre-merge actions
        cast_col = ActionSchema(
                    action = "cast_column dtype",
                    target = []
                    )
        high_card = ActionSchema(
                    action = "reduce high_cardinality",
                    target = []
                    )

        for col in type_conflicts:
            cast_col.target.append(col)
            
        for col in cardinality:
            high_card.target.append(col)


        pre_merge_actions = {
            "cast_column": cast_col,
            "high_cardinality": high_card
        }

        # post-merge checks
        post_merge_checks = [
            "Validate row count after merge",
            "Check duplicate key combinations",
            "Re-evaluate missingness on key columns",
        ]

        recommendations[pair_name] = {
            "file_a": analysis["file_a"],
            "file_b": analysis["file_b"],
            "recommended_join_keys": valid_keys,
            "join_type": join_type,
            "pre_merge_actions": pre_merge_actions,
            "post_merge_checks": post_merge_checks,
        }

    return recommendations


# {
#     "log_transform_candidates": [...],
#     "drop_candidates": [...],
#     "datetime_candidates": [...],
#     "high_cardinality": [...]
# }

# def recommend_from_duplicates(metrics):

#     recs = []

#     for df_name, observation in duplicates.items():
#         for
#         if metric ==
#     # duplicates
#     duplicate_ratio_general = duplicates["duplicate_ratio (general)"]
#     duplicate_ratio_group = duplicates["duplicate_ratio (grouping)"]

#     if duplicate_ratio_general and duplicate_ratio_general > 0.01:
#         recs.append("High duplicates ratio (general)")
#     if duplicate_ratio_group and duplicate_ratio_group > 0.01:
#         recs.append("High duplicates ratio (gropuing)")

#     return


# def recommend_from_missing(metrics):


#     # missing values
#     for col, miss in missing.items():
#         if miss > 0.5:
#             recs.append(f"High missing {col}")

#     return


def recommend_from_distribution(skew_col: dict, kurt_col: dict):

    
    skew_scheme = ActionSchema(
                action = "handle_skewness",
                target = []
                ) 
    kurt_scheme = ActionSchema(
                action = "handle_kurtosis",
                target = []
        )

    # skewness
    # for idx, skew in skew_idx.items():
    #     if abs(skew) > 2:
    #         recs[f"skewness_idx_{idx}"] = idx

    for col, skew in skew_col.items():
        if abs(skew) > 2:
            skew_scheme.target.append(col)

    # kurtosis
    # for idx, kurt in kurt_idx.items():
    #     if kurt > 10:
    #         recs[f"kurtosis_idx_{idx}"] = idx

    for col, kurt in kurt_col.items():
        if kurt > 10:
            kurt_scheme.target.append(col)

    recs = {
        "skewness": skew_scheme,
        "kurtosis": kurt_scheme 
        }
    
    return recs


def filter_recommendations(params, config):
    processing = []

    for p in params:
        hint = p.recommendations_hint
        # ah._get_value(p)
        # hint = getattr(p, "recommendation_hint", None) or {}
        if not isinstance(hint, dict):
            continue

        miss = hint.get("missing", None)
        inf = hint.get("infinite", None)
        zero = hint.get("zero_inf", None)
        dup = hint.get("duplicates", None)
        dist = hint.get("Skew_Kurt", None)
        dt_cand = hint.get("dt_candidates", None)
        gr_cand = hint.get("cardinality", None)

        if miss:
            processing.append(
                ActionSchema(
                    action = "impute_nan", 
                    target = miss,
                    params  = config["missing_analysis"].get("params", {})
                    ))
        if inf:
            processing.append(
                ActionSchema(
                    action = "impute_inf", 
                    target = inf,
                    params  = config["infinite_analysis"].get("params", {})
                    ))
        if zero:
            processing.append(
                ActionSchema(
                    action = "handle_zero_inflation", 
                    target = zero,
                    params  = config["zero_inflation_analysis"].get("params", {})
                    ))
                
        if dup:
            processing.append(
                ActionSchema(
                    action = "drop_duplicates", 
                    target = dup,
                    params  = config["duplicate_analysis"].get("params", {})
                    ))
    

        if dist and isinstance(dist, dict):
            for name, act_sch in dist.values():
                # parts = str(name).split("_", 1)
                # if len(parts) != 2:
                #     continue

                if name == "skewness":
                    processing.append(
                                act_sch
                                )
                        
                if name == "kurtosis":
                    processing.append(
                                act_sch
                                )
                    
        if dt_cand:
            processing.append(
                ActionSchema(
                    action = "parse_datetime", 
                    target = dt_cand,
                    params  = config["detect_datetime_candidates"].get("params", {})
                    ))
            
        if gr_cand:
            processing.append(
                ActionSchema(
                    action = "cardinality_action", 
                    target = gr_cand,
                    params  = config["categorical_summary"].get("params", {})
                    ))

    return processing


# if isinstance(numeric["recommendation_hint"], dict):
#             res = numeric["recommendation_hint"].get("Skew_Kurt", None)

#             for rec_name in res.keys():
#                 if str(rec_name).split("_")[0] == "skewness":
#                     scaling[name] = rec_name
#                 if str(rec_name).split("_")[0] == "kurtosis":
#                     log_transform[name] = rec_name


#             impute_inf[name]


#     metrics = {
#         "duplicates": duplicates,
#         "missing": missing,
#         "num_sum": num_sum,
#         "cat_num": cat_sum,
#         "zero_inf": zero_inf,
#         "dt_candidates": dt_candidates
#         }

#     recommend_data_processing(metrics)
#                                                     # duplicates,
#                                                     # missing,
#                                                     # num_sum,
#                                                     # zero_inf,
#                                                     # dt_candidates)
