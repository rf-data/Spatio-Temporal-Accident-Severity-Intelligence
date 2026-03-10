## eda_rules.py
# imports
from collections import defaultdict

import src.utils.agent_helper as ah
from src.core.finding_classes import ActionSchema
from src.core.report_classes import MergeStatement, MergeStrategy


def dtype_changes(df_old, df_new):

    common_cols = set(df_old.columns) & set(df_new.columns)

    changes = {}

    for col in common_cols:

        old_type = str(df_old[col].dtype)
        new_type = str(df_new[col].dtype)

        if old_type != new_type:
            changes[col] = {
                "old": old_type,
                "new": new_type
            }

    return changes


def check_columns(df_old, df_new):

    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    miss_cols = list(old_cols - new_cols)
    new_cols = list(new_cols - old_cols)
    col_dtype_changes = dtype_changes(df_old, df_new)

    return {
        "new_cols": new_cols,
        "missing_cols": miss_cols,
        "dtype_changes": col_dtype_changes
        }


def key_uniqueness(df, join_keys):

    total_rows = len(df)
    unique_keys = df[join_keys].drop_duplicates().shape[0]

    return {
        "unique_keys": unique_keys,
        "uniqueness_ratio": unique_keys / total_rows
        } 


def pre_merge_comparison(df_dict,
                        merge_analysis,
                        join_keys):
    print("Start 'pre_merge_comparison' on:", df_dict.keys())
    compare_dict = {
            "n_rows": {
                    "a": merge_analysis["n_rows_a"],
                    "b": merge_analysis["n_rows_b"]
                    },
            "dictinct_cols": {
                    "only_in_a": merge_analysis["only_in_a"],
                    "only_in_b": merge_analysis["only_in_b"]
                    },
            "overlap": defaultdict(dict),
            "col_check": defaultdict(dict),
            "key_uniqueness": defaultdict(dict)
            }
    
    names = list(df_dict.keys())
    df_a = df_dict[names[0]]
    df_b = df_dict[names[1]]

    compare_dict["col_check"] = check_columns(df_a, df_b)

    for j_key in join_keys:
        keys_a = df_a[[j_key]].drop_duplicates()
        keys_b = df_b[[j_key]].drop_duplicates()

        # merged = keys_a.merge(keys_b, on=j_key, how="inner")
        # overlap = len(merged)

        set_a = set(keys_a[j_key])
        set_b = set(keys_b[j_key])

        overlap = len(set_a & set_b)

        compare_dict["overlap"][j_key] = {
            "n_unique_keys_a": len(set_a),
            "n_unique_keys_b": len(set_b),
            "overlap_rows": overlap,
            "overlap_vs_A": overlap / len(set_a),
            "overlap_vs_B": overlap / len(set_b)
            }
        
        compare_dict["key_uniqueness"][names[0]][j_key] = key_uniqueness(df_a, 
                                                                         [j_key])
        compare_dict["key_uniqueness"][names[1]][j_key] = key_uniqueness(df_b, 
                                                                         [j_key])
               
    return compare_dict


def evaluate_merge_candidates(dfs: list, 
                              merge_analysis: dict,
                              config: dict):
    print("Start 'evaluate_merge_candidates'")
    results = {}

    for pair_name, analysis in merge_analysis.items():

        a = analysis["file_a"]
        b = analysis["file_b"]

        df_dict = {
            a: dfs[a],
            b: dfs[b]
            }

        common_cols = analysis["common_columns"]
        type_conflicts = analysis["type_conflicts"]
        cardinality = analysis["cardinality_mismatch"]

        # valid join candidates
        valid_keys = [
            col
            for col in common_cols
            if col not in type_conflicts and col not in cardinality
        ]

        # calculate 'expected overlap' + key_uniqueness
        compare_dict = pre_merge_comparison(df_dict, 
                                            analysis,
                                            valid_keys)

        merge_strategy =  generate_merge_strategy(compare_dict,
                                               config)
        
        results[pair_name] = MergeStatement(
                                    files = (a, b),
                                    valid_keys = valid_keys,
                                    overlap = compare_dict["overlap"],
                                    column_check = compare_dict["col_check"],
                                    key_uniqueness = compare_dict["key_uniqueness"],
                                    n_rows = {
                                        a: len(dfs[a]),
                                        b: len(dfs[b])
                                        },
                                    merge_strategy = merge_strategy
                                    )
    
        # {
        #     "files": (a, b),
        #     "valid_keys": valid_keys,
        #     "overlap": compare_dict["overlap"],
        #     "column_check": compare_dict["col_check"],
        #     "key_uniqueness": compare_dict["key_uniqueness"],
        #     "n_rows": {
        #         a: len(dfs[a]),
        #         b: len(dfs[b])
        #     },
        #     "merge_stratgy": merge_strategy
        # }

    return results


def select_join_key(compare_dict, expected_key=None): 
    print("Start 'select_join_key'")    

    uniqueness = compare_dict["key_uniqueness"]
    overlap = compare_dict["overlap"]

    # expected key available?
    if expected_key and expected_key in overlap:
        ratios = [
            uniqueness[file]\
            .get(expected_key, {})\
            .get("uniqueness_ratio", 0)
            for file in uniqueness
            ]

        if min(ratios) > 0.9:
            return {
                "best_key": expected_key,
                "all_scores": {expected_key: min(ratios)}
                }

    # fallback: find best join_key
    best_key = None
    best_score = 0
    score_dict = {}
        
    for key in overlap:
        ratios = [
            uniqueness[file].get(key, {}).get("uniqueness_ratio", 0)
            for file in uniqueness
            ]
        
        score = min(ratios)
        score_dict[key] = score

        if score > best_score:
            best_score = score
            best_key = key
    
    return {
            "best_key": best_key,
            "all_scores": score_dict
            }


def compute_schema_similarity(col_check):

    missing = len(col_check["missing_cols"])
    new = len(col_check["new_cols"])
    dtype = len(col_check["dtype_changes"])

    total_cols = missing + new + max(1, len(col_check["dtype_changes"]))

    penalty = (missing + new + dtype) / (total_cols + missing + new)

    similarity = max(0, 1 - penalty)

    return similarity


def compute_confidence(compare_dict, join_key):

    uniqueness = compare_dict["key_uniqueness"]
    unique_score = min(
                    [
                uniqueness[file][join_key]["uniqueness_ratio"]
                for file in uniqueness
                ]
                ) 
    
    overlap = compare_dict["overlap"]

    overlap_a = overlap[join_key].get("overlap_vs_A", None)
    overlap_b = overlap[join_key].get("overlap_vs_B", None)

    overlap_score = min(overlap_a, overlap_b)
    overlap_score_merge = 1 - float(overlap_score)

    similarity_score = compare_dict["similarity"]
    
    return {
        "merge": float(
                    0.4 * unique_score \
                    + 0.4 * overlap_score_merge \
                    + 0.2 * similarity_score
                    ),
        "append": float(
                    0.4 * unique_score \
                    + 0.4 * overlap_score_merge \
                    + 0.2 * similarity_score
                    )
        }


def generate_merge_strategy(compare_dict, dataset_config):

    print("Start 'generate_merge_strategy'")   

    schema = compare_dict["col_check"]
    overlap = compare_dict["overlap"]
    confidence = None

    compare_dict["similarity"] = compute_schema_similarity(schema)

    # uniqueness = compare_dict["key_uniqueness"]     # ["f_name"]["j_key"]

    expected_key = dataset_config["expectations"].get("expected_primary_key", 
                                                      None)

    results_join_key = select_join_key(compare_dict, expected_key)

    join_key = results_join_key["best_key"]

    if join_key:
        confidence = compute_confidence(compare_dict, join_key)

    else:
        return MergeStrategy(
            strategy = "manual_review",
            reason = "no valid join key",
            join_key = join_key,
            confidence = confidence,
            similarity = compare_dict["similarity"]
        )
    
    # -------------------------
    # Schema check
    # -------------------------
    if schema["missing_cols"]:
        return {
            "strategy": "reject",
            "reason": "schema mismatch",
            "join_key": join_key,
            "confidence": confidence,
            "similarity": compare_dict["similarity"]
        }

    # -------------------------
    # overlap analyse
    # -------------------------
    overlap_A = overlap[expected_key]["overlap_vs_A"]
    overlap_B = overlap[expected_key]["overlap_vs_B"]

    rows_A, rows_B = compare_dict["n_rows"].values()

    # disjoint datasets
    if overlap_A < 0.01 and overlap_B < 0.01:
        return {
            "strategy": "append",
            "join_key": join_key,
            "reason": "datasets disjoint",
            "confidence": confidence, 
            "similarity": compare_dict["similarity"]
        }

    # strong overlap
    if overlap_A > 0.8 and overlap_B > 0.8:
        if rows_A * rows_B > 5e7:
            return {
                "strategy": "reject",
                "reason": "join explosion risk",
                "join_key": join_key,
                "confidence": confidence, 
                "similarity": compare_dict["similarity"]
                }

        return {
            "strategy": "merge",
            "join_key": join_key,
            "confidence": confidence,
            "join_type": "inner",
            "similarity": compare_dict["similarity"]
        }

    return {
        "strategy": "manual_review",
        "join_key": join_key,
        "confidence": confidence,
        "reason": """
        unequivocal; overlap too less for 'merge', 
        overlap too much for 'append'
        """,
        "similarity": compare_dict["similarity"]
    }



# recommendations[pair_name] = {
#             "f_name_a": analysis["file_a"],
#             "f_name_b": analysis["file_b"],
#             "recommended_join_keys": valid_keys,
#             "expected_overlap": compare_dict["overlap"],
#             "column_check": compare_dict["col_check"],
#             "key_uniqueness": compare_dict["key_uniqueness"],
#             "join_type": join_type,
#             "pre_merge_actions": pre_merge_actions,
#             "post_merge_checks": post_merge_checks,
#         }


    #     # determine join type
    #     if len(valid_keys) == 0:
    #         join_type = None
    #     elif len(cardinality) > 0:
    #         join_type = "left"
    #     else:
    #         join_type = "inner"

    #     # pre-merge actions
    #     cast_col = ActionSchema(
    #                 action = "cast_column dtype",
    #                 target = []
    #                 )
    #     high_card = ActionSchema(
    #                 action = "reduce high_cardinality",
    #                 target = []
    #                 )

    #     for col in type_conflicts:
    #         cast_col.target.append(col)
            
    #     for col in cardinality:
    #         high_card.target.append(col)


    #     pre_merge_actions = {
    #         "cast_column": cast_col,
    #         "high_cardinality": high_card
    #     }

    #     # post-merge checks
    #     post_merge_checks = [
    #         "Validate row count after merge",
    #         "Check duplicate key combinations",
    #         "Re-evaluate missingness on key columns",
    #     ]

    #     recommendations[pair_name] = {
    #         "f_name_a": analysis["file_a"],
    #         "f_name_b": analysis["file_b"],
    #         "recommended_join_keys": valid_keys,
    #         "expected_overlap": compare_dict["overlap"],
    #         "column_check": compare_dict["col_check"],
    #         "key_uniqueness": compare_dict["key_uniqueness"],
    #         "join_type": join_type,
    #         "pre_merge_actions": pre_merge_actions,
    #         "post_merge_checks": post_merge_checks,
    #     }

    # return merge_strategy




    # if expected_key: 
    #     key_uni = min(
    #         uniqueness[next(iter(uniqueness))][expected_key]["uniqueness_ratio"],
    #         uniqueness[list(uniqueness.keys())[1]][expected_key]["uniqueness_ratio"]
    #     )

    #     if key_uni < 0.9:
    #         return {
    #             "strategy": "manual_review",
    #             "reason": "key not unique"
    #         }
    # else: 
    #     keys = []
    #     key_candidates = [k for k in keys if k["uniqueness_ratio"] > 0.9]

    # # -------------------------
    # # 3 Overlap Analyse
    # # -------------------------
    # if expected_key: 
        
    
    # else: 
    #     hi = ""
    # # rows_A = compare_dict["n_rows"].get("a")
    # # rows_B = compare_dict["n_rows"].get("b")

    # # if overlap_A < 0.01 and overlap_B < 0.01:

    # #     return {
    # #         "strategy": "append",
    # #         "reason": "datasets disjoint"
    # #     }

    # if overlap_A > 0.8 and overlap_B > 0.8:
        
    #     if rows_A * rows_B > 5e7:
    #         return {
    #             "strategy": "reject",
    #             "reason": "join explosion risk"
    #         }

    #     return {
    #         "strategy": "merge",
    #         "join_key": expected_key,
    #         "join_type": "inner"
    #     }

    # return {
    #     "strategy": "manual_review",
    #     "reason": "partial overlap"
    # }

# {
#   "merge_decision": {
#     "strategy": "append",
#     "join_key": "Num_Acc",
#     "confidence": 0.92,
#     "reason": "no overlap between yearly datasets"
#   }
# }

