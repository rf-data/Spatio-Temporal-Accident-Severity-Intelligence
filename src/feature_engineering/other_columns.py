## geo_columns
# import
import pandas as pd
import numpy as np
from pyproj import Transformer
from collections import defaultdict
import h3
from typing import List

import src.utils.visualisation_helper as viz


# -------------------
# DATASET 'VEHICLE'
# -------------------
# --> drop: public_transport_occupancy, motor, flow_direction
def add_maneuver_feats(df):
    """
    maneuver_map = {
    -1: "unknown",
     0: "unknown",
     1: "straight",
     2: "same_direction_same_lane",
     3: "between_lanes",
     4: "reverse",
     5: "wrong_way",
     6: "crossing_median",
     7: "bus_lane_same_direction",
     8: "bus_lane_opposite_direction",
     9: "merging",
    10: "u_turn",

    11: "lane_change_left",
    12: "lane_change_right",

    13: "drift_left",
    14: "drift_right",

    15: "turn_left",
    16: "turn_right",

    17: "overtake_left",
    18: "overtake_right",

    19: "crossing_road",
    20: "parking_maneuver",
    21: "avoidance",
    22: "door_opening",
    23: "stopped",
    24: "parked",
    25: "driving_on_sidewalk",
    26: "other"
    }
    """
    # feats from 'maneuver'
    df["is_turning"] = df["maneuver"].isin([10,15,16])
    df["is_lane_change"] = df["maneuver"].isin([11,12])
    df["is_overtaking"] = df["maneuver"].isin([17,18])

    return df


def add_vehicle_type_feats(df):
    """
    0: "undetermined",
    1: "bicycle",
    2: "moped_lt_50cc",
    3: "light_quadricycle",
    7: "car",
    10: "light_commercial_vehicle",
    13: "truck_medium",
    14: "truck_heavy",
    15: "truck_with_trailer",
    16: "tractor",
    17: "tractor_semi_trailer",
    20: "special_vehicle",
    21: "agricultural_vehicle",
    30: "scooter_lt_50cc",
    31: "motorcycle_50_125cc",
    32: "scooter_50_125cc",
    33: "motorcycle_gt_125cc",
    34: "scooter_gt_125cc",
    35: "quad_light",
    36: "quad_heavy",
    37: "bus",
    38: "coach",
    39: "train",
    40: "tram",
    41: "3_wheel_lt_50cc",
    42: "3_wheel_50_125cc",
    43: "3_wheel_gt_125cc",
    50: "motorized_personal_transport",
    60: "non_motorized_personal_transport",
    80: "e_bike",
    99: "other"
    """
    # feats from 'vehicle_type'
    df["is_vulnerable_vehicle"] = df["vehicle_type"].isin([1, 2, 30, 
                                                        31, 32, 33, 
                                                        34, 80])
    df["is_heavy_vehicle"] = df["vehicle_type"].isin([13, 14, 15, 
                                                16, 17, 21])

    return 


def add_obstacle_feats(df):
    """
    obs_map = {
    -1: "unknown",
     0: "none",
     1: "parked_vehicle",
     2: "tree",
     3: "metal_guardrail",
     4: "concrete_guardrail",
     5: "other_guardrail",
     6: "building_wall",
     7: "sign_or_emergency_post",
     8: "pole",
     9: "street_furniture",
    10: "parapet",
    11: "island_refuge",
    12: "curb",
    13: "ditch_slope",
    14: "other_on_road",
    15: "other_off_road",
    16: "no_obstacle_runoff",
    17: "culvert"
    }
    obsm_map = {
    -1: "unknown",
     0: "none",
     1: "pedestrian",
     2: "vehicle",
     4: "rail_vehicle",
     5: "domestic_animal",
     6: "wild_animal",
     9: "other"
    }
    """

    # obs + obsm    --> collision_object
    df["collision_object"] = (df["fixed_obstacle"].astype(str) 
                              + "_" + 
                              df["mobile_obstacle"].astype(str))
    
    # df["collision_object"] = ("A_" + df["fixed_obstacle"].astype(str))
    # df["collision_object"] = ("B_" + df["mobile_obstacle"].astype(str))

    df["hit_another_vehicle"] = (df["mobile_obstacle"] == 2 | df["fixed_obstacle"] == 1)
    df["hit_a_pedestrian"] = (df["mobile_obstacle"] == 1)

    return df


def add_veh_id_feats(df):

    df["is_first_vehicle"] = df["num_veh"] == 1
    df["n_vehicles"] = df.groupby("accident_id")["num_veh"].nunique()
    
    return df


def add_collision_feats(df):
    """
    choc_map = {
    -1: "unknown",
     0: "none",
     1: "front",
     2: "front_right",
     3: "front_left",
     4: "rear",
     5: "rear_right",
     6: "rear_left",
     7: "right_side",
     8: "left_side",
     9: "multiple_rollover"
    }
    """
    df["is_frontal_impact"] = (df["impact_point"].isin[1,2,3])
    df["is_rear_impact"] = (df["impact_point"].isin[4,5,6])
    df["is_side_impact"] = df["impact_point"].isin([7,8])
    df["is_rollover"] = (df["impact_point"] == 9)

    return df


def add_complex_feats(df):

    df["risk_pattern"] = (
                    df["maneuver"].astype(str) + "_" +
                    df["choc"].astype(str)
                    )

    return df


# -------------------
# DATASET 'PLACES'
# -------------------
# --> drop: median_width, road_width


"""
road_gradient_map = {
    -1: "unknown",
     1: "flat",
     2: "uphill",
     3: "downhill"
    }

raod_geometry_map = {
    -1: "unknown",
     0: "none",
     1: "bike_lane",
     2: "bus_lane",
     3: "emergency_lane"
    }

traffic_regime_map = {
    -1: "unknown",
     1: "one_way",
     2: "two_way",
     3: "separate_carriageways"
    }

reserved_lane_map = {
    -1: "unknown",
     0: "none",
     1: "bike_lane",
     2: "bus_lane",
     3: "emergency_lane"
    }

accident_location_map = {
    -1: "unknown",
     1: "roadway",
     2: "emergency_lane",
     3: "shoulder",
     4: "sidewalk",
     5: "bike_lane",
     6: "other"
    }    
"""

def add_surface_conditions(df):
    """
    surf_map = {
    -1: "unknown",
     1: "normal",
     2: "wet",
     3: "puddles",
     4: "flooded",
     5: "snow",
     6: "mud",
     7: "ice",
     8: "oil",
     9: "other"
    }
    """



    return 


def is_highway(df):
    """
    catr_map = {
    1: "highway",
    2: "national_road",
    3: "departmental_road",
    4: "communal_road",
    5: "off_public_network",
    6: "parking_area",
    7: "urban_road",
    9: "other"
    }

    """
    
    assert (df["road_type"] == 1) & (df["road_number"].str.startswith("A"))

    return 

def add_road_feats(df):
    
    df["is_high_speed"] = df["vma"] >= 90
    df["is_urban_speed"] = df["vma"] <= 50

    df["is_slippery"] = df["surf"].isin([2,3,4,5,6,7,8])    
    df["is_curve"] = df["road_geometry"].isin([2,3,4])
    df["risk_speed_curve"] = (
                        (df["vma"] >= 80) &
                        (df["is_curve"] == 1)
                        )

    df["is_multilane"] = df["n_lanes"] > 2
    df["is_complex_infra"] = df["infrastructure"].isin([3,5])

    # aus top feats FeatImpoartance or PermImportance 
    # df["is_risky_road"] = (
    #                 (df["is_curve"]) &
    #                 (df["is_slippery"]) &
    #                 (df["n_lanes"] <= 2) &
    #                 (df["vma"] >= 80)
    #                 )
    
    return df


def add_urban_feats(df):
    """
    reserved_lane_map = {
    -1: "unknown",
     0: "none",
     1: "bike_lane",
     2: "bus_lane",
     3: "emergency_lane"
    }

    near_school_map = {
    -1: "unknown",
     0: "no",
     1: "yes"
    }
    """
    df["is_urban"] = df["road_type"].isin([4,7])


    return 


# df["risk_speed_curve"] = (
#     (df["vma"] >= 80) &
#     (df["plan"].isin([2,3,4]))
# )
# df["is_high_speed"] = df["vma"] >= 90
# df["is_urban_speed"] = df["vma"] <= 50
# df["is_curve"] = df["plan"].isin([2,3,4])
# df["is_slippery"] = df["surf"].isin([2,3,4,5,6,7,8])
# df["is_urban"] = df["catr"].isin([4,7])
# df["is_complex_infra"] = df["infra"].isin([3,5])

# df["high_risk_road"] = (
#     (df["is_curve"]) &
#     (df["is_slippery"]) &
#     (df["nbv"] <= 2)
# )

# -------------------
# DATASET 'PERSONS'
# -------------------
"""
--> change order to achieve 'ordinality' (unharmed > minor_injury > hospitalized > killed)


trajet_map = {
    -1: "unknown",
     0: "unknown",
     1: "home_to_work",
     2: "work_to_home",
     3: "work_related",
     4: "school",
     5: "shopping",
     6: "leisure",
     7: "other"
    }   

"""

def correct_injury_order(df):
    """
    injury_severity_map = {
    1: "unharmed",
    2: "killed",
    3: "hospitalized",
    4: "minor_injury"
    }
    """
    injury_dict = {
            "1": 1,
            "2": 4,
            "3": 3,
            "4", 2
            }
    
    df["injury_sev_corr"] = (df["injury_severity"]
                             .astype(str)
                             .map(injury_dict))
    
    df = df.drop(columns=["injury_severity"])

    return df


# def add_person_count():
    # """
    # person_type_map = {
    # 1: "driver",
    # 2: "passenger",
    # 3: "pedestrian"
    # }

    # sexe_map = {
    # 1: "male",
    # 2: "female",
    # -1: "unknown"
    # }
    # """
    # n_driver,
    # n_passenger,
    # n_pedestrian
#     return 

def add_pedestrian_feats(df):
    """
    etatp_map = {
    -1: "unknown",
     1: "alone",
     2: "group"
    }

    locp_map = {
    -1: "unknown",
     0: "not_applicable",
     1: "on_road",
     2: "on_sidewalk",
     3: "on_crosswalk",
     4: "on_crosswalk_with_signal",
     5: "on_crosswalk_without_signal",
     6: "other"
    }

    actp_map = {
    -1: "unknown",
     0: "not_applicable",
     1: "walking_same_direction",
     2: "walking_opposite_direction",
     3: "crossing",
     4: "running",
     5: "standing",
     6: "other"
    }  
    """
    df["risk_profile"] = (
        (df["catu"] == 3) &
        (df["locp"].isin([3,4,5])) &
        (df["actp"] == 3)
    )

    return 


def add_age_feats(df):
    df["age"] = df["year"] - df["birth_year"]

    df["is_young"] = df["age"] < 25
    df["is_elderly"] = df["age"] > 65
    # df["age_bin"] = np.where(df["age"] <= 25,
    #                          0,
    #                          np.where(df["age"] <= 35,
    #                                   1,
    #                                   np.where(df["age"] <= 45,
    #                                            2,
    #                                            np.where(df["age"] <= 55,
    #                                                     3,
    #                                                     np.where(df["age"] <= 65,
    #                                                              4, 
    #                                                              5)
    #                                                     )
    #                                             )
    #                                 )
    #                         )

    df = df.drop(columns=["birth_year", "age"])
    return df

def extract_equipment_flags(df):
    """
    secu_map = {
    -1: "unknown",
     0: "none",
     1: "seatbelt",
     2: "helmet",
     3: "child_seat",
     4: "reflective_vest",
     5: "other"
    }
    """
    if (df["secu1"]) == 1 or (df["secu2"] == 1) or (df["secu3"] == 1):
        df["has_seatbelt"] = 1
    else:
        df["has_seatbelt"] = 0

    if (df["secu1"]) == 2 or (df["secu2"] == 2) or (df["secu3"] == 2):
        df["has_helmet"] = 1
    else:
        df["has_helmet"] = 0

    if (df["secu1"]) == 3 or (df["secu2"] == 3) or (df["secu3"] == 3):
        df["has_child_seat"] = 1
    else:
        df["has_child_seat"] = 0

    # df["is_protected"] = (
    #             (df["has_seatbelt"] == 1) |
    #             (df["has_helmet"] == 1)
    #             )
    # # oder
    # df["n_safety_measures"] = (
    #                     (df["secu1"] > 0).astype(int) +
    #                     (df["secu2"] > 0).astype(int) +
    #                     (df["secu3"] > 0).astype(int)
    #                     )
    
    # df["has_any_equipment"] = ...
    # df["safety_level"] = ...
    
    return df 


def parse_old_secu(val):
    """
    
    """
    if val in [-1, 0]:
        return {"has_equipment": 0, "used_equipment": 0}

    val = str(int(val)).zfill(2)

    return {
        "has_equipment": int(val[0]),
        "used_equipment": int(val[1])
        }
