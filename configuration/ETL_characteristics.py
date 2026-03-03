# from utils.llm_helper import batch_escalation_by_llm
# from configuration.prompt import prompt_v1, allowed_values_v1
# from configuration.json_scheme import scheme_v1
# import os
# import utils.general_helper as gh

config = {
    "log_name": "ETL_CHARACTERISTICS",
    "log_file": "etl_characteristics",
    "encode": {
        "Num_Acc": "id",
        "jour": "day",
        "mois": "month",
        "an": "year",
        "hrmn": "time (hr:mn)",
        "lum": "light conditions",
        "dep": "department",
        "com": "commune",
        "agg": "localisation",
        "int": "intersection type",
        "atm": "weather",
        "col": "collision type",
        "adr": "postal address",
        "lat": "latitude",
        "long": "longitude",
    },
    "num_feats": ["id", "year", "month", "day", "time", "longitude", "lattitude"],
    "cat_feats": [
        "light conditions",
        "localisation",
        "intersection type",
        "weather",
        "collision type",
        "commune",
        "postal address",
        "gps",
        "department",
    ],
}
