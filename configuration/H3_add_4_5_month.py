# from utils.llm_helper import batch_escalation_by_llm
# from configuration.prompt import prompt_v1, allowed_values_v1
# from configuration.json_scheme import scheme_v1
# import os
# import utils.general_helper as gh

config = {
    "log_name": "H3_CHARACTERISTICS",
    "log_file": "H3_characteristics",
    "h3_values": [4, 5],
    "cols_needed": [
        "id",
        "lat_norm",
        "lon_norm",
        "datetime",
        "light conditions",
        "intersection type",
        "weather",
        "collision type",
    ],
    "tmp_tbl": "tmp_h3",
    "freq": "month",
    "p_key": "id_accid",
    "scheme": "accidents",
    "src_table": "characteristics",
}
