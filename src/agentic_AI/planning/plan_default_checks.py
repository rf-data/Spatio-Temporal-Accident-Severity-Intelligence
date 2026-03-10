## plan_default_checks.py
# imports


def plan_default_checks(check_registry, dataset_config):

    structure = dataset_config["structure"]

    checks = []

    if structure["table"]:
        checks += check_registry.get_by_category(
                                            "tabular",
                                            default_only=True,
                                            eda_only=True
                                            )

    if structure["image"]:
        checks += check_registry.get_by_category(
                                            "image",
                                            default_only=True,
                                            eda_only=True
                                            )

    if structure["video"]:
        checks += check_registry.get_by_category(
                                            "video",
                                            default_only=True,
                                            eda_only=True
                                            )

    if structure["time_series"]:
        checks += check_registry.get_by_category(
                                            "timeseries",
                                            default_only=True,
                                            eda_only=True
                                            )

    if structure["geo_data"]:
        checks += check_registry.get_by_category(
                                            "geo",
                                            default_only=True,
                                            eda_only=True
                                            )

    if structure["cross_file"]:
        checks += check_registry.get_by_category(
                                            "cross_file",
                                            default_only=True,
                                            eda_only=True
                                            )
    
    return checks


# def plan_default_checks(check_registry,
#                         dataset_config):

#     # default_tools = ""
#     table_checks = [c.get() for c in all_checks 
#                      if c.category == ""] if table else []
#     img_checks = [c.get() for c in all_checks 
#                    if c.category == ""] if img else []
#     video_checks = [c.get() for c in all_checks 
#                   if c.category == ""] if video else []
#     ts_checks = [c.get() for c in all_checks 
#                   if c.category == ""] if ts else []
#     geo_checks = [c.get() for c in all_checks 
#                    if c.category == ""] if geo else []
    
#     if dataset_config["related_datasets"]:
#         checks += check_registry.get_by_category("cross_file")

#     return table_checks + img_checks + video_checks + ts_checks + geo_checks