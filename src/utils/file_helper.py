# import yaml
import json
import os
from pathlib import Path
import inspect
import torch
import joblib
import numpy as np
from datetime import datetime
from functools import reduce
import io
import ast
import yaml

# import hashlib

import src.utils.general_helper as gh
import src.utils.path_helper as ph

import src.core.logger as log
from src.core.session import session


# -----------------------
# CONFIGURATION METHODS
# -----------------------
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_yaml_config(name):
    # (1) load config + parse arguments
    gh.load_env_vars()
    
    config_folder = os.getenv("CONFIG_PATH")
    
    config_path = Path(config_folder) / f"{name}.yaml"

    print(f"Loading config_file:", ph.shorten_path(config_path))
    config = load_yaml_config(config_path)

    return config 

# -----------------
# MODEL METHODS
# -----------------
def save_model(model, name):
    # setup logger
    logger = session.logger

    #
    folder = os.getenv("PATH_MODEL")
    model_path = Path(f"{folder}({name}.joblib)")
    joblib.dump(model, model_path)

    logger.info("Model saved as %s", ph.shorten_path(model_path))

    return


def load_model(name):
    # setup logger
    logger = session.logger

    #
    folder = os.getenv("PATH_MODEL")
    model_path = Path(f"{folder}({name}.joblib)")
    model = joblib.load(model_path)

    logger.info("Model loaded from %s", ph.shorten_path(model_path))

    return model


# ---------------------
# (C) DF PREVIEW / EDA
# ---------------------


def df_preview(data: dict, logger=None):
    """
    Docstring for data_preview

    """
    # setup logger
    if logger is None:
        logger = session.logger

    for name, df in data.items():
        if logger:
            log.log_section(logger, f"EDA RAW DATA ({name})")
            logger.info("SHAPE:\t %s", df.shape)
            logger.info("INFO:\n %s", info_as_string(df))
            logger.info("TOTAL NAN COUNT:\n %s", df.isna().sum())
            # logger.info("TOTAL NP-INF COUNT:\n %s", np.sum(np.isinf(df)))
            logger.info("HEAD:\n %s \n", df.head(5))

        else:
            print("\n")
            print("=" * 50 + "\n")
            print(
                f"--- EDA RAW DATA ({name}) --- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ---\n"
            )
            print("=" * 50 + "\n")
            print("SHAPE:\t", df.shape)
            print("INFO\n", info_as_string(df))
            print("TOTAL NAN COUNT:\n %s", df.isna().sum())
            print("TOTAL NP-INF COUNT:\n %s", np.sum(np.isinf(df)))
            print(f"HEAD:\n{df.head(5)}\n")

    return


def info_as_string(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


# ---------------------
# DICT / JSON METHODS
# ---------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if inspect.isfunction(obj):
        return gh.snapshot_single_function(obj)
    # if isinstance(obj, torch.Tensor):           # Tensor handling
    #     return obj.detach().cpu().numpy().tolist()
    # if isinstance(obj, torch.device):
        # return str(obj)
    return obj


def save_dict(data: dict, path: Path) -> None:

    path= ph.ensure_dir(path)
    data_new = make_json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        try:
            json.dump(
                data_new,
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            print(f"Dict saved as {ph.shorten_path(path, 3)}")
            
        except TypeError as e:
            print(f"ERROR (non_serializable):\n{e}\n", type(data_new), repr(data_new))


def append_json(data: dict, path: Path) -> None:

    path= ph.ensure_dir(path)
    data_new = make_json_safe(data)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data_new) + "\n")
        print(f"Appending data on {ph.shorten_path(path, 3)}")


def load_dict(path: Path | str) -> dict:
    path= ph.ensure_dir(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Dict loaded: {ph.shorten_path(path, 3)}")
        return data


# -----------------
# TEXT METHODS
# -----------------


def save_text(path: Path, data: str):
    # make_text_safe(data)
    data_new = str(data)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data_new)
    print(f"Text file saved as {ph.shorten_path(path, 3)}")


"""
logger.log_text(
    "red_flags.yaml",
    yaml.safe_dump(RED_FLAGS, sort_keys=True, allow_unicode=True)
)
"""
