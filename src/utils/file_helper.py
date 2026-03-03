# import yaml
import json
import os
from pathlib import Path
import inspect
import joblib
import numpy as np
from datetime import datetime
from functools import reduce
import pandas as pd
from typing import List
import io
import ast
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# import hashlib

import src.utils.general_helper as gh
import src.utils.path_helper as ph

import src.core.logger as log
from src.core.session import session


# -----------------------
# CONFIGURATION METHODS
# -----------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


# -----------------
# DATAFRAME METHODS
# -----------------
# (A) LOAD DFS
# -----------------


def load_dfs(paths: List[str | Path] | str | Path, index_col: str | None = None):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    dfs_dict = {}
    for p in paths:
        df = pd.read_csv(p, index_col=index_col)
        dfs_dict[p] = df

    return dfs_dict


def read_french_csv_smart(path: str) -> pd.DataFrame:
    sep = detect_delimiter(path)
    df = pd.read_csv(
        path,
        sep=sep,
        encoding="latin1",
        decimal=",",
        engine="python",
        on_bad_lines="skip",
        # low_memory=False
    )

    return df


def load_files_from_folder(folder):
    # setup logger
    logger = session.logger

    folder = Path(folder)

    df_dict = {}
    for i, file in enumerate(folder.iterdir()):
        df = read_french_csv_smart(file)
        df = fix_single_column_df(df)
        df_re = df.rename(columns={"Accident_Id": "Num_Acc", "accident_id": "Num_Acc"})

        logger.info("shape:\t%s", df_re.shape)
        logger.info("columns:\t%s", df_re.columns)
        logger.info("head:\n%s", df_re.head(3))
        # logger.info("")

        df_dict[f"df_{i+1}"] = df_re

    return df_dict


def detect_delimiter(path: str) -> str:
    with open(path, encoding="latin1") as f:
        header = f.readline()
    if "\t" in header:
        return "\t"
    elif ";" in header:
        return ";"
    elif "," in header:
        return ","
    else:
        raise ValueError(f"Unknown delimiter in {path}")


# -----------------
# (B) REPAIR DFS
# -----------------


def fix_single_column_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] != 1:
        return df

    header_raw = df.columns[0]
    # print("cols to fix:\t", df.columns, "\n", col)

    if "," in header_raw:
        sep = ","
        # return df[col].str.split(",", expand=True)
    elif "\t" in header_raw:
        sep = "\t"
        # return df[col].str.split("\t", expand=True)
    else:
        # nothing to unpack
        return df
        # raise ValueError("Single column but no obvious delimiter")

    # Create header list
    header = [h.strip() for h in header_raw.split(sep)]

    # Split the ONLY data column into multiple columns
    data = df.iloc[:, 0].astype(str).str.split(sep, expand=True)

    # If file has trailing separators, data may have more cols than header
    if data.shape[1] > len(header):
        # keep only expected cols; drop the rest (usually empty)
        data = data.iloc[:, : len(header)]

    elif data.shape[1] < len(header):
        # data has fewer cols than header -> pad with NA
        for _ in range(len(header) - data.shape[1]):
            data[data.shape[1]] = pd.NA

    data.columns = header

    # Normalize common id column variants
    data = data.rename(columns={"Accident_Id": "Num_Acc", "accident_id": "Num_Acc"})

    return data


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
# (D) SAVE DF
# ---------------------


def save_df_to_parquet(df, f_name, chunked=False):
    # setup logger
    logger = session.logger

    output_folder = os.getenv("PATH_PROCESSED")
    # Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    file_path = Path(output_folder) / f"{f_name}"

    if chunked:
        if isinstance(df, pd.DataFrame):
            save_df_chunkwise(df, file_path, chunk_size=200)

        elif isinstance(df, list):
            save_df_list_chunkwise(df, file_path)

        else:
            df.to_parquet(f"{file_path}.parquet", index=False)

    else:
        df.to_parquet(f"{file_path}.parquet", index=False)

    logger.info(
        "Saved %s → %s (%s)",
        f_name,
        ph.shorten_path(file_path),
        "chunked" if chunked else "full",
    )


def enforce_datetime(df, col="datetime"):
    # setup logger
    logger = session.logger

    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="raise")
        logger.info("Changed '%s' dtype from '%s' to 'datetime64'.", col, df[col].dtype)

    return df


def save_df_chunkwise(df, file_path, chunk_size):
    # setup logger
    logger = session.logger

    writer = None

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(f"{file_path}.parquet", table.schema)

        writer.write_table(table)

        logger.info("Saved chunk 'df[%s: %s]' (total: %s)", i, i + chunk_size, len(df))

        del chunk
        del table
        gc.collect()

    if writer:
        writer.close()

    return


def save_df_list_chunkwise(df_list, file_path):
    # setup logger
    logger = session.logger

    writer = None

    for i, chunk in enumerate(df_list):
        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(f"{file_path}.parquet", table.schema)

        writer.write_table(table)

        logger.info("Saved chunk %s to %s", i + 1, ph.shorten_path(file_path))
        del chunk
        del table
        gc.collect()

    if writer:
        writer.close()

    return


#############################


def sort_extract_df(df, sort_cols, extract_cols, k):
    df_sort = df.sort_values(
        by=sort_cols,
        ascending=False,
    )

    topk_df = df_sort.loc[:, extract_cols].head(k).reset_index(drop=True)

    return df_sort, topk_df


def col_name_correct(df_input, split_value):
    df = df_input.copy()

    col_to_rename = {}
    # name = ""
    for col in df.columns:
        parts = col.split(f"{split_value}")  # here: " "
        if len(parts) >= 2:
            col_to_rename[col] = parts[0]
            # name = parts[1]
        else:
            print(f"[INFO] No column name in df was splitted at '{split_value}'.")

    return df.rename(columns=col_to_rename)


def df_quick_check(df):
    print("[CHECK] SHAPE:\n", df.shape)
    print("\n[CHECK] HEAD:\n", df.head())
    return


def parse_list_str(x):
    if not isinstance(x, str) or x.strip() == "":
        return []

    # Case 1: echte Python-Listenrepräsentation
    if x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    # Case 2: Fallback – kommasepariert
    if "," in x:
        return [v.strip() for v in x.split(",") if v.strip()]

    return []


def merge_dfs(
    df_list: List[pd.DataFrame],
    on_cols: List[str] | str,
    suffix_col: str | None = None,
    drop_cols: List[str] | str | None = None,
    how: str = "inner",
) -> pd.DataFrame:

    if isinstance(on_cols, str):
        on_cols = [on_cols]

    if not on_cols:
        raise ValueError("on_cols must be specified for merge")

    dfs_clean = []
    # cols = []
    for i, df in enumerate(df_list):
        df = df.rename(columns={c: f"{c}_{i}" for c in df.columns if c == suffix_col})

        if drop_cols:
            df_clean = df.drop(
                columns=drop_cols, errors="ignore"
            ).copy()  # .columns = [col for col in df.columns if col not in drop_cols]
        else:
            df_clean = df.copy()

        for col in on_cols:
            if col not in df_clean.columns:
                raise ValueError(f"Column '{col}' is not in df '{i}'.")

        # cols.extend(df.columns)
        dfs_clean.append(df_clean)

    df_merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=on_cols,
            how=how,
        ),
        dfs_clean,
    )

    return df_merged


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

    return obj


def save_dict(path: Path, data: dict) -> None:
    ph.ensure_dir(path)
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
        except TypeError:
            print("NON-SERIALIZABLE:", type(data_new), repr(data_new))


def append_json(path: Path, data: dict) -> None:
    ph.ensure_dir(path)
    data_new = make_json_safe(data)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data_new) + "\n")
        print(f"Appending data on {ph.shorten_path(path, 3)}")


def load_dict(path: Path) -> dict:
    ph.ensure_dir(path)

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
