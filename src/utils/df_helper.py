## df_helper.py
# imports
import pandas as pd
from typing import List

import gc
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path

import src.utils.general_helper as gh
import src.utils.path_helper as ph
from src.core.session import session


def inflate_df(df, config):
    
    res_col = config.get("res", "h3_index")
    # freq = config.get("freq", "W")
    freq_col = config.get("freq_col", "time_bin")
    
    cells = df[res_col].unique()

    # freq_clean = check_translate_freq(freq)
    # lower, upper = get_datetime_limits(df, config)
    # lower = df[freq_col].min()
    # upper = df[freq_col].max()

    time_index = df[freq_col].drop_duplicates().sort_values()
    # time_index = create_time_bins(lower, upper, freq=freq_clean)
    # time_index = df[time_col].unique()
    df_full = create_complete_grid(df, res_col, cells, freq_col, time_index) 

    return df_full 


def create_complete_grid(df, idx_name, idx_values, col_name, col_values):
    full_index = pd.MultiIndex.from_product(
        [idx_values, col_values],
        names=[idx_name, col_name]
    )
    
    df_full = (
        df
        .set_index([idx_name, col_name])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return df_full


def inflate_time_h3(df, time_col, h3_col, freq="W"):

    lower, upper = time.get_datetime_limits(df, time_col)

    time_index = time.create_time_bins(lower, upper, freq=freq)
    h3_index = df[h3_col].unique()

    full_index = pd.MultiIndex.from_product(
                                [h3_index, time_index],
                                names=[h3_col, time_col]
                                )
    df_agg = (
            df
            .groupby([h3_col, time_col])
            .size()
            .rename("n_accidents")
            )

    df_full = (
            df_agg
            .reindex(full_index, fill_value=0)
            .reset_index()
            )

    return df_full


def melt_h3(df, res_range, freq_range):

    res_cols = [f"h3_res{r}" for r in res_range]

    df_long = df.melt(
        id_vars=["ID_accident"] + freq_range,
        value_vars=res_cols,
        var_name="resolution",
        value_name="h3_index"
    )

    # clean 'resolution': "h3res6" → 6
    df_long["resolution"] = df_long["resolution"].str.replace("h3_res", "").astype(int)

    return df_long
   

def melt_time(df_long, freq_range):

    df_time = df_long.melt(
        id_vars=["ID_accident", "resolution", "h3_index"],
        value_vars=freq_range,
        var_name="freq",
        value_name="time_bin"
        )

    return df_time


def aggregate_all(df):

    df_agg = (
        df
        .groupby(["h3_index", "time_bin"])      # "resolution", "freq", 
        .size()
        .reset_index(name="n_accidents")
        )

    return df_agg


def aggregate_single(df, config):
                     
    res_col = config.get("res", "h3_index")
    time_col = config.get("freq_col", "time_bin")
    target_col = config.get("target_col", "n_accidents")
    
    df_agg = (
        df
        .groupby([res_col, time_col])
        .size()
        .reset_index(name=target_col)
        )
    
    # if events_only:
    #     df_agg = df_agg[df_agg[target_col] > 0]

    return df_agg


def load_processed_files(config):
    # lazy import
    from src.agentic_AI.report.raw_data_report import load_eda_summary

    data_processed = os.getenv("PATH_PROCESSED")

    # arg_dict = config.get("general_args", {})
    data_folder = Path(config.get("data_folder", None))
    file_suffix = config.get("file_suffix", ["clean"])
    if isinstance(file_suffix, str):
        file_suffix = [file_suffix]
        
    df_folder = Path(f"{data_processed}/{data_folder}")

    # load report and df_dict 
    report = load_eda_summary(config)
    files = report.get("files", [])

    dfs = {}
    for file in files:
        f_name = Path(file).stem
        
        df_list = []
        for f_suf in file_suffix: 
            df_name = f"{str(f_name).strip()}_{f_suf}"

            f_path = f"{df_folder}/{df_name}.parquet"
            print("Start loading:", ph.shorten_path(f_path))

            try:
                df = pd.read_parquet(f_path)
                df_list.append(df)

                print("[DEBUG] df shape:", df.shape)
                # print("[DEBUG] NaN count in df:\n", df.isna().sum())
            except FileNotFoundError:
                print("File not found:", f_path)
    
        dfs[f_name] = df_list

    return dfs


def load_merge_processed_files(config):

    cols_needed = config.get("necessary_cols", [])
    
    df_dict = load_processed_files(config)

    dfs_merged = {}
    for f_name, df_list in df_dict.items():
        # merge dfs
        if len(df_list) < 2:
            print(f"Invalid count of dfs ({f_name}):", len(df_list))
            continue
        
        elif len(df_list) == 2:
            print(f"Start merging df_list (n={len(df_list)})")
            df_merge = df_list[0].merge(df_list[1], on="ID_accident", how="inner")
            print("[DEBUG] unique in 'target_col' (df_1):", df_list[0]["ID_accident"].nunique())
            print("[DEBUG] unique in 'target_col' (df_2):", df_list[1]["ID_accident"].nunique())
            print("[DEBUG] unique in 'target_col' (df_merged):", df_merge["ID_accident"].nunique())

        else:
            print(f"Start merging df_list (n={len(df_list)})")
            for i, df in enumerate(df_list):
                if i == 0:
                    df_merge = df.copy()
                else: 
                    df_merge = df_merge.merge(df)

        df_red = df_merge[cols_needed].copy()
        dfs_merged[f_name] = df_red

    return dfs_merged


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


def read_french_csv_smart(path: str, nrows=None) -> pd.DataFrame:
    sep = detect_delimiter(path)
    df = pd.read_csv(
        path,
        sep=sep,
        encoding="latin1",
        decimal=",",
        engine="python",
        on_bad_lines="skip",
        nrows=nrows
        # low_memory=False
    )

    return df


def load_files_from_folder(folder, 
                           df_names: List | None=None,
                           suffix: str | None=None,
                           f_type: str | None=None):
    # setup logger
    # logger = session.logger
    print("Start loading files")
    
    if df_names:
        folder = [Path(f"{folder}/{name}") for name in df_names]
        files = list(folder)
    
    else:
        folder = Path(folder)
        files = list(folder.iterdir())

    df_dict = {}
    for file in files:
        
        f_name =  str(file.name)

        if f_type is None:
            f_type = str(file.suffix)
            
        stem_clean = file.stem.strip()
        if suffix:
            file_path = file.with_name(f"{stem_clean}_{suffix}.{f_type}")
        else:
            file_path = file.with_name(f"{stem_clean}.{f_type}")

        if "-" in f_name:
            f_new = f_name.replace("-", "_").replace(" ", "")
        else: 
            f_new = f_name

        if f_type == "csv":
            df = read_french_csv_smart(str(file_path))
            df_fixed = fix_single_column_df(df)
            
            df_dict[f_new] = df_fixed

        if f_type == "parquet":
            df_dict[f_new] = pd.read_parquet(file_path)

        print("Loaded file:\t", ph.shorten_path(file_path))  
        # df_re = df.rename(columns={"Accident_Id": "Num_Acc", "accident_id": "Num_Acc"})

        # logger.info("shape:\t%s", df_re.shape)
        # logger.info("columns:\t%s", df_re.columns)
        # logger.info("head:\n%s", df_re.head(3))
        # logger.info("")


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
    # data = data.rename(columns={"Accident_Id": "Num_Acc", "accident_id": "Num_Acc"})

    return data


# ---------------------
# (D) SAVE DF
# ---------------------
def save_df_to_parquet(df, f_name, folder=None, chunked=False):
    # setup logger
    logger = session.logger

    if folder is None:
        folder = os.getenv("PATH_PROCESSED")
    
    file_path = Path(folder) / f"{f_name}.parquet"

    ph.ensure_dir(file_path)
    # Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    
    if chunked:
        if isinstance(df, pd.DataFrame):
            save_df_chunkwise(df, file_path, chunk_size=200)

        elif isinstance(df, list):
            save_df_list_chunkwise(df, file_path)

        else:
            df.to_parquet(file_path, index=False)

    else:
        df.to_parquet(file_path, index=False)

    if logger:
        logger.info(
            "Saved %s → %s (%s)",
            f_name,
            ph.shorten_path(file_path),
            "chunked" if chunked else "full",
        )
    else:
        print(f"Saved {f_name} →  {ph.shorten_path(file_path)} ({'chunked' if chunked else 'full'})",)


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
            writer = pq.ParquetWriter(file_path, table.schema)

        writer.write_table(table)

        if logger:
            logger.info("Saved chunk 'df[%s: %s]' (total: %s)", i, i + chunk_size, len(df))
        else:
            print(f"Saved chunk 'df[{i}: {i + chunk_size}]' (total: {len(df)})")

        del chunk
        del table
        gc.collect()

    if writer:
        writer.close()

    # print("Saved file in:\t", )
    return


def save_df_list_chunkwise(df_list, file_path):
    # setup logger
    logger = session.logger

    writer = None

    for i, chunk in enumerate(df_list):
        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(file_path, table.schema)

        writer.write_table(table)

        if logger:
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
