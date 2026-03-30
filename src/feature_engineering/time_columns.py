
# import
import numpy as np
import pandas as pd

from src.core.session import session


def get_datetime_limits(df, config): 
    
    dt_col = config["dt_col"]       # , "datetime")

    # print(f"Describe column '{col}':\n", df[col].describe())
    
    dt = df[dt_col].dropna()

    min_year = dt.dt.year.min()
    max_year = dt.dt.year.max()

    lower = pd.Timestamp(f"{min_year}-01-01 00:00:00")
    upper = pd.Timestamp(f"{max_year}-12-31 23:59:59")

    return lower, upper


def create_time_bins(lower, upper, freq="W"):

    freq_safe = check_translate_freq(freq)

    if freq_safe == "W":
        freq_safe = "W-MON"
    elif freq_safe == "2W":
        freq_safe = "2W-MON"
    elif freq == "M":
        freq_safe = "MS"

    return  pd.date_range(start=lower, end=upper, freq=freq_safe)


def apply_time_binning(df, config):
    
    freq = config.get("freq", "month")
    freq_col = config.get("freq_col", "time_bin")
    time_col_new = config.get("time_col_new", "datetime")

    # df_list = []
    # for freq, df_sub in df.groupby(freq_col):
    #     df_sub["time_bin"] = (
    #         df_sub[dt_col]
    #         .dt.to_period(freq)
    #         .dt.to_timestamp()
    #     )
    #     df_list.append(df_sub)
    # return pd.concat(df_list)

    freq_clean = check_translate_freq(freq)
    df = df.copy()
    df[freq_col] = df[time_col_new].dt.to_period(freq_clean).dt.to_timestamp()

    return df


def create_n_weekly(df, time_col, n_weeks=2):

    df = df.copy()

    # Woche auf Montag normalisieren
    week_start = df[time_col] - pd.to_timedelta(df[time_col].dt.weekday, unit="D")

    # Referenz-Montag (fix!)
    ref = pd.Timestamp("2000-01-03")

    # Abstand in Wochen
    delta_weeks = ((week_start - ref).dt.days // 7)

    # n-week buckets
    n_week_bin = (delta_weeks // n_weeks) * n_weeks

    n_dict = {
        "2": "two",
        "3": "three",
        "4": "four", 
        "5": "five"
    }

    n_as_word = n_dict[str(n_weeks)]
    df[f"{n_as_word}_weekly"] = ref + pd.to_timedelta(n_week_bin * 7, unit="D")

    return df


def cyclic_encode_col(df_in, encode_cols):
    # setup logger
    logger = session.logger

    # feats = session.exp_params.get("features", None)
    df = df_in.copy()

    # cols_to_encode = encode_cols.get("cyclic_encode", [])

    encode_dict = {
            "month": 12, 
            "week": 52,
            "day": 30,
            "weekday": 7,
            "hour": 24,
        }
    
    invalid_period = []
    for period in encode_cols: 
        # if period in df.columns:
        period_int = encode_dict.get(period, None)

        if (period_int is None) or (period not in df.columns):
            invalid_period.append(period)
            continue
            # raise ValueError(f"")

        df[f"{period}_sin"] = np.sin(2 * np.pi * df[period] / period_int)
        df[f"{period}_cos"] = np.cos(2 * np.pi * df[period] / period_int)

        logger.info(
                "Added encoded time columns (['%s', '%s']) to df.",
                f"{period}_sin",
                f"{period}_cos",
                )
        
    if len(invalid_period) > 0:
        logger.info("Following 'periods' could not be encoded:\t%s", 
                    invalid_period)
    # df_new = df.drop(columns=col_to_drop)

    return df


def create_timestamp_col(df_in, config): 
    # dt_dict: dict, extract_cols: List | None = None):

    df = df_in.copy()
    
    # extract year from ID
    df["year"] = df["ID_accident"].astype(str).str[:4].astype(int)

    time_processing = config.get("time_processing")
    time_col = time_processing.get("time_col", {})
    # time_col_new = time_processing.get("time_col_new", "timestamp")
    timestamp_col = time_processing.get("timestamp_col")
    split_time_col = time_processing.get("split_time_col", None)

    if timestamp_col is not None: 
        return pd.to_datetime(df[timestamp_col], errors="coerce")

    all_col = []
    if split_time_col:
        dt_format = split_time_col.get("dt_format")
        extract_cols = split_time_col.get("to_extract", [])
        add_cols = split_time_col.get("to_add", [])
        all_col = [extract_cols + add_cols]

        df["time_clean"] = df[time_col].apply(parse_time_hhmm)
        dt_col = "time_parsed"
        df[dt_col] = pd.to_datetime(
                    df["time_clean"],
                    format=dt_format,
                    errors="coerce",
                    )

        print("[DEBUG] df head dt_cols:", df[[dt_col, "time_clean", time_col]].head(3))
        df = extract_col_from_datetime(df, dt_col, extract_cols)

    print("df head:\n", df.head(3).T)
    # dt_col = 

    dt_dict = {
        "year": df["year"].astype(float), 
        "month": df["month"].astype(float), 
        "day": df["day"].astype(float),
        "hour": df["hour"].astype(float), 
        "minute": df["minute"].astype(float),
        }
     
    return pd.to_datetime({
        "year": dt_dict.get("year", 1),
        "month": dt_dict.get("month", 1),
        "day": dt_dict.get("day", 1),
        "hour": dt_dict.get("hour", 1),
        "minute": dt_dict.get("minute", 1)
        }, errors="coerce")
    # for col in all_col:
    #     if col in list(df.columns):
    #         parts[col] = df[col]
        
    #     else: 
    #         parts[col] = 1

    # , errors="coerce")
    # }

    # mapping = {
    #     "year": "year",
    #     "month": "month",
    #     "day": "day",
    #     "hour": "hour",
    #     "minute": "minute",
    #     "second": "second",
    # }

    # for key, pandas_key in mapping.items():
    #     col = config.get(key)
    #     if col and col in df.columns:
    #         parts[pandas_key] = pd.to_numeric(df[col], errors="coerce")

    # if parts:
    #     return pd.to_datetime(parts, errors="coerce")
    
    # raise ValueError("No valid timestamp configuration provided")


def extract_col_from_datetime(df, dt_col, cols):
    # get logger
    logger = session.logger

    ts = df[dt_col]

    feature_map = {
        "year": ts.dt.year,
        "month": ts.dt.month,
        "week": ts.dt.isocalendar().week,
        "day": ts.dt.day,
        "weekday": ts.dt.weekday,
        "hour": ts.dt.hour,
        "minute": ts.dt.minute,
        "second": ts.dt.second,
    }

    cols_filt = [c for c in cols if c in feature_map.keys()]
    # print(f"[DEBUG] cols - cols_filt:\n{cols}\n{cols_filt}")

    for col in cols_filt:
            df[col] = feature_map[col]

    print(f"[DEBUG - SPLIT DT_COL] df 'cols_filt' head:\n", df[cols_filt].head(3))

    return df



def check_translate_freq(freq):

    freq_dict = {
            "monthly_start": "MS",
            "monthly_end": "ME",
            "monthly": "M",
            "daily": "D",
            "weekly": "W",
            "two_weekly": "2W",
            "hourly": "H"
            }
    
    freq_allowed = list(freq_dict.values())

    if freq not in freq_allowed:
        freq = freq_dict.get(freq, None)
    
    if freq is None:
        raise KeyError("Argument for 'freq' is not allowed:\t", freq)
    
    return freq


def parse_time_hhmm(s_in):

    if pd.isna(s_in):
        return pd.NA
    s = str(s_in).strip()


    # format HH:MM
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            return pd.NA
        
        h, m = parts  # s = "".join(s_out)

        if not (h.isdigit() and m.isdigit()):
            return pd.NA
        
        s = f"{h.zfill(2)}{m.zfill(2)}"

    if not s.isdigit():
        return pd.NA

    if len(s) == 4:
        hh = int(s[:2])
        mm = int(s[2:])

    elif len(s) == 3:
        hh = int(s[0])
        mm = int(s[1:])
        return f"0{s[0]}:{s[1:]}"

    else:
        # false_time.append((s_in, s))
        return pd.NA

    # hard validity checks
    if not (0 <= hh <= 23) or not (0 <= mm <= 59):
        return pd.NA
 
    return f"{hh:02d}:{mm:02d}"



def add_time_cols(df, cols_new: dict):

    time_col_new = cols_new.get("time_col_new", "timestamp")
    add_cols = cols_new.get("add_cols", {})
    # cyclic_encode = cols_new.get("cyclic_encode", [])

    df_ext = extract_col_from_datetime(df, time_col_new, add_cols)
    
    if ("is_weekend" in add_cols) and ("weekday" in df.columns):  
        #  \        and ("is_weekend" not in df.columns):
        df_ext["is_weekend"] = df_ext["weekday"] >= 5

    if "is_holiday" in add_cols:
        # to be added sson
        hi = 'tba'

    # if cyclic_encode:
    #     df_ext = cyclic_encode_col(df_ext, cyclic_encode)

    return df



# def time_col_split(df, time_processing: dict):
#     split_time = time_processing.get("split_time_col", {})
    
#     extract_cols = split_time.get("to_extract", [])

    
#     dt_parts = dt_format.replace(":", "_").replace("-", "_").split("_")

    

#         for dt_p in dt_parts: 
#             if "h" in dt_p.lower():
#                 df["hour"] = df[dt_col].dt.hour

#             if "m" in dt_p.lower():
#                 df["minute"] = df[dt_col].dt.hour

#     if extract_cols is not None:
#         for col in extract_cols:

# "year": df[dt_col].dt.year,
#             "month": df[dt_col].dt.month,
#             "week": df[dt_col].dt.week,
#             "day": df[dt_col].dt.day,
#             "weekday": df[dt_col].dt.weekday,
#             "hour": df[dt_col].dt.hour,
#             "minute": df[dt_col].dt.minute,
#             "second": df[dt_col].dt.second
#     new_time_col = dt_dict.get("new_time_col", "timestamp")

    # year = time_col_order.get("year", None)
    # month = time_col_order.get("month", None)
    # day = time_col_order.get("day_of_month", None)
    # hour = time_col_order.get("hour", None)
    # minute = time_col_order.get("minute", None)
    # second = time_col_order.get("second", None)
    # order = time_col_order.get("order", None)
    
    # # extract order of time columns
    # parts = order.split("_")
    # if len(parts) != 2:
    #     print(f"""
    #         Provided time column order is not valid.
    #         Expected: 'Year_Month_Day' block is separated by '_' from 'hour_minute' block. 
    #                 e.g. '%Y-%m-%d_%H:%M:%S'
    #         Provided: {order}
    #           """)
    # else:


    # print(f"'Order' (length: {order_list}):", order_list)

    # year_month_day = [year, month, day]
    # hour_min_sec = [hour, minute]

    # df["timestamp"] = pd.to_datetime(
    #     df[year_month_day].astype(str).agg("-".join, axis=1)
    #     + " "
    #     + df[hour_min_sec],
    #     errors="coerce",
    # )

    # return df

