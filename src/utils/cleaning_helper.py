# imports
import pandas as pd

from src.core.session import session


def correct_charact_cols(df_in):
    df = df_in.copy()

    for col in ["commune", "department"]:
        df[col] = (
            df[col].astype(str).str.strip().replace({"nan": pd.NA}).astype("category")
        )

        df["longitude"] = (
            df["longitude"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA})
        )

    for col in ["longitude", "latitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def rename_cols(df_in):
    encode = session.encode

    df = df_in.rename(columns=encode).copy()

    return df

    # df.to_csv()


###########


def parse_time_hhmm(s_in):
    false_time = []

    if pd.isna(s_in):
        return pd.NA
    s = str(s_in).strip()

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
        false_time.append((s_in, s))
        return pd.NA

    # hard validity checks
    if not (0 <= hh <= 23):
        return pd.NA
    if not (0 <= mm <= 59):
        return pd.NA
    return f"{hh:02d}:{mm:02d}"
