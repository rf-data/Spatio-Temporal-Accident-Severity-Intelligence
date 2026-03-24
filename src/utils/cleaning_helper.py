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

