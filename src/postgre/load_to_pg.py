# imports
import os
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import io

import src.utils.general_helper as gh
import src.utils.postgre_helper as post


def load_to_pg():
    # load env variables
    gh.load_env_vars()
    user = os.getenv("POSTGRE_USER")
    pw = os.getenv("POSTGRE_PASSWORD")
    db = os.getenv("DB_NAME")
    processed = os.getenv("PATH_PROCESSED")
    scheme = os.getenv("TABLE_SCHEME")

    engine = post.get_engine()

    cols = [
        "id",
        # calender 1
        "year",
        "month",
        "day",
        "hour",
        # calender 2
        "weekday",
        "is_weekend",
        "time_clean",
        # context 1
        "light conditions",
        "localisation",
        "intersection type",
        # context 2
        "weather",
        "collision type",
        # admin
        "department",
        "commune",
        # geo
        "lat_norm",
        "lon_norm",
    ]

    # load df
    df_path = f"{processed}/df_character_postgres.csv"
    df_charac = pd.read_csv(
        df_path,
        dtype={
            "commune": "string",
            "department": "string",
        },
        low_memory=False,
    )

    df_pg = df_charac[cols].copy()

    df_pg = df_pg.rename(columns={"time_clean": "time_of_day"})
    df_pg["time_of_day"] = pd.to_datetime(
        df_pg["time_of_day"], format="%H:%M", errors="coerce"
    ).dt.time

    int_cols = [
        "year",
        "month",
        "day",
        "weekday",
        "hour",
        "light conditions",
        "localisation",
        "intersection type",
        "weather",
        "collision type",
    ]
    for c in int_cols:
        df_pg[c] = df_pg[c].astype("Int64")  # pandas nullable int

    # df_pg.columns = [
    #     "id", "datetime", "year",
    #     "month", "day", "hour",
    #     "weekday", "is_weekend", "light_conditions",
    #     "localisation", "intersection_type", "weather",
    #     "collision_type", "department", "commune",
    #     "lat_norm", "lon_norm"
    # ]

    # def iter_chunks(df, chunk_size=25):
    #     for start in range(0, len(df), chunk_size):
    #         yield df.iloc[start : start + chunk_size]

    # for df in gh.iter_chunks(df_pg, chunk_size=5_000):

    print("df length:\t", len(df_pg))
    buffer = io.StringIO()
    df_pg.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
    buffer.seek(0)

    #
    conn = engine.raw_connection()
    cur = conn.cursor()

    cur.copy_expert(
        """
        COPY accidents.characteristics (
            id, 
            year, month, day, hour, 
            weekday, is_weekend, time_of_day,
            light_conditions, localisation, intersection_type, 
            weather, collision_type, 
            department, commune, 
            lat_norm, lon_norm
        )
        FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')
        """,
        buffer,
    )

    conn.commit()
    cur.close()
    conn.close()

    # df.to_sql(
    #         table,
    #         engine,
    #         schema=scheme,
    #         if_exists="append",
    #         index=False,
    #         method="multi",
    #         # chunksize=50_000
    #     )


if __name__ == "__main__":
    load_to_pg()
