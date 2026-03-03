## postgre_helper.py
# imports
from sqlalchemy import create_engine, text, inspect
import os
from typing import List
import pandas as pd
import io

import src.utils.general_helper as gh
from src.core.session import session


def check_table(engine, schema, table_name):
    # setup logger
    logger = session.logger

    #
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)

    if table_name not in tables:
        logger.error(f"❌ Table {schema}.{table_name} does NOT exist.")
        return

    df = pd.read_sql(f"SELECT COUNT(*) as n FROM {schema}.{table_name}", engine)
    logger.info(f"✅ Table {schema}.{table_name} exists.")
    logger.info(f"Rows: {df.loc[0, 'n']}")

    return


def copy_data_to_pg(df, cols, conn, table=None):
    # setup logger
    logger = session.logger
    h3_values = session.h3_values

    logger.info(
        "Copying data to DB_table '%s' \nshape = %s \ncolumns = %s",
        table,
        df.shape,
        df.columns,
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
    buffer.seek(0)

    #
    cur = conn.connection.cursor()

    if isinstance(cols, str):
        col_str = cols

    elif isinstance(cols, List):
        col_str = ", ".join(cols)

    else:
        logger.error("'cols' is neither str nor list.\t--> %s", type(cols))
        raise TypeError

    cur.copy_expert(
        f"""
        COPY {table} (
            {col_str}
        )
        FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')
        """,
        buffer,
    )

    # conn.commit()
    # cur.close()
    # conn.close()

    return conn


def generate_tmp_tbl(name, cols, p_key=None):
    # setup logger
    logger = session.logger

    # # # setup DB_engine
    # engine = post.get_engine()

    # check dtype
    if isinstance(cols, list):
        col_string = ", ".join(cols)  # [:-1]
    elif isinstance(cols, str):
        col_string = cols
    else:
        logger.error("'cols' is neither str nor list.\t--> %s", type(cols))
        raise TypeError

    if p_key is None:
        p_key = session.p_key

    tmp_tbl = f"""
    CREATE TEMP TABLE {name} (
      {p_key} BIGINT,
      {col_string}
      );
    """

    return tmp_tbl


def generate_update_query(src_table, dst_table, cols, p_key=None):
    # setup logger
    logger = session.logger

    #
    cols_pre = [f"{col} = t.{col}," for col in cols]
    cols_new = " ".join(cols_pre)[:-1]

    if p_key is None:
        p_key = session.p_key

    update = f"""
    UPDATE {dst_table} c
    SET
        {cols_new}
    FROM {src_table} t
    WHERE c.{p_key} = t.{p_key};
  """

    return update


def get_engine(env_name=".env"):
    gh.load_env_vars(env_name)
    user = os.getenv("POSTGRE_USER")
    pw = os.getenv("POSTGRE_PASSWORD")
    db = os.getenv("DB_NAME")

    engine = create_engine(f"postgresql+psycopg2://{user}:{pw}@localhost:5432/{db}")

    return engine


def update_year():
    update = text("""
        UPDATE accidents.characteristics
        SET year = year + 2000
        WHERE year < 100;
    """)

    secure = text("""
        ALTER TABLE accidents.characteristics
        ADD CONSTRAINT chk_year_valid
        CHECK (year BETWEEN 2000 AND 2035);
    """)

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text(update))
        print(f"Output (update):", result.fetchall())

        alter = conn.execute(text(secure))
        print(f"Output (alter):", alter.fetchall())

    return


def add_fill_column():
    add = """
        ALTER TABLE accidents.characteristics
        ADD COLUMN datetime TIMESTAMP;
    """
    fill = """
        UPDATE accidents.characteristics
        SET datetime =
        make_date(year, month, day)
        + time_of_day;
    """
    return


def rename_pkey_col():
    # change name of column
    query = """
        ALTER TABLE accidents.characteristics
        RENAME COLUMN id TO accident_id;
        """
    check = """
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'accidents.characteristics'::regclass
    AND contype = 'p';
    """

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text(query))
        print("Output (update col_name):", result.fetchall())

        check_resp = conn.execute(text(check))
        print("\nOutput (pkey check):", check_resp.fetch_all())


def set_indices():
    calender_index = """
    CREATE INDEX IF NOT EXISTS idx_acc_calendar
        ON accidents.characteristics (year, month, day);
    """
    geo_year_index = """
    CREATE INDEX IF NOT EXISTS idx_acc_geo_year
        ON accidents.characteristics (year, lat_norm, lon_norm);
    """
    geo_index = """
    CREATE INDEX IF NOT EXISTS idx_acc_geo
        ON accidents.characteristics(lat_norm, lon_norm);
    """
    department_index = """
    CREATE INDEX IF NOT EXISTS idx_acc_department
        ON accidents.characteristics (department);
    """

    engine = get_engine()
    with engine.begin() as conn:
        for idx in [calender_index, geo_year_index, geo_index, department_index]:
            result = conn.execute(text(idx))
            print(f"Output ({idx})", result.fetchall())

    return
