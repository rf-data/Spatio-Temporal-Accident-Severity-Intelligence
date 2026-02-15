# imports 
from sqlalchemy import create_engine, text
import os

import src.utils.general_helper as gh

def get_engine():
    gh.load_env_vars()
    user = os.getenv("POSTGRE_USER")
    pw = os.getenv("POSTGRE_PASSWORD")
    db = os.getenv("DB_NAME")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{pw}@localhost:5432/{db}"
    )

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
        for idx in [calender_index, geo_year_index, 
                    geo_index, department_index]:
            result = conn.execute(text(idx))
            print(f"Output ({idx})", result.fetchall())

    
    return 

