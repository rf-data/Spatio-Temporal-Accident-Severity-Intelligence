# imports 
from sqlalchemy import text

# ---------------------
# QUERIES
# ---------------------

ROW_COUNT = text("""
    SELECT COUNT(*) FROM accidents.characteristics
""")

GEO_RATIO = text("""
    SELECT
      COUNT(*) FILTER (WHERE lat_norm IS NOT NULL AND lon_norm IS NOT NULL)::float
      / COUNT(*)
    FROM accidents.characteristics
""")

HOUR_RANGE = text("""
    SELECT MIN(hour), MAX(hour)
    FROM accidents.characteristics
""")

MONTH_RANGE = text("""
    SELECT MIN(month), MAX(month)
    FROM accidents.characteristics
""")

YEAR_RANGE = text("""
    SELECT MIN(year), MAX(year)
    FROM accidents.characteristics
""")

DAILY_AGGREGATION = text("""
    SELECT
        day,
        COUNT(*) AS n_accidents
    FROM accidents.characteristics
    GROUP BY 1
    ORDER BY 1
    LIMIT 10;                      
""")

TIME_SPACE_AGGREGATION = text("""
    SELECT
        department,
        month,
        COUNT(*) AS n
    FROM accidents.characteristics
    WHERE lat_norm IS NOT NULL
    GROUP BY 1, 2;                                                     
""")

RUSH_HOUR = text(""" 
    SELECT *,
        CASE 
            WHEN hour BETWEEN 7 AND 9 THEN 1 
            WHEN hour BETWEEN 16 AND 18 THEN 1 
            ELSE 0 
        END AS is_rush_hour
    FROM accidents.characteristics;               
""")

H3_PREP_VIEW = text(""" 
    CREATE VIEW accidents.base_for_h3 AS
    SELECT
        id,
        day,
        hour,
        weekday,
        is_weekend,
        weather,
        light_conditions,
        lat_norm,
        lon_norm
    FROM accidents.characteristics
    WHERE lat_norm IS NOT NULL;
""")
# ---------------------
# FUNCTIONS
# ---------------------

def make_rush_hour(conn):
    conn.execute(RUSH_HOUR)
    return 

def view_pre_h3(conn):
    conn.execute(H3_PREP_VIEW)

def check_row_count(conn, min_rows=500_000):
    n = conn.execute(ROW_COUNT).scalar()
    assert n >= min_rows, f"Too few rows: {n}"

def check_geo_ratio(conn, min_ratio=0.45):
    ratio = conn.execute(GEO_RATIO).scalar()
    assert ratio >= min_ratio, f"Geo ratio too low: {ratio:.2f}"

def check_calender_range(conn):
    start, end = conn.execute(YEAR_RANGE).one()
    assert start >= 2001, print("start - end:\t", start, end)
    assert end <= 2035, print("start - end:\t", start, end)

    start_h, end_h = conn.execute(HOUR_RANGE).one()
    assert start_h >= 0, print("start - end:\t", start_h, end_h)
    assert end_h <= 24, print("start - end:\t", start_h, end_h)

    start_m, end_m = conn.execute(MONTH_RANGE).one()
    assert start_m > 0, print("start - end:\t", start_m, end_m)
    assert end_m <= 12, print("start - end:\t", start_m, end_m)
