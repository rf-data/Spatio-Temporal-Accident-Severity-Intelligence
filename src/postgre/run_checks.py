import src.postgre.postgre_helper as post
from src.postgre.queries import (
    check_row_count,
    check_geo_ratio,
    check_calender_range,
)

def main():
    engine = post.get_engine()
    with engine.connect() as conn:
        check_row_count(conn)
        check_geo_ratio(conn)
        check_calender_range(conn)

    print("All DB checks passed ✔")

if __name__ == "__main__":
    main()
