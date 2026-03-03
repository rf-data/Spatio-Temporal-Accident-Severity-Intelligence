# imports
import os
from datetime import datetime
from sqlalchemy import text
import pandas as pd

from src.core.session import session
from src.core.logger import create_logger
import src.utils.postgre_helper as post
import src.utils.general_helper as gh
import src.utils.path_helper as ph
from configuration.H3_evaluate_res import config


def retrieve_base_stat(table, conn):
    # setup logger
    logger = session.logger

    # query
    base_stat = f"""
        SELECT
            COUNT(*) AS rows,
            MIN(n_accidents) AS MIN,
            AVG(n_accidents) AS MEAN,
            VAR_SAMP(n_accidents) AS VAR, 
            STDDEV_SAMP(n_accidents) AS STD, 
            VAR_SAMP(n_accidents) / STDDEV_SAMP(n_accidents) AS RATIO_VAR_MEAN,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY n_accidents) AS Q25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_accidents) AS MEDIAN, 
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY n_accidents) AS Q75,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY n_accidents) AS Q90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY n_accidents) AS Q95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY n_accidents) AS Q99,
            MAX(n_accidents) AS MAX,
            SUM(CASE WHEN n_accidents = 0 THEN 1 ELSE 0 END) * 1.0 AS ZERO_COUNT, 
            SUM(CASE WHEN n_accidents = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS ZERO_SHARE
        FROM {table};
        """

    logger.info("Start compiling base statistics for table '%s'", table)

    result = conn.execute(text(base_stat))
    df = pd.DataFrame(result.fetchall())

    return df


def calculate_entropy(table, conn):
    # setup logger
    logger = session.logger

    query_entropy = f"""
        WITH frequence AS (
            SELECT
                n_accidents,
                COUNT(*) AS f
            FROM {table}
            GROUP BY n_accidents
        ),
        total AS (
            SELECT SUM(f) AS N FROM frequence
        )
        SELECT
            -SUM( (f::float / N) * LN(f::float / N) ) AS entropy
        FROM frequence, total;
        """

    logger.info("Start calculating entropy from table '%s'", table)

    result = conn.execute(text(query_entropy))
    entropy = result.scalar()

    return entropy


def calculate_gini(table, conn):
    # setup logger
    logger = session.logger

    # query_message
    query_gini = f"""
        WITH ordered AS (
            SELECT
                n_accidents,
                ROW_NUMBER() OVER (ORDER BY n_accidents) AS i
            FROM {table}
        ),
        agg AS (
            SELECT
                COUNT(*)::float AS n, 
                SUM(n_accidents)::float AS total_sum, 
                SUM(i * n_accidents)::float AS weighted_sum
            FROM ordered
        )
        SELECT
            CASE
                WHEN total_sum = 0 THEN 0
                ELSE (2 * weighted_sum / (n * total_sum)) - ((n + 1) / n) 
            END AS gini
        FROM agg;
        """

    logger.info("Start calculating Gini coefficient from table '%s'", table)

    result = conn.execute(text(query_gini))
    gini_coef = result.scalar()

    return gini_coef


def estimate_time_stability(table, freq, conn):
    # setup logger
    logger = session.logger

    # query_message
    query_time_stab = f"""
        WITH lagged AS (
            SELECT
                h3_index,
                {freq}_start,
                n_accidents,
                LAG(n_accidents) OVER (
                    PARTITION BY h3_index
                    ORDER BY {freq}_start
                ) AS prev_count
            FROM {table}
        )
        SELECT
            CORR(n_accidents::float, prev_count::float) AS temporal_stability
        FROM lagged
        WHERE prev_count IS NOT NULL;
        """

    logger.info("Start estimating temporal stability from table '%s'", table)

    result = conn.execute(text(query_time_stab))
    time_stability = result.scalar()

    return time_stability


def describe_save_h3_df(df, f_name, idx_new=None):
    # setup logger
    logger = session.logger

    #
    if idx_new is not None:
        df.set_index(idx_new, inplace=True)

    # for col in ["mean", "var",
    #             "std", "ratio_var_mean",
    #              "zero_share",
    #             "entropy", "gini",
    #             "time_stability"]:
    #     df[col] = df[col].astype(float).round(3)

    for col in ["zero_count", "rows", "min", "median", "max"]:
        df[col] = df[col].astype(int)

    pd.set_option("display.float_format", "{:.3f}".format)

    logger.info("h3_stat_merge -- INFO ---\n%s\n", df.info())
    logger.info("h3_stat_merge -- OVERVIEW ---\n%s\n", df.T)

    folder = os.getenv("PATH_PROCESSED")
    df_path = f"{folder}/{f_name}.csv"
    # df.to_csv(df_path)
    logger.info("Saved df '%s' to %s", f_name, ph.shorten_path(df_path))

    return


# retrieve basic statistics from SQL tables
def evaluate_h3_resolutions():
    # load env variables
    gh.load_env_vars()

    session.load_config(config)
    log_name = session.log_name
    name_logfile = session.log_file

    # # load logger
    logger = create_logger(name=log_name, file_name=name_logfile)

    session.logger = logger

    # setup DB_engine
    engine = post.get_engine()

    #
    h3_values = session.h3_values
    frequence = [session.freq]
    inflate = session.inflate

    h3_dict = {}
    with engine.begin() as conn:
        for val in h3_values:
            for freq in frequence:
                table = f"accidents.h3_res{val}_{freq}{'_zeroinf' if inflate else ''}"

                df = retrieve_base_stat(table, conn)

                df["entropy"] = calculate_entropy(table, conn)
                df["gini"] = calculate_gini(table, conn)
                df["time_stability"] = estimate_time_stability(table, freq, conn)

                df["res"] = val
                df["freq"] = freq
                h3_dict[f"res_{val}"] = df

    h3_stat_merge = pd.concat(h3_dict.values(), ignore_index=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    describe_save_h3_df(h3_stat_merge, f_name="df_h3_eval", idx_new=["res", "freq"])

    logger.info("Completed H3 resolution evaluation at %s", now)

    return


if __name__ == "__main__":
    evaluate_h3_resolutions()
