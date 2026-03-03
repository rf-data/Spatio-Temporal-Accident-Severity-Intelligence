## tools.py
# imports
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.core.tools_classes import tools_registry, add_tool, ToolResult

# from agent_audit.tools.tools_general import FORBIDDEN_KEYWORDS, ALLOWED_TABLES, ToolResult

# import src.utils.postgre_helper as post
# from src.core.session import session


@add_tool(
    tools_registry,
    description="Checks for gaps in time series data.",
    category="time_series",
)
def get_time_gaps(
    engine: Engine, table: str, schema: str, time_column: str, frequency: str
) -> ToolResult:

    freq_map = {"day": "1 day", "week": "1 week", "month": "1 month"}

    if frequency not in freq_map:
        return ToolResult(
            tool_name="get_time_gaps",
            success=False,
            message=f"Unsupported frequency: {frequency}",
        )

    interval = freq_map[frequency]

    query = text(f"""
        WITH bounds AS (
            SELECT 
                MIN({time_column}) AS min_date,
                MAX({time_column}) AS max_date
            FROM {schema}.{table}
        ),
        series AS (
            SELECT generate_series(
                (SELECT min_date FROM bounds),
                (SELECT max_date FROM bounds),
                INTERVAL '{interval}'
            ) AS expected_date
        )
        SELECT s.expected_date
        FROM series s
        LEFT JOIN {schema}.{table} t
        ON s.expected_date = t.{time_column}
        WHERE t.{time_column} IS NULL
        ORDER BY s.expected_date
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            gaps = [row[0] for row in result.fetchall()]

        return ToolResult(
            tool_name="get_time_gaps",
            success=True,
            data={"gap_count": len(gaps), "missing_dates": gaps},
            metadata={
                "schema": schema,
                "table": table,
                "column": time_column,
                "frequency": frequency,
            },
        )

    except Exception as e:
        return ToolResult(tool_name="get_time_gaps", success=False, message=str(e))


@add_tool(
    tools_registry,
    description="Computes z_score for given column.",
    category="time_series",
)
def compute_zscore(engine: Engine, table: str, column: str, schema: str) -> ToolResult:
    query = text(f"""
        SELECT 
            AVG({column}) AS mean,
            STDDEV({column}) AS std
        FROM {schema}.{table}
    """)

    try:
        with engine.connect() as conn:
            stats = conn.execute(query).fetchone()

        mean = stats[0]
        std = stats[1]

        if std == 0 or std is None:
            return ToolResult(
                tool_name="compute_zscore",
                success=False,
                message="Standard deviation is zero or null.",
            )

        z_query = text(f"""
            SELECT 
                ({column} - :mean) / :std AS z_score
            FROM {schema}.{table}
        """)

        with engine.connect() as conn:
            z_values = conn.execute(z_query, {"mean": mean, "std": std}).fetchall()

        return ToolResult(
            tool_name="compute_zscore",
            success=True,
            data={"mean": mean, "std": std, "z_scores": [z[0] for z in z_values]},
        )

    except Exception as e:
        return ToolResult(tool_name="compute_zscore", success=False, message=str(e))
