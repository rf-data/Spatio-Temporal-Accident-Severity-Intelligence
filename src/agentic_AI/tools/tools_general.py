## tools_general.py
# imports
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

# import src.utils.postgre_helper as post
from src.core.tools_classes import ToolResult
from src.utils.agent_helper import validate_query
from src.core.tools_classes import tools_registry, add_tool


@add_tool(
    tools_registry,
    description="Check if the specified table exists in the database.",
    category="general",
)
def table_exists(engine: Engine, table: str, schema: str) -> ToolResult:
    """Check if the specified table exists in the database."""
    #
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)

    exists = table in tables

    return ToolResult(
        tool_name="table_exists",
        success=exists,
        message=(
            f"Table {schema}.{table} exists."
            if exists
            else f"Table {schema}.{table} does NOT exist."
        ),
        metadata={"schema": schema, "table": table},
    )


@add_tool(
    tools_registry,
    description="Get the number of rows in the specified table.",
    category="general",
)
def get_row_count(engine: Engine, table: str, schema: str) -> ToolResult:
    """Get the number of rows in the specified table."""
    query = text(f"SELECT COUNT(*) as n FROM {schema}.{table}")

    #
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            count = result[0]

        return ToolResult(
            tool_name="get_row_count",
            success=True,
            data={"row_count": int(count)},
            metadata={"schema": schema, "table": table},
        )

    except Exception as e:
        return ToolResult(
            tool_name="get_row_count",
            success=False,
            message=str(e),
            metadata={"schema": schema, "table": table},
        )


@add_tool(tools_registry, description="Run a SQL query", category="general")
def run_sql(engine: Engine, query: str) -> ToolResult:

    # validate if query is permitted
    is_valid, message = validate_query(query)

    if not is_valid:
        return ToolResult(
            tool_name="run_sql", success=False, message=f"Query rejected: {message}"
        )

    # execute query and return results
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

        return ToolResult(
            tool_name="run_sql",
            success=True,
            data={
                "columns": list(columns),
                "rows": [dict(zip(columns, row)) for row in rows],
            },
        )

    except Exception as e:
        return ToolResult(tool_name="run_sql", success=False, message=str(e))


@add_tool(
    tools_registry,
    description="Check if 'SUM(metric GROUP BY group_column) == stored_aggregate'.",
    category="general",
)
def aggregate_check(
    engine,
    table: str,
    schema: str,
    group_column: str,
    metric_column: str,
    aggregate_column: str,
    tolerance: float = 0.01,
) -> ToolResult:
    """
    🎯 Ziel
    Prüfen, ob:
    SUM(metric GROUP BY group_column)
    ==
    stored_aggregate

    🔧 Design
    Parameter:
    engine
    table
    schema
    group_column
    metric_column
    aggregate_column (optional)
    tolerance (float)

    🧠 Logik
    Rechne Aggregat
    Vergleiche mit gespeicherter Aggregat-Spalte
    Berechne absolute + relative Differenz
    Flagge Abweichungen > tolerance
    """
    query = text(f"""
        WITH recalculated AS (
            SELECT 
                {group_column},
                SUM({metric_column}) AS computed_value
            FROM {schema}.{table}
            GROUP BY {group_column}
        )
        SELECT 
            r.{group_column},
            r.computed_value,
            t.{aggregate_column} AS stored_value,
            (r.computed_value - t.{aggregate_column}) AS difference,
            CASE 
                WHEN t.{aggregate_column} = 0 THEN NULL
                ELSE (r.computed_value - t.{aggregate_column}) / t.{aggregate_column}
            END AS relative_diff
        FROM recalculated r
        JOIN {schema}.{table} t
        ON r.{group_column} = t.{group_column}
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            columns = result.keys()

        flagged = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            rel_diff = row_dict.get("relative_diff")

            if rel_diff is not None and abs(rel_diff) > tolerance:
                flagged.append(row_dict)

        return ToolResult(
            tool_name="aggregate_check",
            success=True,
            data={
                "checked_rows": len(rows),
                "violations": flagged,
                "violation_count": len(flagged),
            },
            metadata={
                "schema": schema,
                "table": table,
                "group_column": group_column,
                "metric_column": metric_column,
                "aggregate_column": aggregate_column,
                "tolerance": tolerance,
            },
        )

    except Exception as e:
        return ToolResult(tool_name="aggregate_check", success=False, message=str(e))


@add_tool(tools_registry, description="Get table schema.", category="general")
def get_table_schema(engine: Engine, table: str, schema: str) -> ToolResult:
    inspector = inspect(engine)

    try:
        columns = inspector.get_columns(table, schema=schema)
        schema_info = [
            {"name": col["name"], "type": str(col["type"]), "nullable": col["nullable"]}
            for col in columns
        ]

        return ToolResult(
            tool_name="get_table_schema",
            success=True,
            data=schema_info,
            metadata={"schema": schema, "table": table},
        )

    except Exception as e:
        return ToolResult(tool_name="get_table_schema", success=False, message=str(e))


@add_tool(
    tools_registry,
    description="Get ratio of nan-values in a given column.",
    category="general",
)
def get_null_ratio(engine: Engine, table: str, column: str, schema: str) -> ToolResult:
    query = text(f"""
        SELECT 
            COUNT(*) AS total,
            COUNT({column}) AS non_null
        FROM {schema}.{table}
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()

        total = result[0]
        non_null = result[1]
        null_ratio = 1 - (non_null / total) if total > 0 else None

        return ToolResult(
            tool_name="get_null_ratio",
            success=True,
            data={"total": total, "null_ratio": null_ratio},
            metadata={"schema": schema, "table": table, "column": column},
        )

    except Exception as e:
        return ToolResult(tool_name="get_null_ratio", success=False, message=str(e))
