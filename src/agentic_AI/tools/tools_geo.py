## tools.py
# imports
from sqlalchemy import text

# from src.audit.tools.tools_general import FORBIDDEN_KEYWORDS, ALLOWED_TABLES, ToolResult
from src.core.tools_classes import tools_registry, add_tool

# import src.utils.postgre_helper as post
# from src.core.session import session


# -----------------------
# GEO TOOL FUNCTIONS
# -----------------------
@add_tool(
    tools_registry,
    description="Compares metrics for two given H3_resolutions.",
    category="geo_data",
)
def cross_resolution_check(engine, table_low, table_high, schema):
    """SKIZZE !!!!
    Idee:
    Gruppiere res_low
    Aggregiere
    Vergleiche mit res_high

    Hier musst du definieren:
    wie parent_index berechnet wird
    welche H3-Level verglichen werden

    Das wird dein Signature-Tool.
    """
    query = text(f"""
        SELECT 
            l.parent_index,
            SUM(l.n_accidents) AS sum_children,
            h.n_accidents AS parent_value,
            SUM(l.n_accidents) - h.n_accidents AS difference
        FROM {schema}.{table_low} l
        JOIN {schema}.{table_high} h
        ON l.parent_index = h.h3_index
        GROUP BY l.parent_index, h.n_accidents
    """)
