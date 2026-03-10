# ## tools.py
# # imports
# from sqlalchemy import text
# from sqlalchemy.engine import Engine

# from src.core.tools_classes import tools_registry, add_tool, ToolResult

# # from agent_audit.tools.tools_general import FORBIDDEN_KEYWORDS, ALLOWED_TABLES, ToolResult

# # import src.utils.postgre_helper as post
# # from src.core.session import session

# ##############################
# # CONFIGURATION
# ##############################
# KNOWN_FORMATS = [
#     "%Y-%m-%d",
#     "%d.%m.%Y",
#     "%Y/%m/%d",
#     "%d/%m/%Y",
#     "%Y-%m-%d %H:%M:%S"
# ]

# ##############################
# # GENERAL
# ##############################
# def datetime_already_split(df):

#     parts = ["year", "month","day","hour","minute", "second"
#              "an","mois","jour", "heure", "seconde"
#              "yr", "sec", "hrmn",
#              "Jahr", "Monat", "Tag", "Stunde", "Sekunde"]

#     found = [c for c in df.columns if c.lower() in parts]

#     return {
#         "detected_parts": found,
#         "already_split": len(found) >= 2
#     }


# @add_tool(tools_registry, 
#       description="Checks if column can be transformed into 'datetime'.",
#       category="raw_tab_file_EDA")
# def detect_datetime_candidates(df: pd.DataFrame, 
#                                config: dict | None = None):
#     candidates = {
#         "single_col": [],
#         "splitted_cols": []
#         }

#     split_dict = datetime_already_split(df)
    
#     if split_dict["already_split"]:
#         candidates["splitted_cols"] = split_dict["detected_parts"]

#     obj_cols = df.select_dtypes(include=["object", "string"]).columns

#     for col in obj_cols:
#         if "date" in col.lower():
#             candidates["single_col"].append(col)
#             continue

#         sample = df[col].dropna().astype(str).head(100)
        
#         if sample.empty:
#             continue
        
#         for fmt in KNOWN_FORMATS:
#             parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
#             success_ratio = parsed.notna().mean()

#             if success_ratio > 0.8:
#                 candidates["single"].append(col)      

#     hint = {"dt_candidates": candidates} if candidates else None
    
#     return Observation(
#                 tool_name="detect_datetime_candidates",
#                 category="raw_tab_file_EDA",
#                 column="str- + obj-columns",
#                 description="Detects datetime-like columns based on parsing success ratio.",
#                 metrics={
#                     "dt_candidates": candidates
#                     },
#                 recommendation_hint=hint
#                 )


# # @add_tool(
# #     tools_registry,
# #     description="Checks for gaps in time series data.",
# #     category="time_series",
# # )
# # def get_time_gaps(
# #     engine: Engine, table: str, schema: str, time_column: str, frequency: str
# # ) -> ToolResult:

# #     freq_map = {"day": "1 day", "week": "1 week", "month": "1 month"}

# #     if frequency not in freq_map:
# #         return ToolResult(
# #             tool_name="get_time_gaps",
# #             success=False,
# #             message=f"Unsupported frequency: {frequency}",
# #         )

# #     interval = freq_map[frequency]

# #     query = text(f"""
# #         WITH bounds AS (
# #             SELECT 
# #                 MIN({time_column}) AS min_date,
# #                 MAX({time_column}) AS max_date
# #             FROM {schema}.{table}
# #         ),
# #         series AS (
# #             SELECT generate_series(
# #                 (SELECT min_date FROM bounds),
# #                 (SELECT max_date FROM bounds),
# #                 INTERVAL '{interval}'
# #             ) AS expected_date
# #         )
# #         SELECT s.expected_date
# #         FROM series s
# #         LEFT JOIN {schema}.{table} t
# #         ON s.expected_date = t.{time_column}
# #         WHERE t.{time_column} IS NULL
# #         ORDER BY s.expected_date
# #     """)

# #     try:
# #         with engine.connect() as conn:
# #             result = conn.execute(query)
# #             gaps = [row[0] for row in result.fetchall()]

# #         return ToolResult(
# #             tool_name="get_time_gaps",
# #             success=True,
# #             data={"gap_count": len(gaps), "missing_dates": gaps},
# #             metadata={
# #                 "schema": schema,
# #                 "table": table,
# #                 "column": time_column,
# #                 "frequency": frequency,
# #             },
# #         )

# #     except Exception as e:
# #         return ToolResult(tool_name="get_time_gaps", success=False, message=str(e))


# # @add_tool(
# #     tools_registry,
# #     description="Computes z_score for given column.",
# #     category="time_series",
# # )
# # def compute_zscore(engine: Engine, table: str, column: str, schema: str) -> ToolResult:
# #     query = text(f"""
# #         SELECT 
# #             AVG({column}) AS mean,
# #             STDDEV({column}) AS std
# #         FROM {schema}.{table}
# #     """)

# #     try:
# #         with engine.connect() as conn:
# #             stats = conn.execute(query).fetchone()

# #         mean = stats[0]
# #         std = stats[1]

# #         if std == 0 or std is None:
# #             return ToolResult(
# #                 tool_name="compute_zscore",
# #                 success=False,
# #                 message="Standard deviation is zero or null.",
# #             )

# #         z_query = text(f"""
# #             SELECT 
# #                 ({column} - :mean) / :std AS z_score
# #             FROM {schema}.{table}
# #         """)

# #         with engine.connect() as conn:
# #             z_values = conn.execute(z_query, {"mean": mean, "std": std}).fetchall()

# #         return ToolResult(
# #             tool_name="compute_zscore",
# #             success=True,
# #             data={"mean": mean, "std": std, "z_scores": [z[0] for z in z_values]},
# #         )

# #     except Exception as e:
# #         return ToolResult(tool_name="compute_zscore", success=False, message=str(e))
