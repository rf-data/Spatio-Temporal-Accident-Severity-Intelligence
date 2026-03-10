# ## checks_general.py
# # imports
# from src.agentic_AI.findings.audit_findings import resolve_severity
# import src.agentic_AI.tools.tools_general as gen_tool

# from src.core.checks_classes import checks_registry, add_check
# from src.core.finding_classes import AuditFinding, SeverityLevel


# @add_check(
#     checks_registry,
#     description="Checks for aggregated consistency.",
#     category="general",
# )
# def check_aggregate_consistency(engine, fn_arguments, config):

#     table = fn_arguments.table  # , None)
#     db_schema = fn_arguments.db_schema  # "]  #, None)
#     group_column = fn_arguments.group_column  # , None)
#     metric_column = fn_arguments.metric_column  # , None)
#     aggregate_column = fn_arguments.aggregate_column  # , None)
#     tolerance = (
#         config.get("tools", None).get("cross_resolution").get("tolerance")
#     )  # , None)

#     if not gen_tool.table_exists(engine, db_schema, table):
#         return [
#             AuditFinding(
#                 check_name="check_aggregate_consistency",
#                 issue_type="missing_table",
#                 severity=SeverityLevel.critical,
#                 message=f"No table '{table}' in '{db_schema}'.",
#             )
#         ]

#     findings = []

#     result = gen_tool.aggregate_check(
#         engine,
#         table,
#         db_schema,
#         group_column,
#         metric_column,
#         aggregate_column,
#         tolerance,
#     )

#     if not result.success:
#         findings.append(
#             AuditFinding(
#                 check_name="cross_resolution",
#                 issue_type="tool_failure",
#                 severity=SeverityLevel.critical,
#                 message=result.message,
#             )
#         )
#         return findings

#     violation_count = result.data["violation_count"]

#     if violation_count > 0:
#         severity = resolve_severity("cross_resolution", violation_count, config)

#         findings.append(
#             AuditFinding(
#                 check_name="cross_resolution",
#                 issue_type="aggregation_mismatch",
#                 severity=severity,
#                 message=f"{violation_count} aggregation mismatches detected.",
#                 metric="violation_count",
#                 value=violation_count,
#                 context=result.metadata,
#             )
#         )

#     return findings


# # check_statistical_outliers()
# # check_table_integrity()


# # -----------------------------------
# # GATHER RESULTS FROM GENERAL_CHECKS
# # -----------------------------------
# # def gather_gen_findings(engine, config):

# #     gen_findings = []

# #     gen_findings.extend(
# #         check_aggregate_consistency(
# #             engine,
# #             table="h3_res5_week",
# #             schema="accidents",
# #             group_column="parent_index",
# #             metric_column="n_accidents",
# #             aggregate_column="parent_value",
# #             config=config
# #         )
# #     )

# #     return gen_findings
