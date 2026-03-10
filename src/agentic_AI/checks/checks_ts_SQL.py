# ## checks_ts.py
# # imports
# from src.agentic_AI.findings.audit_findings import resolve_severity
# import Road_accidents.src.agentic_AI.tools.tools_ts_SQL as ts_tool
# import src.agentic_AI.tools.tools_general as gen_tool
# from src.core.finding_classes import AuditFinding, SeverityLevel
    
# # @add_check(checks_registry, 
# #       description="Checks for aggregated consistency.",
# #       category="general")
# def check_time_integrity(engine, 
#                         fn_arguments,
#                         config): 
                         
#     # table, 
#     # db_schema, 
#     # time_column, 
#     # frequency, 

#     if not gen_tool.table_exists(engine, db_schema, table):
#         return [AuditFinding(... issue_type="missing_table")]
    
#     findings = []

#     result = ts_tool.get_time_gaps(engine, 
#                                    table, 
#                                    schema, 
#                                    time_column, 
#                                    frequency)

#     if not result.success:
#         findings.append(
#             Finding(
#                 check_name="time_gaps",
#                 issue_type="tool_failure",
#                 severity=SeverityLevel.critical,
#                 message=result.message
#             )
#         )
#         return findings

#     gap_count = result.data["gap_count"]

#     if gap_count > 0:
#         severity = resolve_severity("time_gaps", gap_count, config)

#         findings.append(
#             Finding(
#                 check_name="time_gaps",
#                 issue_type="missing_time_periods",
#                 severity=severity,
#                 message=f"{gap_count} missing time periods detected.",
#                 metric="gap_count",
#                 value=gap_count,
#                 context=result.metadata
#             )
#         )

#     return findings


# # check_time_continuity()



# # ------------------------------
# # GATHER RESULTS FROM TS_CHECKS 
# # ------------------------------
# # def gather_ts_findings():

# #     engine = agent.context.engine
# #     checks = agent.context.checks

# #     ts_findings = []

# #     for check in checks:
        
# #         results = check()
# #         ts_findings.extend(results)


# #         check_time_integrity(
# #             engine,
# #             table="h3_res5_week",
# #             schema="accidents",
# #             time_column="week_start",
# #             frequency="week",
# #             config=config
# #         )
        
# #     return ts_findings
