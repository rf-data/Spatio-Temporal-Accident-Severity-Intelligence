## prompts.py
# imports
from src.agentic_AI.agent.schemes import JSON_SCHEME


# --------------------
# DATA AUDITOR
# --------------------
# PLANNING
# --------------------
def build_planning_prompt(
    goal,
    available_checks,
    scope,
    #   add_context:List|None=None,
    #   add_checks:dict|None=None
):

    pre = f"""
You are a planning agent in a data {scope} system.

Goal:
{goal}

Available checks:
{list(available_checks.keys())}
"""
    msg = [pre]

    # if add_context:
    #     msg.append(f"""
    # Dataset Overview:\n
    # {add_context}
    # """)

    # if add_checks:
    #     msg.append(f"""
    # Available checks grouped by category:
    # {add_checks}
    # """)

    msg.append("""
Rules:
- Only choose from the available checks.
- Do not invent new checks.
- Return a JSON list of check names in execution order.

Example:
["time_integrity", "aggregate_consistency"]

Return JSON only.
""")

    return "\n".join(msg)


def build_reflection_prompt(
    goal, scope, current_findings, executed_checks, available_checks
):

    return f"""
You are a reflective {scope} agent.

Goal:
{goal}

Executed checks so far:
{executed_checks}

Current findings:
{current_findings}

Available checks:
{list(available_checks.keys())}

Rules:
- Only select checks that have NOT been executed yet.
- Only select from available checks.
- If no additional checks are necessary, return [].
- Return JSON list only.

Example:
["z_score_check"]

Return JSON only.
"""


# --------------------
# REASONING
# --------------------
def build_audit_report_comparison_prompt(risk_summary, memory_analysis):

    return f"""
You are a data audit interpretation assistant.

Current Risk:
Score: {risk_summary.total_score}
Level: {risk_summary.risk_level}

Memory Analysis:
Score Difference: {memory_analysis["score_diff"]}
Previous Level: {memory_analysis["risk_level_previous"]}
Current Level: {memory_analysis["risk_level_current"]}
New Issues: {memory_analysis["new_issues"]}
Resolved Issues: {memory_analysis["resolved_issues"]}

Top Findings:
{risk_summary.top_findings}

Explain:
1. Whether the system stability is improving or degrading.
2. Whether escalation is recommended.
3. What the new issues imply.
4. Whether resolved issues indicate recovery.

Return JSON only.
"""


def build_audit_reasoning_prompt(risk_summary, goal):

    return f"""
You are a data quality auditing assistant.

Your task is to analyze structured audit findings from a dataset. Goal of this 
analysis was: {goal}

IMPORTANT:
- Do NOT invent data.
- Only use the provided findings.
- Do NOT suggest SQL queries.
- Focus on interpretation and recommendations.

Overall risk score: {risk_summary.total_score}
Risk level: {risk_summary.risk_level}

Top findings:
{risk_summary.top_findings}

Provide:
1. Executive summary
2. Risk assessment explanation
3. Likely root causes
4. Concrete remediation steps
5. Confidence level (low/medium/high)

Return your answer strictly in JSON format matching:

{JSON_SCHEME}
"""
