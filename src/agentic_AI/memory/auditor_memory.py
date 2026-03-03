## layer_memory.py
# imports


# ----------------------
# RISK_DRIFT_DETECTION
# ----------------------
def compare_with_previous(current_report, previous_report):

    if not previous_report:
        return None

    previous_findings = {
        (f["check_name"], f["issue_type"]) for f in previous_report["findings"]
    }

    current_findings = {
        (f["check_name"], f["issue_type"]) for f in current_report.findings
    }

    new_issues = current_findings - previous_findings
    resolved_issues = previous_findings - current_findings

    score_diff = current_report.total_score - previous_report["total_score"]
    level_changed = current_report.risk_level != previous_report["risk_level"]

    compare_dict = {
        "score_diff": score_diff,
        "risk_level_previous": previous_report["risk_level"],
        "risk_level_current": current_report.risk_level,
        "new_issues": list(new_issues),
        "resolved_issues": list(resolved_issues),
    }

    if level_changed or (score_diff > 30):
        state.llm_analysis = run_llm_reasoning(state.risk_summary)

    # if score_diff > 50:
    #     full_check_suite()

    return compare_dict
