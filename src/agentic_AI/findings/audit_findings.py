## audit_findings.py
# imports
from typing import Dict, Any, List

# from src.agent_audit.checks.checks_ts import gather_ts_findings
# from src.agent_audit.checks.checks_general import gather_gen_findings
from src.core.finding_classes import (
    SeverityLevel,
    SEVERITY_POINTS,
    RiskLevel,
    AuditRiskSummary,
)


def resolve_severity(check_name: str, value: float, config: dict) -> SeverityLevel:
    rule = config.get(check_name)

    if not rule:
        return SeverityLevel.low

    threshold = rule.get("threshold")

    if threshold is not None and value is not None:
        if abs(value) >= threshold:
            return SeverityLevel(rule["severity"])
        else:
            return SeverityLevel.low

    return SeverityLevel(rule["severity"])


# --------------------
# EVALUATE FINDINGS
# --------------------
def _value_weight(finding) -> float:
    """
    Lightweight weighting based on finding.value, if numeric.
    Keeps behavior predictable. You can swap this later.
    """
    v = finding.value
    if v is None:
        return 1.0
    try:
        v = float(v)
    except Exception:
        return 1.0

    # Gentle scaling:
    # 1 -> 1.0, 5 -> 1.2, 20 -> 1.5, 100 -> 2.0 (approx)
    if v <= 1:
        return 1.0
    if v <= 5:
        return 1.2
    if v <= 20:
        return 1.5
    if v <= 100:
        return 2.0
    return 2.5


def score_single_finding(finding, config: dict) -> Dict[str, Any]:
    base = SEVERITY_POINTS.get(finding.severity.value, 0)

    check_mult = config["multipliers"].get("check", {}).get(finding.check_name, 1.0)
    issue_mult = config["multipliers"].get("issue", {}).get(finding.issue_type, 1.0)
    val_mult = _value_weight(finding)

    raw = base * check_mult * issue_mult * val_mult
    cap = config.get("max_points_per_finding", 100)
    points = int(min(raw, cap))

    return {
        "points": points,
        "base": base,
        "check_multiplier": check_mult,
        "issue_multiplier": issue_mult,
        "value_multiplier": val_mult,
    }


def risk_level_from_score(total_score: int, thresholds: dict) -> RiskLevel:
    # thresholds: {"ok":0,"low":20,"medium":60,"high":120,"critical":200}
    # We choose the highest level whose threshold is <= score.
    ordered = sorted(thresholds.items(), key=lambda x: x[1])
    level = "ok"
    for name, thr in ordered:
        if total_score >= thr:
            level = name
    return RiskLevel(level)


def score_audit_findings(findings: List["AuditFinding"], config) -> AuditRiskSummary:
    scored = []
    total = 0

    for f in findings:
        breakdown = score_single_finding(f, config)
        entry = {"finding": f.model_dump(), "score": breakdown}
        scored.append(entry)
        total += breakdown["points"]

    thresholds = config.get(
        "thresholds", {"ok": 0, "low": 20, "medium": 60, "high": 120, "critical": 200}
    )
    level = risk_level_from_score(total, thresholds)

    # sort by points desc to get top issues
    scored_sorted = sorted(scored, key=lambda x: x["score"]["points"], reverse=True)
    top = scored_sorted[:5]

    return AuditRiskSummary(
        total_score=int(total),
        risk_level=level,
        finding_count=len(findings),
        scored_findings=scored_sorted,
        top_findings=top,
    )


# --------------------
# GATHER FINDINGS
# --------------------
# def gather_findings():
#     func_list = [gather_ts_findings, gather_gen_findings]

#     all_findings = []
#     for func in func_list:
#         all_findings.extend(func())

#     return all_findings
