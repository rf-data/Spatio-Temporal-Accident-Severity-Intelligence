## raw_data_findings.py
# imports
from typing import Dict, Any, List

from src.core.finding_classes import PreparationSummary


def score_raw_data_findings(
    findings: List["RawDataFinding"], config
) -> PreparationSummary:
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
