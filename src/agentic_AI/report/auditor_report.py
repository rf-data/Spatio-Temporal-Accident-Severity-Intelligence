## report.py
# imports
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import json

import src.utils.general_helper as gh
from src.core.report_classes import AuditReport


def save_report(report: AuditReport, name, base_path=None):

    # timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")

    if base_path is None:
        # (1) load config
        # gh.load_env_vars(name=".env.audit_agent")
        base_path = os.getenv("FOLDER_REPORT")

    json_path = f"{base_path}/audit_{name}.json"
    md_path = f"{base_path}/audit_{name}.md"

    # JSON
    with open(json_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    # Markdown
    md_content = build_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md_content)

    return {"json": json_path, "markdown": md_path}


def build_json_report(risk_summary, findings, llm_analysis=None):

    return AuditReport(
        generated_at=datetime.now(),
        total_score=risk_summary.total_score,
        risk_level=risk_summary.risk_level.value,
        finding_count=risk_summary.finding_count,
        findings=[f.model_dump() for f in findings],
        top_findings=risk_summary.top_findings,
        llm_analysis=llm_analysis.model_dump() if llm_analysis else None,
    )


def build_markdown_report(report: AuditReport) -> str:

    md = []
    md.append(f"# Data Audit Report")
    md.append(f"Generated at: {report.generated_at}")
    md.append("")
    md.append(f"## Overall Risk")
    md.append(f"- Risk Level: **{report.risk_level.upper()}**")
    md.append(f"- Total Score: {report.total_score}")
    md.append(f"- Number of Findings: {report.finding_count}")
    md.append("")

    # LLM Section
    if report.llm_analysis:
        md.append("## Executive Summary")
        md.append(report.llm_analysis["executive_summary"])
        md.append("")

        md.append("## Risk Assessment")
        md.append(report.llm_analysis["risk_assessment"])
        md.append("")

        md.append("## Root Cause Hypotheses")
        for cause in report.llm_analysis["root_cause_hypotheses"]:
            md.append(f"- {cause}")
        md.append("")

        md.append("## Recommendations")
        for rec in report.llm_analysis["recommendations"]:
            md.append(f"### {rec['title']}")
            md.append(rec["explanation"])
            md.append(f"**Action:** {rec['recommended_action']}")
            md.append("")

    # Top Findings
    md.append("## Top Findings")
    for entry in report.top_findings:
        f = entry["finding"]
        points = entry["score"]["points"]
        md.append(f"- [{f['severity'].upper()}] {f['issue_type']} (Score: {points})")
        md.append(f"  - {f['message']}")
    md.append("")

    # All Findings
    md.append("## All Findings")
    for f in report.findings:
        md.append(f"- [{f['severity'].upper()}] {f['check_name']} - {f['issue_type']}")
        md.append(f"  - {f['message']}")
    md.append("")

    return "\n".join(md)
