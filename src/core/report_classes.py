## report_classes.py
# imports
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# --------------------
# DATA AUDITOR
# --------------------
class AuditReport(BaseModel):
    generated_at: datetime
    total_score: int
    risk_level: str
    finding_count: int
    findings: list
    top_findings: list
    llm_analysis: Optional[dict]



# --------------------
# RAW DATA AGENT
# --------------------
class PreparationSummary(BaseModel):
    processing: dict
    merge: dict
    sql_schema: dict | None = None
    feature_engineering: dict | None = None
    metadata: dict | None = None