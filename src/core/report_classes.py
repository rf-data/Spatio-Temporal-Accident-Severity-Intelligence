## report_classes.py
# imports
from collections import defaultdict
from pydantic import BaseModel
from typing import Optional, List
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
    metric_results: dict
    processing: dict
    merge: dict
    schema_proposal: dict | None = None
    feature_engineering: dict | None = None
    metadata: dict | None = None


class MergeStrategy(BaseModel):
    strategy: str
    reason: str
    join_key: str | None
    confidence: dict[str, float] | None
    similarity: float | None
    

class MergeStatement(BaseModel):
    files: tuple = None
    valid_keys: str | List = None
    overlap: dict | None = None
    column_check: dict | None = None
    key_uniqueness: dict | None = None
    n_rows: dict | int | None = None
    merge_strategy: MergeStrategy | None = None
    
