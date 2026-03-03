## findings.py
# imports
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum


# --------------------
# DATA AUDITOR
# --------------------
class SeverityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class AuditFinding(BaseModel):
    check_name: str
    issue_type: str
    severity: SeverityLevel
    message: str

    metric: Optional[str] = None
    value: Optional[float] = None
    expected: Optional[float] = None
    deviation: Optional[float] = None

    context: Optional[Dict[str, Any]] = None


class RiskLevel(str, Enum):
    ok = "ok"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


SEVERITY_POINTS = {"low": 5, "medium": 15, "high": 35, "critical": 70}


class AuditRiskSummary(BaseModel):
    total_score: int
    risk_level: RiskLevel
    finding_count: int
    scored_findings: List[Dict[str, Any]]  # each includes finding + score breakdown
    top_findings: List[Dict[str, Any]]


###################################################################################################################
# -----------------------------------------------------------------------------------------------------------------
###################################################################################################################
# --------------------
# RAW DATA AGENT
# --------------------
# class RawDataFinding(BaseModel):
#     check_name: str
#     issue_type: str

#     metric: Optional[str] = None
#     value: Optional[float] = None


#     expected: Optional[float] = None
#     deviation: Optional[float] = None

#     context: Optional[Dict[str, Any]] = None


class DiagnosticFinding(BaseModel):
    check_name: str
    description: str
    category: str
    # issue_type: str
    column: Optional[str]
    metrics: Optional[dict]
    severity: Optional[str]
    recommendation_hint: Optional[str | dict]


class ActionSchema(BaseModel):
    action: str
    target: List[str] | None = None
    params: dict | None = None

    model_config = {
        "extra": "forbid"
        }

