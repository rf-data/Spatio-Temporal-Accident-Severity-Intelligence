## tools_general.py
# imports
import re

from pydantic import BaseModel
from typing import Any, Optional, Dict
import pandas as pd

# -------------------------
# SQL_DATA AUDITOR
# -------------------------
FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
]

ALLOWED_TABLES = ["audit_views.weekly_summary", "audit_views.resolution_consistency"]


###################################################################################################################
# -----------------------------------------------------------------------------------------------------------------
###################################################################################################################
# -------------------------
# RAW_DATA AUDITOR
# -------------------------
recommendations = {
    "skewness": 'recommendation: "log-transform candidate" if abs(skew) > 2',
    "kurtosis": 'recommendation: "heavy-tailed distribution – outlier robust scaling" if kurtosis > 10',
}
