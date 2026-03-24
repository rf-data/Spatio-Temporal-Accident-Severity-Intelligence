## agent_class.py
# imports
from pydantic import BaseModel, model_validator, Field
from typing import List, Optional, Dict
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

import src.utils.file_helper as fh
from src.agentic_AI.findings.audit_findings import score_audit_findings
# from src.agentic_AI.findings.raw_data_findings import score_eda_findings   #  gather_findings
from src.agentic_AI.planning.agent_planning import plan_checks, reflect_and_plan
from src.agentic_AI.planning.plan_default_checks import plan_default_checks
from src.agentic_AI.reasoning.auditor_reasoning import run_llm_reasoning
from src.agentic_AI.report.raw_data_report import generate_preparation_summary

from src.core.finding_classes import (
                                AuditFinding,
                                AuditRiskSummary,
                                DiagnosticFinding,
                            )
from src.core.tools_classes import ToolRegistry
from src.core.checks_classes import CheckRegistry
from src.core.report_classes import PreparationSummary


# class RawDataArguments(BaseModel):
# --------------------
# AUDIT_AGENT
# --------------------
# INPUT
# --------------------
class AuditArguments(BaseModel):
    db_schema: Optional[str] = None
    frequency: Optional[str] = None
    group_column: Optional[str] = None
    metric_column: Optional[str] = None
    aggregate_column: Optional[str] = None
    table_prefix: Optional[str] = None

    table: Optional[str] = None
    time_column: Optional[str] = None

    @model_validator(mode="after")
    def build_derived_fields(self):
        if self.table_prefix and self.frequency:
            self.table = f"{self.table_prefix}_{self.frequency}"
        if self.frequency:
            self.time_column = f"{self.frequency}_start"
        return self


# --------------------
# OUTPUT
# --------------------
class AuditLLMRecommendation(BaseModel):
    title: str
    explanation: str
    recommended_action: str


class AuditLLMAnalysis(BaseModel):
    executive_summary: str
    risk_assessment: str
    root_cause_hypotheses: List[str]
    recommendations: List[AuditLLMRecommendation]
    confidence: str


# --------------------
# AGENT
# --------------------
class AuditState(BaseModel):
    goal: str = None
    llm_model: str = None
    llm_temperature: float = None
    executed_checks: List[str] = Field(default_factory=list)
    findings: List[AuditFinding] = Field(default_factory=list)
    risk_summary: Optional[AuditRiskSummary] = None
    llm_analysis: Optional[AuditLLMAnalysis] = None
    comparison: Optional[AuditLLMAnalysis] = None


class AuditContext:
    def __init__(
        self, engine, tool_registry: ToolRegistry, check_registry: CheckRegistry
    ):
        self.engine = engine
        self.tools = tool_registry
        self.checks = check_registry


class AuditAgent:
    def __init__(self, context: AuditContext, config: AuditArguments):

        self.context = context
        self.audit_config = config.get("audit", None)
        self.risk_config = config.get("risk", None)
        self.llm_config = config.get("llm", None)
        self.agent_config = config.get("agent", None)

        arguments_dict = self.audit_config.get("arguments", None)
        self.arguments = AuditArguments(**arguments_dict)

        self.state = AuditState(
            llm_model=self.llm_config.get("model", None),
            llm_temperature=self.llm_config.get("temperature", 0.0),
        )

    def execute_check(self, check_name: str):
        
        check_fn = self.context.checks.get(check_name)
        # check_cfg = build_check_config(self.audit_config["tables"]["weekly_table"])

        findings = check_fn(self.context.engine, self.arguments, self.audit_config)

        self.state.findings.extend(findings)
        self.state.executed_checks.append(check_name)

        return

    def run(self, goal: str):
        self.state.goal = goal

        # initial checks
        planned_checks = plan_checks(
            self.state.goal,
            self.context.checks._checks,
            self.state.llm_model,
            self.state.llm_temperature,
            agent="audit",
        )
        # [check for check in self.context.checks
        #                   if check.category == "general"]

        # Execution
        iteration = 0
        max_iterations = self.agent_config.get("max_interations", 1)
        reflexive = self.agent_config.get("reflexive", False)

        while iteration < max_iterations and planned_checks:
            for check_name in planned_checks:
                if check_name in self.state.executed_checks:
                    continue

                self.execute_check(check_name)

            if reflexive:
                check_dict = self.context.checks._checks

                planned_checks = reflect_and_plan(
                    self.state.goal, check_dict, self.state, agent="audit"
                )

                iteration += 1

        self.state.risk_summary = score_audit_findings(
            self.state.findings, self.risk_config
        )

        # LLM interpretation if necessary
        if self.state.risk_summary.risk_level.value in ["medium", "high", "critical"]:
            self.state.llm_analysis = run_llm_reasoning(
                self.state.risk_summary, self.state
            )

        return self.state


###################################################################################################################
# -----------------------------------------------------------------------------------------------------------------
###################################################################################################################
# --------------------
# RAW_DATA_AGENT
# --------------------
# INPUT
# --------------------
class RawDataArguments(BaseModel):
    folder: str | Path = None
    data_folder: str | Path = None
    report_name: str = None
    summary_name: str = None
    schema_name: str = None
    table_name: str = None
    file_type: Optional[List[str]] = ["csv", "parquet"]
    sample_size: Optional[int] = None
    chunk_size: Optional[int] = None
    sampling_strategy: Optional[str] = None
    drop_duplicates: Optional[dict[str, str]] = None
    impute_nan: Optional[dict[str, str]] = None
    # file_types Filter
    recursive: bool = False
    # encoding
    # infer_datetime flag


# --------------------
# OUTPUT
# --------------------
# class RawDataLLMRecommendation(BaseModel):
#     title: str
#     explanation: str
#     recommended_action: str


# class RawDataLLMAnalysis(BaseModel):
#     executive_summary: str
#     risk_assessment: str
#     root_cause_hypotheses: List[str]
#     recommendations: List[RawDataLLMRecommendation]
#     confidence: str


# --------------------
# AGENT
# --------------------
class RawDataState(BaseModel):
    goal: str = None
    llm_model: str = None
    llm_temperature: float = None
    executed_checks: List[str] = Field(default_factory=list)
    findings: List[DiagnosticFinding] = Field(default_factory=list)
    cross_file_analysis: List[DiagnosticFinding] = Field(default_factory=list)
    preparation_summary: Optional[PreparationSummary] = None
    # llm_analysis: Optional[RawDataLLMAnalysis] = None
    # comparison: Optional[RawDataLLMAnalysis] = None

    model_config = {
            "arbitrary_types_allowed": True
        }


class RawDataContext:
    def __init__(self, 
                 tool_registry: ToolRegistry, 
                 check_registry: CheckRegistry):
        self.tools = tool_registry
        self.checks = check_registry


class RawDataAgent:
    def __init__(self, context: RawDataContext, config):

        self.config = config
        self.agent_config = config.get("agent", None)
        self.arguments = config.get("general_args", None)
        self.context = context
        self.dataset_config = config.get("dataset", None)
        self.llm_config = config.get("llm", None)
        self.raw_data: Dict[str, pd.DataFrame] | None = None
        self.state = RawDataState(
            llm_model=self.llm_config.get("model", 
                                          None),
            llm_temperature=self.llm_config.get("temperature", 
                                                0.0),
            )
        self.tools_config = config.get("tool_args", None)

        # self.state = RawDataState(...)
        # self.eda_config = config.get("eda", None)
        # self.risk_config = config.get("risk", None)

        # arguments_dict = self.eda_config.get("arguments", None)
        self.arguments["folder"] = os.getenv("PATH_RAW")
        self.arguments = RawDataArguments(**self.arguments)


    def load_data(self):
        
        p_folder = Path(self.arguments.folder)
        data_folder = Path(self.arguments.data_folder)

        f_path = Path(f"{p_folder}/{data_folder}")

        csv_files = [f for f in f_path.iterdir() if f.suffix == ".csv"]
        parquet_files = [f for f in f_path.iterdir() if f.suffix == ".parquet"]

        dfs = {}
        for f in csv_files:

            f_name =  str(f.name)
            if "-" in f_name:
                f_new = f_name.replace("-", "_")
            else: 
                f_new = f_name

            dfs[f_new] = fh.read_french_csv_smart(f_new)

        for f in parquet_files:
            f_name =  str(f.name)
            dfs[f_name] = pd.read_parquet(f)

        self.raw_data = dfs

    def execute_check(self, check_name: str):
        if callable(check_name):
            check_fn = check_name
        else:
            check_fn = self.context.checks.get(check_name)

        print("[DEBUG] fn_name:", check_fn)
        findings = check_fn(self.raw_data, 
                            config=self.config)

        if isinstance(findings, tuple):
            print("Found tuple:", findings)
            self.state.findings.extend(findings)

        else:
            self.state.findings.append(findings)

        self.state.executed_checks.append(check_name)

        return findings

    def run(self, goal: str):
        self.state.goal = goal

        # initial checks
        checks = self.context.checks
        planned_checks = plan_default_checks(checks, 
                                             self.dataset_config)
        
        print("Following checks will be conducted:\n", planned_checks)
        # plan_checks(
        #     self.state.goal,
        #     self.context.checks._checks,
        #     self.state.llm_model,
        #     self.state.llm_temperature,
        #     agent="raw_data",
        # )


        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print("Starting checks")
        for check_name in planned_checks:
            if check_name in self.state.executed_checks:
                continue

            self.execute_check(check_name)

        # 🔹 AGGREGATION LAYER (NOT rule inference if rules already ran during checks)
        files_analysed = list(self.raw_data.keys())
        metadata = {
            "summary_name": self.arguments.summary_name or None,
            "schema_name": self.arguments.schema_name or None,
            "table_name": self.arguments.table_name or None, 
            "files_analyzed": files_analysed,
            "analysis_timestamp": now,
            "agent_version": self.agent_config.get('version', 'tba'),
        }
        
        print("Start generating prep_summary")
        self.state.preparation_summary = generate_preparation_summary(
                                                self.raw_data, 
                                                self.state.findings,
                                                metadata
                                                )
        # self.state.preparation_summary.sql_schema = generate_sql_schema()

        

        # self.state.raw_data = dfs
        # self.state.preparation_summary["sql_schema"] = {}
        # self.state.preparation_summary["scripts_to_generate"] = []

        # self.state.preparation_summary = {
        #                 "data_processing": data_processing_plan,
        #                 "merge_strategy": merge_plan,
        #                 "sql_schema": {},
        #                 "scripts_to_generate": []
        #                 }
        # self.state.preparation_summary = score_raw_data_findings(
        #     self.state.findings,
        #                                              self.risk_config)

        return self.state
