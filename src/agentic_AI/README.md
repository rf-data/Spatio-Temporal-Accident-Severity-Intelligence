# README – Multi-Agent Analysis System
## Overview
The **Agent Analysis System** provides an autonomous analytical layer for the road accident risk project.

Instead of manually running exploratory analysis scripts, specialized agents perform structured inspections of datasets and generate reproducible analytical reports.

The system follows a **tool-based agent architecture**, where agents execute predefined analytical checks and document their findings.

## Agent Types
### Raw Data Agent

**Purpose:**   
Initial inspection of datasets before any transformation or modeling.

**_Typical checks_** include:   
- dataset structure   
- variable types   
- missing values   
- duplicates   
- distributional properties   
- outlier detection   
- basic descriptive statistics

**Output:**   
- structured markdown and JSON reports   
- dataset diagnostics   
- merge analysis

### SQL Upload Agent
**Purpose:**
- Ensures reliable ingestion of processed datasets into PostgreSQL.   

**Responsibilities:**   
- validate dataset schema   
- upload data to target tables   
- confirm row counts   
- check for schema inconsistencies

### Audit Agent (_pending improvements_)
**Purpose:**   
- Performs systematic validation checks on derived datasets.

**Examples:**    
- statistical plausibility checks   
- aggregation validation   
- consistency between tables   
- detection of unexpected data patterns

The agent generates audit-style analytical reports that summarize all checks.

## Architecture
The system consists of four core components:

Agent
│
├── State
│
├── Tools
│
├── Check Registry
│
└── Reporting Layer

### Agent
The agent coordinates the analysis process.

**Responsibilities:***   
- execute analytical checks   
- update internal state   
- generate reports

### Tools and ToolsRegistry

Tools are pure analytical functions that operate on datasets. They are registred in a ToolsRegistry which also contains additional information on each tool, e.g.:  
- description   
- category (e.g. "tabular" or "time-series")    
- used in exploratory data analysis   
- used as default/first-line check   
- applied as cross-file-check

Tools are reusable across different agents.

### Checks and ChecksRegistry   
Each agent maintains a registry of available analytical checks (similar to ToolsRegistry). Checks are used to gather information from various tools and if necessary to forward this information to prepare a data processing or merge strategy.   

This design allows dynamic addition of new checks.

### State   
The agent maintains a state object containing:   
- dataset references   
- intermediate results   
- report sections   
- executed checks

### Reporting Layer   
The system generates structured markdown reports summarizing the analysis.

**Example report sections:**   
- dataset overview   
- schema inspection   
- distribution analysis   
- anomalies detected   
- summary of checks

### Execution   
Agents are typically executed via **CLI**:   
`python -m src.agent_AI.run_raw_data_agent --name $(config_name) `

or via **Makefile**:   
`make raw_data_agent_run NAME=$(config_name) `

Configuration parameters can be provided via config yaml-files or CLI options.

### Design Philosophy

The agent system was designed with the following principles:

**Reproducibility**   

All analysis steps should be:   
- deterministic   
- documented   
- re-usuable

**Modularity**   
New checks can easily be added without modifying the agent core.

**Transparency**   

The system generates human-readable and machine-readable reports (markdown and json) for every analysis run, respectively .

**Extensibility**   

The architecture allows future extensions such as:   
- feature and model performance critique agents   
- feature diagnostics/optimisation agents   
- dataset monitoring   
- reinforcement-style analysis loops

**Future Extensions**   

Planned extensions include:   
- automated feature diagnostics   
- model evaluation agents   
- time-series anomaly detection   
- reinforcement learning style analysis loops   
- integration with experiment tracking