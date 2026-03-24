# 🛣️ Spatio-Temporal-Accident-Severity-Intelligence
## Data
The data was published by the French Ministry of the Interior ([homepage](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)) and covers all accidents that have occurred from 2005 to 2024 on French territory. To simplify the analysis, only accidents registered on the French mainland and Corsica were included. All accidents which have taken place in DOM-TOM departments were excluded.   
The data is provided on an annual base and spread over several databases (`characteristics`, `places`, `persons`, `vehicles` and `registered vehicles`). During the first steps, only data from `characteristics` from all years available (2005 --> 2024) was used.

## Step 2: 🤖 Autonomous Data Analysis (Agent System, version: 0.2)
Starting with version 0.2, the project introduces an agent-based analysis layer that autonomously inspects datasets, validates assumptions, and generates structured analytical reports.

Instead of relying solely on manual exploratory analysis, the system uses specialized agents that execute predefined analytical checks and generate reproducible documentation of the results.

### Motivation
Large spatio-temporal datasets often require repetitive exploratory steps:   
- dataset validation   
- schema inspection   
- statistical sanity checks   
- feature diagnostics   
- anomaly detection

The agent system automates these steps and provides consistent, reproducible analytical reports.

### Agent Architecture
The system currently contains several specialized agents:
```
Agent	            Purpose
Raw Data Agent	    Performs automated EDA and structural inspection of raw datasets
SQL Upload Agent	Loads prepared datasets into PostgreSQL and validates ingestion
Audit Agent	        Runs structured analytical checks and generates audit-style reports
```

Each agent follows a tool-based architecture where analytical functions (`tools` + `checks`) can be dynamically registered and executed.

### Key Features
- modular check registry   
- structured reporting   
- reproducible audit pipeline   
- automatic detection of data issues   
- support for iterative analysis loops

### Design Principles
- separation of tools and reasoning   
- reproducibility   
- modular check system   
- extensible agent framework   

**The agent framework enables future extensions such as:**   
- automatic feature diagnostics   
- model validation agents   
- dataset monitoring   
- reinforcement-style analysis loops

👉 See the [dedicated documentation](src/agentic_AI/README.md):

## Step 1: **Road Accident Risk Mapping** (Version: 0.1)
### Overview
This project analyzes traffic accident data using H3 spatial indexing and time-based aggregation to identify structured risk patterns across different spatial resolutions.

The goal is to move beyond simple heatmaps and evaluate:
- Spatial resolution effects (H3 res 4–9)
- Temporal aggregation strategies (monthly / weekly)
- Zero-inflation behavior
- Distributional properties (entropy, Gini, dispersion)
- Temporal stability of risk signals

The project is built as a reproducible data engineering + analytical pipeline.

### Architecture
**Layer 1 – Raw Data**
- PostgreSQL ingestion
- Cleaned accident dataset
- H3 index generation per event

**Layer 2 – Aggregation**
- H3 × Time bin tables
- Zero-inflated full grids
- Statistical base metrics

**Layer 3 – Feature Store**
- Parquet export of aggregated tables
- Resolution-specific datasets

**Layer 4 – Analysis & Visualization**
- GeoPandas heatmaps
- Log-scaled and linear/non-scaled risk visualization (see plots in [folder](./data/plots/risk_heatmaps))
- Resolution comparison (see [interactive map](./data/plots/risk_heatmaps/h3_acci_heatmap_fol_res4_5_6_month.html))
- Quantitative evaluation metrics

**Key Methodological Components**
For each H3 resolution:
- Zero share
- Variance / mean ratio (dispersion)
- Shannon entropy
- Gini coefficient
- Temporal correlation (lag-1 stability)

These metrics support data-driven selection of optimal spatial and temporal resolution.

**Technologies**
- Python (Pandas, NumPy)
- PostgreSQL + SQLAlchemy
- H3 (Uber Hexagonal Indexing)
- GeoPandas + Folium
- PyArrow (Parquet)
- Matplotlib

**Current Status**
- H3 indexing implemented (res 4–9)
- Monthly zero-inflated grids created
- Statistical evaluation pipeline implemented
- Parquet export for feature storage

Further extensions will include:
- Weekly aggregation
- Model-based risk estimation
- Predictive modeling (Poisson / NB / ML)
- Interactive visualization layer

**Reproducibility**
- Configuration via .env
- Data export via chunked SQL → Parquet pipeline
- Derived tables can be regenerated from raw database.
