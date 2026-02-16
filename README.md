# 🛣️ Spatio-Temporal-Accident-Severity-Intelligence
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

### Data
The data was published by the French Ministry of the Interior (https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)[homepage] and covers all accidents that have occurred from 2005 to 2024 on French territory. To simplify the analysis, only accidents registered on the French mainland and Corsica were included. All accidents which have taken place in DOM-TOM departments were excluded. 

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
- Log-scaled and linear/non-scaled risk visualization (/home/robfra/0_Portfolio_Projekte/Road_accidents/data/plots/risk_heatmaps)[folder]
- Resolution comparison (/home/robfra/0_Portfolio_Projekte/Road_accidents/data/plots/risk_heatmaps/h3_acci_heatmap_fol_res4_5_6_month.html)[interactive map]
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