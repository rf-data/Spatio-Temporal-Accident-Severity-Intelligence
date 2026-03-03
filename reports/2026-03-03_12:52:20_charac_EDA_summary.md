# Raw Data EDA Report

files analysed:
caracteristiques_2005.csv, caracteristiques_2006.csv, caracteristiques_2007.csv, caracteristiques_2008.csv, caracteristiques_2009.csv, caracteristiques_2010.csv, caracteristiques_2011.csv, caracteristiques_2012.csv, caracteristiques_2013.csv, caracteristiques_2014.csv, caracteristiques_2015.csv, caracteristiques_2016.csv, caracteristiques-2017.csv, caracteristiques-2018.csv, caracteristiques-2019.csv, caracteristiques-2020.csv, caracteristiques-2021.csv, caracteristiques-2022.csv, caract-2023.csv, caract-2024.csv

analysis_timestamp:	2026-03-03_12:52:20

agent_version:	tba
## Data Processing Plan
### File: caracteristiques_2005.csv
### File: caracteristiques_2006.csv
### File: caracteristiques_2007.csv
### File: caracteristiques_2008.csv
### File: caracteristiques_2009.csv
### File: caracteristiques_2010.csv
### File: caracteristiques_2011.csv
### File: caracteristiques_2012.csv
### File: caracteristiques_2013.csv
### File: caracteristiques_2014.csv
### File: caracteristiques_2015.csv
### File: caracteristiques_2016.csv
### File: caracteristiques-2017.csv
### File: caracteristiques-2018.csv
### File: caracteristiques-2019.csv
### File: caracteristiques-2020.csv
### File: caracteristiques-2021.csv
### File: caracteristiques-2022.csv
### File: caract-2023.csv
### File: caract-2024.csv
## Merge Strategy
### caract-2023.csv_vs_caract-2024.csv
- Join Type: inner
- Join Keys: Num_Acc, mois, hrmn, atm, int, adr, dep, col, an, jour, lat, com, agg, lum, long

## Feature Engineering
- create_datetime_columns: 
- zero_streak_feature: 
- lag_rolling_features: 
- seasonality_features: 
- log_transform_candidates: 
- scaling_candidates: 

## SQL Schema Proposal
Table: None