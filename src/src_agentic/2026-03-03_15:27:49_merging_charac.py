# ===== Merge Plan: caract-2023.csv_vs_caract-2024.csv =====
# ----- RECOMMENDATONS BEFORE MERGE ----- 
# TODO: Ensure dtype consistency for column '('action', 'cast_column dtype')'
# Example: caract-2023.csv['('action', 'cast_column dtype')'] = caract-2023.csv['('action', 'cast_column dtype')'].astype('int')
# Example: caract-2024.csv['('action', 'cast_column dtype')'] = caract-2024.csv['('action', 'cast_column dtype')'].astype('int')

# TODO: Ensure dtype consistency for column '('target', [])'
# Example: caract-2023.csv['('target', [])'] = caract-2023.csv['('target', [])'].astype('int')
# Example: caract-2024.csv['('target', [])'] = caract-2024.csv['('target', [])'].astype('int')

# TODO: Ensure dtype consistency for column '('params', None)'
# Example: caract-2023.csv['('params', None)'] = caract-2023.csv['('params', None)'].astype('int')
# Example: caract-2024.csv['('params', None)'] = caract-2024.csv['('params', None)'].astype('int')

# TODO: High cardinality detected in '('action', 'reduce high_cardinality')'.
# Consider aggregation, binning, frequency encoding or filtering.

# Option 1: Keep top-N categories
# top_vals = caract-2023.csv['('action', 'reduce high_cardinality')'].value_counts().nlargest(20).index
# caract-2023.csv['('action', 'reduce high_cardinality')'] = caract-2023.csv['('action', 'reduce high_cardinality')'].where(caract-2023.csv['('action', 'reduce high_cardinality')'].isin(top_vals), 'OTHER')

# Option 2: Frequency encoding
# freq_map = caract-2023.csv['('action', 'reduce high_cardinality')'].value_counts(normalize=True)
# caract-2023.csv['('action', 'reduce high_cardinality')_freq'] = caract-2023.csv['('action', 'reduce high_cardinality')'].map(freq_map)

# TODO: High cardinality detected in '('target', [])'.
# Consider aggregation, binning, frequency encoding or filtering.

# Option 1: Keep top-N categories
# top_vals = caract-2023.csv['('target', [])'].value_counts().nlargest(20).index
# caract-2023.csv['('target', [])'] = caract-2023.csv['('target', [])'].where(caract-2023.csv['('target', [])'].isin(top_vals), 'OTHER')

# Option 2: Frequency encoding
# freq_map = caract-2023.csv['('target', [])'].value_counts(normalize=True)
# caract-2023.csv['('target', [])_freq'] = caract-2023.csv['('target', [])'].map(freq_map)

# TODO: High cardinality detected in '('params', None)'.
# Consider aggregation, binning, frequency encoding or filtering.

# Option 1: Keep top-N categories
# top_vals = caract-2023.csv['('params', None)'].value_counts().nlargest(20).index
# caract-2023.csv['('params', None)'] = caract-2023.csv['('params', None)'].where(caract-2023.csv['('params', None)'].isin(top_vals), 'OTHER')

# Option 2: Frequency encoding
# freq_map = caract-2023.csv['('params', None)'].value_counts(normalize=True)
# caract-2023.csv['('params', None)_freq'] = caract-2023.csv['('params', None)'].map(freq_map)

# ---- Merge: caract-2023.csv_vs_caract-2024.csv ----
merged_df = pd.merge(caract-2023.csv, caract-2024.csv, on=['atm', 'jour', 'col', 'com', 'Num_Acc', 'mois', 'hrmn', 'an', 'adr', 'lum', 'int', 'lat', 'agg', 'long', 'dep'], how='inner')