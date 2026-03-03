## generate raw_data_scripts
# imports
import os

import src.utils.general_helper as gh
import src.utils.path_helper as ph

from src.core.finding_classes import ActionSchema
from src.core.report_classes import PreparationSummary


###################

def save_script(script, name, base_path=None):

    # timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")

    if base_path is None:
        # (1) load config
        gh.load_env_vars()
        base_path = os.getenv("FOLDER_SCRIPT")

    # json_path = f"{base_path}/audit_{name}.json"
    py_path = f"{base_path}/{name}.py"
    ph.ensure_dir(py_path)

    # py_file
    with open(py_path, "w") as f:
        f.write(script)

    return {"markdown": py_path}



def handle_drop_duplicates(action, var_name):
    subset = action.params.get("subset") if action.params else None
    keep = action.params.get("keep", "first") if action.params else "first"

    if subset:
        return [
            f"{var_name} = {var_name}.drop_duplicates(subset={subset}, keep='{keep}')"
        ]
    else:
        return [
            f"{var_name} = {var_name}.drop_duplicates(keep='{keep}')"
        ]


def handle_parse_datetime(action, var_name):
    lines = []
    for col in action.target or []:
        lines.append(
            f"{var_name}['{col}'] = pd.to_datetime({var_name}['{col}'], errors='coerce')"
        )
    return lines


def handle_impute_nan(action, var_name):
    strategy = action.params.get("strategy", "median") if action.params else "median"
    lines = []

    for col in action.target or []:
        if strategy == "median":
            lines.append(
                f"{var_name}['{col}'] = {var_name}['{col}'].fillna({var_name}['{col}'].median())"
            )
        elif strategy == "mean":
            lines.append(
                f"{var_name}['{col}'] = {var_name}['{col}'].fillna({var_name}['{col}'].mean())"
            )
        elif strategy == "zero":
            lines.append(
                f"{var_name}['{col}'] = {var_name}['{col}'].fillna(0)"
            )
        elif strategy == "drop":
            lines.append(
                f"{var_name}['{col}'] = {var_name}['{col}'].dropna()"
            )
        else:
            lines.append(
                f"# TODO: implement custom imputation for '{col}'"
            )

    return lines


def handle_scaling(action, var_name):
    lines = []
    lines.append("scaler = StandardScaler()")

    for col in action.target or []:
        lines.append(
            f"{var_name}['scaled_{col}'] = scaler.fit_transform({var_name}[['{col}']])"
        )

    return lines


def handle_log_transform(action, var_name):
    lines = []

    for col in action.target or []:
        lines.append(
            f"{var_name}['log_{col}'] = np.log1p({var_name}['{col}'])"
        )

    return lines


def generate_cleaning_script(summary: PreparationSummary):

    lines = []
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("from sklearn.preprocessing import StandardScaler")
    lines.append("")

    for file_name, actions in summary.processing.items():

        # anpassen falls parquet o.ä. 
        var_name = file_name.replace(".", "_")
        lines.append(f"try:")
        lines.append(f"    {var_name} = pd.read_csv('{file_name}')")
        lines.append(f"except Exception as e:")
        lines.append(f"    print(\"Error while loading '{file_name}': {{e}}\")")

        for action_dict in actions:

            action = ActionSchema(**action_dict)
            handler = ACTION_DISPATCH.get(action.action)

            if not handler:
                lines.append(f"# WARNING: No handler implemented for '{action.action}'")
                continue

            lines.extend(handler(action, var_name))
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def generate_merge_script(summary: PreparationSummary):

    lines = []

    for name, strategy in summary.merge.items():

        keys = strategy.get("recommended_join_keys")
        join_type = strategy.get("join_type", "inner")
        df_a = strategy.get("file_a", None)
        df_b = strategy.get("file_b", None)
        pre_merge = strategy.get("pre_merge_actions", {})
        post_merge = strategy.get("post_merge_actions", {})

        if not keys or not df_a or not df_b:
            continue

        lines.append(f"# ===== Merge Plan: {name} =====")

        if pre_merge:
            lines.append("# ----- RECOMMENDATONS BEFORE MERGE ----- ")
            for action, cols in pre_merge.items():
                if action == "cast_column":
                    for col in cols:
                        lines.append(
                            f"# TODO: Ensure dtype consistency for column '{col}'"
                        )
                        lines.append(
                            f"# Example: {df_a}['{col}'] = {df_a}['{col}'].astype('int')"
                        )
                        lines.append(
                            f"# Example: {df_b}['{col}'] = {df_b}['{col}'].astype('int')"
                        )
                        lines.append("")

                if action == "high_cardinality":
                    for col in cols:
                        lines.append(
                            f"# TODO: High cardinality detected in '{col}'."
                        )
                        lines.append(
                            f"# Consider aggregation, binning, frequency encoding or filtering."
                        )
                        lines.append("")
                        lines.append(f"# Option 1: Keep top-N categories")
                        lines.append(
                            f"# top_vals = {df_a}['{col}'].value_counts().nlargest(20).index"
                        )
                        lines.append(
                            f"# {df_a}['{col}'] = {df_a}['{col}'].where({df_a}['{col}'].isin(top_vals), 'OTHER')"
                        )
                        lines.append("")
                        lines.append(f"# Option 2: Frequency encoding")
                        lines.append(
                            f"# freq_map = {df_a}['{col}'].value_counts(normalize=True)"
                        )
                        lines.append(
                            f"# {df_a}['{col}_freq'] = {df_a}['{col}'].map(freq_map)"
                        )
                        lines.append("")


        lines.append(f"# ---- Merge: {name} ----")
        lines.append(
            f"merged_df = pd.merge({df_a}, {df_b}, on={keys}, how='{join_type}')"
        )

        if post_merge:
            lines.append("# ----- RECOMMENDATONS AFTER MERGE ----- ")
            for check in post_merge:
                lines.append(f"# {check}")
            lines.append("")

    return "\n".join(lines)



def generate_sql_query(summary: PreparationSummary):

    schema = summary.sql_schema
    if not schema:
        return ""

    lines = []
    lines.append(f"CREATE TABLE {schema['table_name']} (")

    cols = []
    for col, dtype in schema["columns"].items():
        cols.append(f"    {col} {dtype}")

    # lines.append(",\n".join(cols))

    if schema.get("primary_key"):
        pk = ", ".join(schema["primary_key"])
        lines.append(f"    PRIMARY KEY ({pk})")

    lines.append(",\n).join(cols)")
    lines.append(");")

    return "\n".join(lines)


ACTION_DISPATCH = {
    "drop_duplicates": handle_drop_duplicates,
    "parse_datetime": handle_parse_datetime,
    "impute_nan": handle_impute_nan,
    # "impute_inf": handle_impute_inf,   # analog definieren
    "scaling_candidate": handle_scaling,
    "log_transform_candidate": handle_log_transform,
}