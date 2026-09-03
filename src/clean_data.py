"""
clean_data.py
--------------
Loads the raw discharge extract and produces an analysis-ready dataset.
Every cleaning decision is logged so the process is auditable.
"""

import pandas as pd
import numpy as np

log = []

def note(msg):
    log.append(msg)
    print(msg)

df = pd.read_csv("data/raw_patient_discharges.csv")
note(f"Loaded raw file: {len(df)} rows, {df.shape[1]} columns")

# 1. Drop exact duplicate patient records
before = len(df)
df = df.drop_duplicates(subset="patient_id", keep="first")
note(f"Removed {before - len(df)} duplicate patient_id rows")

# 2. Fix impossible values: length_of_stay cannot be negative
bad_los = (df["length_of_stay_days"] < 0).sum()
df.loc[df["length_of_stay_days"] < 0, "length_of_stay_days"] = np.nan
note(f"Flagged {bad_los} negative length-of-stay values as missing")

# 3. Impute missing numeric values with the column median (robust to outliers)
for col in ["num_medications", "length_of_stay_days"]:
    n_missing = df[col].isna().sum()
    if n_missing:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        note(f"Imputed {n_missing} missing '{col}' values with median ({median_val:.1f})")

# 4. Type enforcement
int_cols = ["age", "length_of_stay_days", "num_procedures", "num_medications",
            "num_diagnoses", "prior_admissions_1yr", "prior_er_visits_1yr"]
for c in int_cols:
    df[c] = df[c].round().astype(int)

# 5. Derived fields used throughout the analysis
df["comorbidity_count"] = (
    df["diabetes"] + df["heart_failure"] + df["copd"] + df["chronic_kidney_disease"]
)
df["age_group"] = pd.cut(
    df["age"], bins=[17, 44, 64, 79, 96],
    labels=["18-44", "45-64", "65-79", "80+"]
)
df["high_utilizer"] = (
    (df["prior_admissions_1yr"] >= 2) | (df["prior_er_visits_1yr"] >= 2)
).astype(int)

# 6. Sanity bounds check
assert df["age"].between(18, 95).all(), "Age out of expected bounds"
assert df["readmitted_30d"].isin([0, 1]).all(), "Target not binary"
note("Passed sanity checks: age bounds, binary target")

note(f"Final analysis-ready dataset: {len(df)} rows, {df.shape[1]} columns")
note(f"30-day readmission rate: {df['readmitted_30d'].mean():.1%}")

df.to_csv("data/clean_patient_discharges.csv", index=False)

with open("data/cleaning_log.txt", "w") as f:
    f.write("\n".join(log))

print("\nSaved: data/clean_patient_discharges.csv")
print("Saved: data/cleaning_log.txt")
