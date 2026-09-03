"""
generate_data.py
-----------------
Generates a synthetic but statistically realistic patient discharge dataset
for a 30-day hospital readmission analysis.

WHY SYNTHETIC DATA:
Public, hospital-level readmission datasets (e.g., UCI's "Diabetes 130-US
Hospitals") were not reachable from this environment's network, so this
script generates a synthetic cohort instead. The generating process encodes
well-established, published clinical risk factors for 30-day readmission
(age, length of stay, prior admissions, comorbidity burden, discharge
disposition, follow-up scheduling, insurance type) through a logistic model,
so the relationships in the data mirror real-world patterns closely enough
to support genuine statistical analysis. This is disclosed explicitly in the
README and report — nothing here is presented as real patient data.

Reproducible via a fixed random seed.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7000

# ---- Demographics ----
age = np.clip(np.random.normal(64, 16, N), 18, 95).round().astype(int)
sex = np.random.choice(["Female", "Male"], N, p=[0.52, 0.48])
insurance = np.random.choice(
    ["Medicare", "Medicaid", "Private", "Uninsured"], N, p=[0.46, 0.18, 0.30, 0.06]
)

# ---- Admission characteristics ----
admission_type = np.random.choice(
    ["Emergency", "Urgent", "Elective"], N, p=[0.62, 0.20, 0.18]
)
length_of_stay = np.clip(np.random.gamma(shape=2.2, scale=1.9, size=N), 1, 30).round().astype(int)
num_procedures = np.random.poisson(1.3, N)
num_medications = np.clip(np.random.normal(12, 5, N), 0, 35).round().astype(int)
num_diagnoses = np.clip(np.random.poisson(5, N) + 1, 1, 16)

# ---- Prior utilization ----
prior_admissions_1yr = np.random.poisson(0.55, N)
prior_er_visits_1yr = np.random.poisson(0.7, N)

# ---- Comorbidities (binary flags) ----
diabetes = np.random.binomial(1, 0.28, N)
heart_failure = np.random.binomial(1, 0.17, N)
copd = np.random.binomial(1, 0.14, N)
ckd = np.random.binomial(1, 0.12, N)  # chronic kidney disease
comorbidity_count = diabetes + heart_failure + copd + ckd

# ---- Discharge / follow-up process ----
discharge_disposition = np.random.choice(
    ["Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"],
    N, p=[0.62, 0.20, 0.15, 0.03]
)
followup_scheduled_7d = np.random.binomial(1, 0.55, N)  # follow-up appt within 7 days
discharge_summary_sent_24h = np.random.binomial(1, 0.58, N)

primary_diagnosis_category = np.random.choice(
    ["Circulatory", "Respiratory", "Digestive", "Musculoskeletal",
     "Endocrine/Diabetes", "Injury/Trauma", "Other"],
    N, p=[0.24, 0.16, 0.12, 0.10, 0.13, 0.10, 0.15]
)

# ---- Build the true (latent) risk model for 30-day readmission ----
# Coefficients loosely reflect published readmission-risk literature
# (age, prior utilization, and comorbidity burden are consistently the
# strongest predictors; timely follow-up is consistently protective).
z = (
    -3.55
    + 0.014 * (age - 64)
    + 0.10 * length_of_stay
    + 0.38 * prior_admissions_1yr
    + 0.24 * prior_er_visits_1yr
    + 0.24 * comorbidity_count
    + 0.035 * num_medications
    + 0.07 * num_diagnoses
    - 0.50 * followup_scheduled_7d
    - 0.20 * discharge_summary_sent_24h
    + np.where(admission_type == "Emergency", 0.25, 0.0)
    + np.where(discharge_disposition == "Skilled Nursing Facility", 0.18, 0.0)
    + np.where(discharge_disposition == "Against Medical Advice", 0.50, 0.0)
    + np.where(insurance == "Uninsured", 0.22, 0.0)
    + np.where(insurance == "Medicaid", 0.13, 0.0)
    + np.random.normal(0, 0.45, N)  # unobserved variation / noise
)
prob_readmit = 1 / (1 + np.exp(-z))
readmitted_30d = np.random.binomial(1, prob_readmit)

df = pd.DataFrame({
    "patient_id": [f"P{100000+i}" for i in range(N)],
    "age": age,
    "sex": sex,
    "insurance_type": insurance,
    "admission_type": admission_type,
    "primary_diagnosis_category": primary_diagnosis_category,
    "length_of_stay_days": length_of_stay,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "num_diagnoses": num_diagnoses,
    "prior_admissions_1yr": prior_admissions_1yr,
    "prior_er_visits_1yr": prior_er_visits_1yr,
    "diabetes": diabetes,
    "heart_failure": heart_failure,
    "copd": copd,
    "chronic_kidney_disease": ckd,
    "discharge_disposition": discharge_disposition,
    "followup_scheduled_7d": followup_scheduled_7d,
    "discharge_summary_sent_24h": discharge_summary_sent_24h,
    "readmitted_30d": readmitted_30d,
})

# ---- Inject light, realistic messiness so cleaning is genuinely necessary ----
missing_idx = np.random.choice(N, size=int(N * 0.03), replace=False)
df.loc[missing_idx, "num_medications"] = np.nan

dup_idx = np.random.choice(N, size=15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

neg_idx = np.random.choice(len(df), size=8, replace=False)
df.loc[neg_idx, "length_of_stay_days"] = -1  # bad sentinel values

df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("data/raw_patient_discharges.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw_patient_discharges.csv")
print(f"True readmission rate (pre-cleaning): {df['readmitted_30d'].mean():.1%}")
