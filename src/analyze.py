"""
analyze.py
-----------
Statistical analysis of 30-day readmission risk:
  1. Chi-square tests for categorical risk factors
  2. Multivariable logistic regression -> adjusted odds ratios
  3. A simple, transparent risk-scoring rule derived from the model
  4. Cost-impact estimate

Outputs a JSON summary (results/analysis_results.json) consumed by the
visualization and reporting steps, plus printed output for the log.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("data/clean_patient_discharges.csv")
results = {}


def fit_logistic_mle(X, y, feature_names, max_iter=100, tol=1e-8):
    """
    Fits a logistic regression by Newton-Raphson (statsmodels-free MLE),
    returning coefficients, Wald standard errors, and p-values so we can
    report adjusted odds ratios with confidence intervals — the same
    output a statsmodels Logit fit would give.
    """
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xd.shape[1])
    for _ in range(max_iter):
        z = Xd @ beta
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        grad = Xd.T @ (y - p)
        H = -(Xd.T * W) @ Xd
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    z = Xd @ beta
    p = 1 / (1 + np.exp(-z))
    W = p * (1 - p)
    H = -(Xd.T * W) @ Xd
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    zscores = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zscores)))
    names = ["Intercept"] + feature_names
    out = pd.DataFrame({
        "coef": beta, "se": se, "z": zscores, "p_value": pvals,
    }, index=names)
    out["ci_low"] = out["coef"] - 1.96 * out["se"]
    out["ci_high"] = out["coef"] + 1.96 * out["se"]

    # pseudo R^2 (McFadden)
    ll_full = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    p_null = np.full(len(y), y.mean())
    ll_null = np.sum(y * np.log(p_null + 1e-12) + (1 - y) * np.log(1 - p_null + 1e-12))
    pseudo_r2 = 1 - ll_full / ll_null
    return out, pseudo_r2

# ---------------------------------------------------------------
# 1. Chi-square tests: categorical factor vs. readmission
# ---------------------------------------------------------------
chi_sq_results = {}
categorical_factors = [
    "followup_scheduled_7d", "discharge_disposition", "insurance_type",
    "admission_type", "high_utilizer", "age_group"
]
for factor in categorical_factors:
    contingency = pd.crosstab(df[factor], df["readmitted_30d"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    chi_sq_results[factor] = {"chi2": round(chi2, 2), "p_value": round(p, 5)}

results["chi_square_tests"] = chi_sq_results
print("=== Chi-square tests (categorical factor vs. readmission) ===")
for k, v in chi_sq_results.items():
    sig = "***" if v["p_value"] < 0.001 else ("**" if v["p_value"] < 0.01 else ("*" if v["p_value"] < 0.05 else "ns"))
    print(f"  {k:28s} chi2={v['chi2']:>8.2f}  p={v['p_value']:.5f}  {sig}")

# ---------------------------------------------------------------
# 2. Multivariable logistic regression -> adjusted odds ratios
# ---------------------------------------------------------------
model_df = df.copy()
insurance_dummies = pd.get_dummies(model_df["insurance_type"], prefix="insurance", drop_first=False)
insurance_dummies = insurance_dummies.drop(columns=["insurance_Private"])  # Private = reference
disposition_dummies = pd.get_dummies(model_df["discharge_disposition"], prefix="disposition", drop_first=False)
disposition_dummies = disposition_dummies.drop(columns=["disposition_Home"])  # Home = reference
admission_dummies = pd.get_dummies(model_df["admission_type"], prefix="admission", drop_first=False)
admission_dummies = admission_dummies.drop(columns=["admission_Elective"])  # Elective = reference

numeric_features = model_df[[
    "age", "length_of_stay_days", "prior_admissions_1yr", "prior_er_visits_1yr",
    "comorbidity_count", "num_medications", "num_diagnoses",
    "followup_scheduled_7d", "discharge_summary_sent_24h",
]].astype(float)

X_design = pd.concat([numeric_features, insurance_dummies.astype(float),
                       disposition_dummies.astype(float), admission_dummies.astype(float)], axis=1)
feature_names = list(X_design.columns)
y = model_df["readmitted_30d"].values.astype(float)

# standardize continuous count/scale variables for numerical stability, then
# rescale coefficients back so odds ratios stay in original (interpretable) units
X_raw = X_design.values
scales = X_raw.std(axis=0)
scales[scales == 0] = 1
X_scaled = X_raw / scales

fit_result, pseudo_r2 = fit_logistic_mle(X_scaled, y, feature_names)
# rescale: beta_original = beta_scaled / scale
fit_result_rescaled = fit_result.copy()
for col in ["coef", "se", "ci_low", "ci_high"]:
    fit_result_rescaled.loc[feature_names, col] = fit_result.loc[feature_names, col].values / scales

or_table = fit_result_rescaled.drop(index="Intercept").copy()
or_table["odds_ratio"] = np.exp(or_table["coef"])
or_table["or_ci_low"] = np.exp(or_table["ci_low"])
or_table["or_ci_high"] = np.exp(or_table["ci_high"])
or_table = or_table.sort_values("odds_ratio")

print("\n=== Logistic regression: adjusted odds ratios ===")
print(or_table[["odds_ratio", "or_ci_low", "or_ci_high", "p_value"]].round(3))

results["logistic_regression"] = {
    "pseudo_r2": round(pseudo_r2, 4),
    "n_obs": int(len(y)),
    "odds_ratios": {
        idx: {
            "odds_ratio": round(row.odds_ratio, 3),
            "ci_low": round(row.or_ci_low, 3),
            "ci_high": round(row.or_ci_high, 3),
            "p_value": round(row.p_value, 5),
        }
        for idx, row in or_table.iterrows()
    }
}
or_table.to_csv("data/odds_ratios.csv")

# Model discrimination: AUC via simple train/holdout split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

X = X_design  # reuse the same design matrix built for the odds-ratio model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
results["logistic_regression"]["holdout_auc"] = round(auc, 3)
print(f"\nHoldout AUC: {auc:.3f}")

# ---------------------------------------------------------------
# 3. Transparent risk score (simple point system clinicians can use)
#    Derived from which factors are strongest/significant in the model.
# ---------------------------------------------------------------
def risk_points(row):
    pts = 0
    pts += 2 if row["prior_admissions_1yr"] >= 2 else (1 if row["prior_admissions_1yr"] == 1 else 0)
    pts += 2 if row["prior_er_visits_1yr"] >= 2 else (1 if row["prior_er_visits_1yr"] == 1 else 0)
    pts += 1 if row["comorbidity_count"] >= 2 else 0
    pts += 1 if row["length_of_stay_days"] >= 7 else 0
    pts += 1 if row["age"] >= 75 else 0
    pts -= 2 if row["followup_scheduled_7d"] == 1 else 0
    pts += 1 if row["discharge_disposition"] in ["Skilled Nursing Facility", "Against Medical Advice"] else 0
    return pts

df["risk_score"] = df.apply(risk_points, axis=1)
df["risk_tier"] = pd.cut(
    df["risk_score"], bins=[-10, 0, 2, 4, 20],
    labels=["Low", "Moderate", "High", "Very High"]
)
tier_summary = df.groupby("risk_tier", observed=True)["readmitted_30d"].agg(["mean", "count"])
results["risk_tiers"] = {
    str(idx): {"readmit_rate": round(row["mean"], 4), "n": int(row["count"])}
    for idx, row in tier_summary.iterrows()
}
print("\n=== Risk tier validation ===")
print(tier_summary)

df.to_csv("data/clean_patient_discharges.csv", index=False)  # persist risk_score/tier

# ---------------------------------------------------------------
# 4. Cost impact estimate
# ---------------------------------------------------------------
AVG_COST_PER_READMISSION = 15200  # published CMS/HCUP-range estimate, documented in report
n_readmits = int(df["readmitted_30d"].sum())
total_cost = n_readmits * AVG_COST_PER_READMISSION

no_followup = df[df["followup_scheduled_7d"] == 0]
followup = df[df["followup_scheduled_7d"] == 1]
rate_no_followup = no_followup["readmitted_30d"].mean()
rate_followup = followup["readmitted_30d"].mean()
potential_prevented = int((rate_no_followup - rate_followup) * len(no_followup))
potential_savings = potential_prevented * AVG_COST_PER_READMISSION

results["cost_impact"] = {
    "avg_cost_per_readmission": AVG_COST_PER_READMISSION,
    "total_readmissions": n_readmits,
    "total_estimated_cost": int(total_cost),
    "readmit_rate_no_followup": round(rate_no_followup, 4),
    "readmit_rate_with_followup": round(rate_followup, 4),
    "patients_without_followup": int(len(no_followup)),
    "potentially_preventable_readmissions": potential_prevented,
    "potential_annual_savings": int(potential_savings),
}
print("\n=== Cost impact ===")
print(json.dumps(results["cost_impact"], indent=2))

with open("data/analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: data/analysis_results.json")
print("Saved: data/odds_ratios.csv")
