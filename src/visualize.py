"""
visualize.py
--------------
Produces 6 publication-quality figures from the cleaned data and analysis
results. One consistent style and palette across all charts; each chart
makes exactly one point.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------
# Shared style: one palette, one type ramp, minimal chart junk
# ---------------------------------------------------------------
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
ORANGE = "#eb6834"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"

SEQ_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]  # for ordinal risk tiers

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
})


def clean_axes(ax, hide_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not hide_y)
    if hide_y:
        ax.yaxis.set_visible(False)
    ax.grid(axis="y" if not hide_y else "x", alpha=0.6)
    ax.grid(axis="x", visible=False) if not hide_y else None


df = pd.read_csv("data/clean_patient_discharges.csv")
with open("data/analysis_results.json") as f:
    results = json.load(f)

overall_rate = df["readmitted_30d"].mean()

# =================================================================
# FIGURE 1 — Headline: readmission rate by follow-up scheduling
# =================================================================
fig, ax = plt.subplots(figsize=(8, 5.2))
grp = df.groupby("followup_scheduled_7d")["readmitted_30d"].mean()
labels = ["No follow-up\nscheduled", "Follow-up scheduled\nwithin 7 days"]
vals = [grp[0], grp[1]]
colors = [CRITICAL, GOOD]
bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="none")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.1%}",
             ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
ax.axhline(overall_rate, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
ax.text(1.62, overall_rate + 0.006, f"overall avg {overall_rate:.1%}", va="bottom", ha="center",
        fontsize=9.5, color=INK_MUTED, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
ax.set_ylim(0, max(vals) * 1.28)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylabel("30-day readmission rate")
ax.set_title("Scheduling a 7-day follow-up nearly halves readmission risk",
             fontsize=13.5, fontweight="bold", color=INK, pad=14, loc="left")
ax.set_xlim(-0.5, 1.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig("figures/01_followup_impact.png", dpi=200)
plt.close()

# =================================================================
# FIGURE 2 — Adjusted odds ratio forest plot
# =================================================================
or_df = pd.read_csv("data/odds_ratios.csv", index_col=0)
label_map = {
    "prior_admissions_1yr": "Prior admissions (per +1, last yr)",
    "comorbidity_count": "Comorbidity count (per +1)",
    "admission_Emergency": "Admitted via Emergency (vs. Elective)",
    "disposition_Against Medical Advice": "Discharged Against Medical Advice (vs. Home)",
    "prior_er_visits_1yr": "Prior ER visits (per +1, last yr)",
    "disposition_Skilled Nursing Facility": "Discharged to Skilled Nursing (vs. Home)",
    "length_of_stay_days": "Length of stay (per +1 day)",
    "num_diagnoses": "Number of diagnoses (per +1)",
    "num_medications": "Number of medications (per +1)",
    "age": "Age (per +1 year)",
    "insurance_Uninsured": "Uninsured (vs. Private)",
    "disposition_Home Health Care": "Discharged to Home Health (vs. Home)",
    "admission_Urgent": "Admitted Urgent (vs. Elective)",
    "insurance_Medicare": "Medicare (vs. Private)",
    "insurance_Medicaid": "Medicaid (vs. Private)",
    "discharge_summary_sent_24h": "Discharge summary sent <24h",
    "followup_scheduled_7d": "Follow-up scheduled within 7 days",
}
plot_vars = [
    "followup_scheduled_7d", "discharge_summary_sent_24h", "age",
    "num_medications", "num_diagnoses", "length_of_stay_days",
    "prior_er_visits_1yr", "disposition_Skilled Nursing Facility",
    "admission_Emergency", "comorbidity_count", "prior_admissions_1yr",
]
plot_df = or_df.loc[plot_vars].copy()
plot_df["label"] = [label_map[v] for v in plot_df.index]
plot_df = plot_df.sort_values("odds_ratio")

fig, ax = plt.subplots(figsize=(10.5, 7))
y_pos = np.arange(len(plot_df))
colors = [GOOD if v < 1 else (BLUE if p < 0.05 else INK_MUTED)
          for v, p in zip(plot_df["odds_ratio"], plot_df["p_value"])]
ax.hlines(y_pos, plot_df["or_ci_low"], plot_df["or_ci_high"], color=colors, linewidth=2)
ax.scatter(plot_df["odds_ratio"], y_pos, color=colors, s=55, zorder=3, edgecolor=SURFACE, linewidth=1)
ax.axvline(1.0, color=INK, linewidth=1, linestyle=(0, (3, 3)))
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["label"], fontsize=10.5)
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}x"))
ax.set_xlabel("Adjusted odds ratio for 30-day readmission (log scale)")
ax.set_title("What actually moves readmission odds, holding other factors constant",
              fontsize=12.5, fontweight="bold", color=INK, pad=14, loc="left")
fig.text(0.99, 0.015, "Blue/green = statistically significant (p<0.05)   Gray = not significant",
          ha="right", fontsize=9, color=INK_MUTED)
clean_axes(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", alpha=0.5)
plt.tight_layout(rect=(0, 0.035, 1, 1))
plt.savefig("figures/02_odds_ratio_forest.png", dpi=200)
plt.close()

# =================================================================
# FIGURE 3 — Readmission rate by risk tier (validates the risk score)
# =================================================================
tier_order = ["Low", "Moderate", "High", "Very High"]
tier_df = df.groupby("risk_tier", observed=True).agg(
    rate=("readmitted_30d", "mean"), n=("readmitted_30d", "size")
).reindex(tier_order)

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(tier_df.index, tier_df["rate"], color=SEQ_BLUES, width=0.6)
for b, (rate, n) in zip(bars, zip(tier_df["rate"], tier_df["n"])):
    ax.text(b.get_x() + b.get_width() / 2, rate + 0.008, f"{rate:.1%}",
             ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(b.get_x() + b.get_width() / 2, -0.028, f"n={n:,}",
             ha="center", fontsize=9, color=INK_MUTED)
ax.set_ylim(-0.05, tier_df["rate"].max() * 1.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylabel("30-day readmission rate")
ax.set_xlabel("Discharge risk tier (rule-based score)")
ax.set_title("A simple 4-tier risk score cleanly separates readmission risk",
              fontsize=13.5, fontweight="bold", color=INK, pad=14)
clean_axes(ax)
plt.tight_layout()
plt.savefig("figures/03_risk_tier_validation.png", dpi=200)
plt.close()

# =================================================================
# FIGURE 4 — Readmission rate by prior admissions (dose-response)
# =================================================================
df["prior_admissions_capped"] = df["prior_admissions_1yr"].clip(upper=3)
grp4 = df.groupby("prior_admissions_capped")["readmitted_30d"].agg(["mean", "size"])
labels4 = ["0", "1", "2", "3+"]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(labels4, grp4["mean"], color=BLUE, linewidth=2.5, marker="o",
        markersize=9, markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
ax.fill_between(labels4, grp4["mean"], color=BLUE, alpha=0.08)
for x, y in zip(labels4, grp4["mean"]):
    ax.text(x, y + 0.012, f"{y:.1%}", ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.set_ylim(0, grp4["mean"].max() * 1.35)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylabel("30-day readmission rate")
ax.set_xlabel("Hospital admissions in the prior 12 months")
ax.set_title("Readmission risk rises sharply with recent admission history",
              fontsize=13.5, fontweight="bold", color=INK, pad=14)
clean_axes(ax)
plt.tight_layout()
plt.savefig("figures/04_prior_admissions_doseresponse.png", dpi=200)
plt.close()

# =================================================================
# FIGURE 5 — Cost impact: actual vs. potentially preventable
# =================================================================
ci = results["cost_impact"]
total_cost_m = ci["total_estimated_cost"] / 1e6
savings_m = ci["potential_annual_savings"] / 1e6
remaining_m = total_cost_m - savings_m

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(["Estimated annual\nreadmission cost"], [remaining_m], color=BLUE_DARK, height=0.45, label="Remaining cost")
ax.barh(["Estimated annual\nreadmission cost"], [savings_m], left=[remaining_m], color=GOOD, height=0.45,
        label="Potentially preventable (closing the follow-up gap)")
ax.text(remaining_m / 2, 0, f"\\${remaining_m:.1f}M", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
ax.text(remaining_m + savings_m / 2, 0, f"\\${savings_m:.1f}M", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
ax.set_xlabel(f"Estimated annual cost, millions of dollars (at \\${ci['avg_cost_per_readmission']/1000:.1f}K per readmission)")
ax.set_title(f"Closing the follow-up gap could save an estimated \\${savings_m:.1f}M per year",
              fontsize=13.5, fontweight="bold", color=INK, pad=14)
ax.legend(loc="upper center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.22), ncol=1)
ax.set_xlim(0, total_cost_m * 1.1)
clean_axes(ax, hide_y=False)
ax.grid(axis="x", alpha=0.5)
ax.grid(axis="y", visible=False)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("figures/05_cost_impact.png", dpi=200)
plt.close()

# =================================================================
# FIGURE 6 — Readmission rate by comorbidity burden
# =================================================================
# cap at 3+ (n=12 patients have all 4 comorbidities -- too few to plot reliably)
df["comorbidity_capped"] = df["comorbidity_count"].clip(upper=3)
grp6 = df.groupby("comorbidity_capped")["readmitted_30d"].agg(["mean", "size"])
labels6 = ["0", "1", "2", "3+"]

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(labels6, grp6["mean"], color=BLUE, width=0.55)
for b, v, n in zip(bars, grp6["mean"], grp6["size"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.009, f"{v:.1%}",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(b.get_x() + b.get_width() / 2, -0.022, f"n={n:,}",
             ha="center", fontsize=9, color=INK_MUTED)
ax.set_ylim(-0.035, grp6["mean"].max() * 1.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylabel("30-day readmission rate")
ax.set_xlabel("Number of chronic comorbidities (diabetes, heart failure, COPD, CKD)")
ax.set_title("Each additional chronic condition compounds readmission risk",
              fontsize=13.5, fontweight="bold", color=INK, pad=14)
clean_axes(ax)
plt.tight_layout()
plt.savefig("figures/06_comorbidity_burden.png", dpi=200)
plt.close()

print("Saved 6 figures to figures/")
