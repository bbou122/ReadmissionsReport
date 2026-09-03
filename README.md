# Reducing 30-Day Hospital Readmissions

A healthcare analytics project identifying which patients are most likely to be
readmitted within 30 days of discharge, why, and what a hospital could realistically
do about it — built end to end (data generation → cleaning → statistical modeling →
visualization → executive reporting) as a portfolio piece.

**[Read the full report ->](./report/index.html)** (open in any browser)

## The business problem

Hospitals are penalized financially (via CMS's Hospital Readmissions Reduction
Program) when too many patients bounce back within 30 days, and every readmission
also carries a direct cost of roughly $15,000 in the U.S. A hospital's care
coordination team asked: *which patients are highest-risk at the moment of
discharge, which of the things we control (follow-up scheduling, discharge
paperwork) actually move the needle, and what would fixing that be worth?*

## Data

**This project uses a synthetically generated dataset, not real patient data —
documented explicitly, not hidden.** A public dataset (UCI's "Diabetes 130-US
Hospitals") was the original target, but was not reachable over the network
available in the build environment. Rather than fake a data source, `src/generate_data.py`
builds a 7,000-patient cohort from a logistic model whose coefficients are
calibrated to match published readmission-risk literature (prior utilization,
comorbidity burden, and follow-up timing are consistently the strongest real-world
predictors), so the statistical relationships in the data are realistic even
though no individual record is real. Light, intentional messiness (missing values,
duplicate records, a few impossible sentinel values) is injected so the cleaning
step is doing genuine work, not theater. See the report's Methodology & Limitations
section for the full disclosure.

## What's in this repo

```
hospital-readmissions-project/
├── data/
│   ├── raw_patient_discharges.csv       # synthetic raw extract (7,015 rows incl. dupes/errors)
│   ├── clean_patient_discharges.csv     # analysis-ready (7,000 rows)
│   ├── cleaning_log.txt                 # every cleaning decision, logged
│   ├── odds_ratios.csv                  # logistic regression output
│   └── analysis_results.json            # all stats in one machine-readable file
├── figures/                             # 6 final PNG charts (200 dpi)
├── src/
│   ├── generate_data.py                 # synthetic cohort generator
│   ├── clean_data.py                    # cleaning pipeline + audit log
│   ├── analyze.py                       # chi-square tests, logistic regression, risk score, cost model
│   └── visualize.py                     # all 6 charts, one consistent style
├── report/
│   └── index.html                       # the polished, standalone report (open this)
├── requirements.txt
└── README.md
```

## Methodology

1. **Cleaning** (`clean_data.py`): dropped 15 duplicate patient records, flagged
   8 impossible negative length-of-stay values as missing, median-imputed 218
   missing values across two columns, enforced types, ran sanity-bound assertions.
   Every step is logged to `data/cleaning_log.txt`.
2. **Univariate testing** (`analyze.py`): chi-square tests of independence between
   each categorical risk factor and 30-day readmission.
3. **Multivariable modeling**: a logistic regression (fit via Newton-Raphson MLE,
   implemented directly with NumPy/SciPy — no `statsmodels` dependency, since the
   build environment couldn't reach PyPI for it) estimating each factor's
   **adjusted** odds ratio, i.e. its effect holding the other 16 factors constant.
   Reported with 95% Wald confidence intervals and p-values. Discrimination was
   checked out-of-sample (holdout AUC = 0.681, in the realistic range for this kind
   of administrative/clinical feature set — real readmission models in the
   literature typically land 0.65–0.75).
4. **Risk scoring**: a simple, transparent point-based rule derived from the
   regression's strongest, most significant predictors — the kind of thing a
   care coordination team could actually apply at the nurses' station without a
   model API. Validated by checking that it produces a clean, monotonic
   readmission-rate gradient across four risk tiers (10.0% → 16.5% → 25.9% →
   40.4%).
5. **Cost impact**: applied a published mid-range per-readmission cost estimate
   (~$15,200) to quantify the dollar impact of the single most actionable finding.

## Key findings

- Patients **without** a 7-day follow-up appointment scheduled at discharge are
  readmitted at **18.1%** vs. **13.0%** for those with one scheduled — a gap that
  survives adjustment for age, comorbidities, and prior utilization (adjusted OR
  0.65, p < 0.001).
- **Prior utilization is the single strongest predictor**: each additional
  hospital admission in the past year raises the adjusted odds of readmission by
  ~56%; each additional ER visit by ~25%.
- **Comorbidity burden compounds risk**: patients with 3+ of {diabetes, heart
  failure, COPD, chronic kidney disease} are readmitted at 23.6% vs. 12.7% for
  patients with none.
- The 4-tier risk score built from these factors cleanly separates a 10.0%-risk
  "Low" group from a 40.4%-risk "Very High" group using only data available at
  discharge.
- Closing the follow-up-scheduling gap for the ~45% of patients currently
  discharged without one is estimated to prevent **~158 readmissions/year** and
  save **~$2.4M/year**, against a total estimated annual readmission cost of
  ~$16.3M in this cohort.

## Limitations (stated plainly)

- **Synthetic data.** Real hospital data would have messier, less cleanly
  separable relationships, missing-not-at-random patterns, and site-specific
  quirks this dataset doesn't capture.
- **Cross-sectional, not causal.** The follow-up-scheduling effect is adjusted
  for observed confounders but not randomized — patients who get follow-ups
  scheduled may differ in unmeasured ways (e.g., health literacy, family support).
  A true causal claim would need a randomized or quasi-experimental design.
- **Pseudo-R² of 0.066 / AUC of 0.68** means the model explains real but modest
  signal — appropriate for prioritization and resource targeting, not for
  denying care or making individual clinical decisions.
- Cost estimate is a single published average, not this (synthetic) hospital's
  actual case-mix-adjusted cost.

## What I'd do with more time/data

- Get access to a real de-identified dataset (e.g., through a data use agreement)
  and re-run the identical pipeline.
- Add a proper train/test/temporal-holdout split with calibration curves, not
  just AUC.
- Test the follow-up effect with a quasi-experimental design (e.g., difference-in-
  differences around a scheduling policy change) to strengthen the causal claim.
- Build a small Streamlit/Flask front-end so care coordinators could score a
  patient at discharge interactively.

## How to reproduce

```bash
pip install -r requirements.txt
cd src
python generate_data.py   # -> data/raw_patient_discharges.csv
python clean_data.py      # -> data/clean_patient_discharges.csv
python analyze.py         # -> data/analysis_results.json, odds_ratios.csv
python visualize.py       # -> figures/*.png
```

## Author

Braden Bourg — built as a portfolio project to demonstrate an end-to-end data
analytics workflow: business framing, data cleaning, statistical inference
(not just descriptive stats), reproducible code, and stakeholder-ready
communication.
