#!/usr/bin/env python3

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results/APOBEC_PCA_with_country.csv"
)

OUT = ROOT / "results/revision_R1"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SEED = 20260920
N_BOOT = 2000
N_SPLITS = 10

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

required = [
    "APOBEC_prop",
    "Total_SBS_burden",
    "SBS1",
    "SBS3",
    "SBS5",
    "SBS18",
    "Country"
]

missing = [x for x in required if x not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

dat = df[required].dropna().copy()

continuous = [
    "Total_SBS_burden",
    "SBS1",
    "SBS3",
    "SBS5",
    "SBS18"
]

print("=" * 90)
print("REVIEWER 2 — ORIGINAL REGRESSION DIAGNOSTICS")
print("=" * 90)

print(f"\nN complete cases: {len(dat)}")

print("\nCountry counts:")
print(dat["Country"].value_counts().to_string())


# ============================================================
# 1. EXACT ORIGINAL MODELS
# ============================================================

formula1 = (
    "APOBEC_prop ~ "
    "Total_SBS_burden + SBS1 + SBS3 + SBS5 + SBS18"
)

formula2 = (
    "APOBEC_prop ~ "
    "Total_SBS_burden + SBS1 + SBS3 + SBS5 + SBS18 + C(Country)"
)

m1 = smf.ols(formula1, data=dat).fit()
m2 = smf.ols(formula2, data=dat).fit()

with open(TABLES / "R2_original_model1_summary.txt", "w") as f:
    f.write(m1.summary().as_text())

with open(TABLES / "R2_original_model2_country_summary.txt", "w") as f:
    f.write(m2.summary().as_text())


model_fit = pd.DataFrame([
    {
        "Model": "Model1_no_country",
        "N": int(m1.nobs),
        "R2": m1.rsquared,
        "Adjusted_R2": m1.rsquared_adj,
        "AIC": m1.aic,
        "BIC": m1.bic
    },
    {
        "Model": "Model2_with_country",
        "N": int(m2.nobs),
        "R2": m2.rsquared,
        "Adjusted_R2": m2.rsquared_adj,
        "AIC": m2.aic,
        "BIC": m2.bic
    }
])

model_fit.to_csv(
    TABLES / "R2_original_model_fit.csv",
    index=False
)


# ============================================================
# 2. VIF — CONTINUOUS PREDICTORS
# ============================================================

X = dat[continuous].copy()
X_const = sm.add_constant(X)

vif_rows = []

for i, col in enumerate(X_const.columns):
    vif_rows.append({
        "Variable": col,
        "VIF": variance_inflation_factor(
            X_const.values, i
        )
    })

vif = pd.DataFrame(vif_rows)

vif.to_csv(
    TABLES / "R2_original_model_VIF.csv",
    index=False
)


# ============================================================
# 3. CONDITION INDICES
#
# Standardize continuous predictors first, then add intercept.
# Condition index = largest singular value / each singular value.
# ============================================================

Xs = StandardScaler().fit_transform(
    dat[continuous]
)

X_condition = np.column_stack([
    np.ones(len(dat)),
    Xs
])

singular_values = np.linalg.svd(
    X_condition,
    full_matrices=False,
    compute_uv=False
)

condition_indices = (
    singular_values.max() /
    singular_values
)

condition_table = pd.DataFrame({
    "Dimension": np.arange(
        1,
        len(condition_indices) + 1
    ),
    "Singular_value": singular_values,
    "Condition_index": condition_indices
})

condition_table.to_csv(
    TABLES / "R2_original_model_condition_indices.csv",
    index=False
)


# ============================================================
# 4. 10-FOLD CROSS-VALIDATED R2
#
# Model 1 first because it directly evaluates the original
# genomic predictor model without categorical encoding issues.
# ============================================================

X_cv = dat[continuous].to_numpy()
y_cv = dat["APOBEC_prop"].to_numpy()

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

lr = LinearRegression()

cv_scores = cross_val_score(
    lr,
    X_cv,
    y_cv,
    cv=kf,
    scoring="r2"
)

cv_table = pd.DataFrame({
    "Fold": np.arange(1, N_SPLITS + 1),
    "R2": cv_scores
})

cv_table.to_csv(
    TABLES / "R2_original_model_10fold_CV_R2.csv",
    index=False
)


# ============================================================
# 5. BOOTSTRAP MODEL 1
#
# Store R2 and coefficients from bootstrap samples.
# ============================================================

rng = np.random.default_rng(SEED)

boot_rows = []

for b in range(N_BOOT):

    idx = rng.integers(
        0,
        len(dat),
        len(dat)
    )

    boot = dat.iloc[idx]

    fit = smf.ols(
        formula1,
        data=boot
    ).fit()

    row = {
        "Bootstrap": b + 1,
        "R2": fit.rsquared,
        "Adjusted_R2": fit.rsquared_adj
    }

    for term in fit.params.index:
        row[term] = fit.params[term]

    boot_rows.append(row)

boot = pd.DataFrame(boot_rows)

boot.to_csv(
    TABLES / "R2_original_model_bootstrap_raw.csv",
    index=False
)


def percentile_summary(series):

    return {
        "Mean": series.mean(),
        "Median": series.median(),
        "CI_low": series.quantile(0.025),
        "CI_high": series.quantile(0.975)
    }


boot_summary_rows = []

for variable in [
    "R2",
    "Adjusted_R2",
    "Intercept",
    *continuous
]:

    s = percentile_summary(
        boot[variable]
    )

    boot_summary_rows.append({
        "Variable": variable,
        **s
    })

boot_summary = pd.DataFrame(
    boot_summary_rows
)

boot_summary.to_csv(
    TABLES / "R2_original_model_bootstrap_summary.csv",
    index=False
)


# ============================================================
# 6. COEFFICIENT STABILITY:
#    REMOVE EACH PREDICTOR ONE AT A TIME
# ============================================================

stability_rows = []

full_params = m1.params

for removed in continuous:

    kept = [
        x for x in continuous
        if x != removed
    ]

    formula = (
        "APOBEC_prop ~ " +
        " + ".join(kept)
    )

    fit = smf.ols(
        formula,
        data=dat
    ).fit()

    for term in kept:

        full_beta = full_params[term]
        reduced_beta = fit.params[term]

        if full_beta != 0:
            pct_change = (
                (reduced_beta - full_beta)
                / abs(full_beta)
            ) * 100
        else:
            pct_change = np.nan

        stability_rows.append({
            "Removed_predictor": removed,
            "Remaining_predictor": term,
            "Full_model_beta": full_beta,
            "Reduced_model_beta": reduced_beta,
            "Percent_change_beta": pct_change,
            "Reduced_model_R2": fit.rsquared
        })

stability = pd.DataFrame(
    stability_rows
)

stability.to_csv(
    TABLES / "R2_original_model_coefficient_stability.csv",
    index=False
)


# ============================================================
# 7. FULL MODEL COEFFICIENT TABLE
# ============================================================

ci = m1.conf_int()

coef = pd.DataFrame({
    "Term": m1.params.index,
    "Beta": m1.params.values,
    "SE": m1.bse.values,
    "CI_low": ci[0].values,
    "CI_high": ci[1].values,
    "P_value": m1.pvalues.values
})

coef.to_csv(
    TABLES / "R2_original_model1_coefficients.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("ORIGINAL MODEL FIT")
print("=" * 90)
print(model_fit.to_string(index=False))


print("\n" + "=" * 90)
print("VIF")
print("=" * 90)
print(vif.to_string(index=False))


print("\n" + "=" * 90)
print("CONDITION INDICES")
print("=" * 90)
print(condition_table.to_string(index=False))


print("\n" + "=" * 90)
print("MODEL 1 COEFFICIENTS")
print("=" * 90)
print(coef.to_string(index=False))


print("\n" + "=" * 90)
print("10-FOLD CROSS-VALIDATION")
print("=" * 90)

print(cv_table.to_string(index=False))

print(
    f"\nMean CV R2: {cv_scores.mean():.6f}"
)

print(
    f"Median CV R2: {np.median(cv_scores):.6f}"
)

print(
    f"SD CV R2: {cv_scores.std(ddof=1):.6f}"
)


print("\n" + "=" * 90)
print("BOOTSTRAP SUMMARY")
print("=" * 90)

print(
    boot_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("COEFFICIENT STABILITY")
print("=" * 90)

print(
    stability.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"""
Key files:

{TABLES / 'R2_original_model_fit.csv'}
{TABLES / 'R2_original_model1_coefficients.csv'}
{TABLES / 'R2_original_model_VIF.csv'}
{TABLES / 'R2_original_model_condition_indices.csv'}
{TABLES / 'R2_original_model_10fold_CV_R2.csv'}
{TABLES / 'R2_original_model_bootstrap_summary.csv'}
{TABLES / 'R2_original_model_coefficient_stability.csv'}
"""
)
