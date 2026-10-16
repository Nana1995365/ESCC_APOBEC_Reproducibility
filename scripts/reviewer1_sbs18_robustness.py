#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results/signatures_fast/SBS96/Suggested_Solution/"
      "COSMIC_SBS96_Decomposed_Solution/Activities/"
      "COSMIC_SBS96_Activities.txt"
)

OUT = ROOT / "results/revision_R1"
TABLES = OUT / "tables"

TABLES.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT, sep="\t")

sbs = [
    c for c in df.columns
    if c.startswith("SBS")
]

for c in sbs:
    df[c] = pd.to_numeric(df[c], errors="raise")


# ============================================================
# DERIVED VARIABLES
# ============================================================

df["APOBEC"] = (
    df["SBS2"] +
    df["SBS13"]
)

df["TOTAL"] = (
    df[sbs].sum(axis=1)
)

df["NON_APOBEC"] = (
    df["TOTAL"] -
    df["APOBEC"]
)

# Burden excluding both APOBEC and SBS18.
# This avoids using a burden covariate containing either
# variable involved in the focal APOBEC-SBS18 relationship.

df["BACKGROUND"] = (
    df["TOTAL"]
    - df["APOBEC"]
    - df["SBS18"]
)

if (df["BACKGROUND"] < 0).any():
    raise ValueError("Negative background burden detected.")


# ============================================================
# HELPERS
# ============================================================

def corr(x, y, label):

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    return {
        "Analysis": label,
        "N": len(x),
        "Pearson_r": pr,
        "Pearson_P": pp,
        "Spearman_rho": sr,
        "Spearman_P": sp
    }


def save_model(model, name):

    ci = model.conf_int()

    tab = pd.DataFrame({
        "Term": model.params.index,
        "Beta": model.params.values,
        "SE": model.bse.values,
        "CI_low": ci[0].values,
        "CI_high": ci[1].values,
        "P_value": model.pvalues.values
    })

    tab.to_csv(
        TABLES / f"{name}_coefficients.csv",
        index=False
    )

    with open(
        TABLES / f"{name}_summary.txt",
        "w"
    ) as f:
        f.write(model.summary().as_text())

    return tab


# ============================================================
# 1. RAW ABSOLUTE-COUNT ASSOCIATION
# ============================================================

results = []

results.append(
    corr(
        df["APOBEC"],
        df["SBS18"],
        "Raw_absolute_counts"
    )
)


# ============================================================
# 2. LOG1P ABSOLUTE-COUNT ASSOCIATION
# ============================================================

df["log_APOBEC"] = np.log1p(df["APOBEC"])
df["log_SBS18"] = np.log1p(df["SBS18"])
df["log_BACKGROUND"] = np.log1p(df["BACKGROUND"])

results.append(
    corr(
        df["log_APOBEC"],
        df["log_SBS18"],
        "Log1p_absolute_counts"
    )
)


# ============================================================
# 3. SBS18-POSITIVE TUMORS ONLY
# ============================================================

positive = df[
    df["SBS18"] > 0
].copy()

results.append(
    corr(
        positive["APOBEC"],
        positive["SBS18"],
        "SBS18_positive_only_raw_counts"
    )
)

results.append(
    corr(
        np.log1p(positive["APOBEC"]),
        np.log1p(positive["SBS18"]),
        "SBS18_positive_only_log1p"
    )
)


# ============================================================
# 4. RESIDUAL/PARTIAL ASSOCIATION
#
# Remove association of APOBEC and SBS18 with background
# mutational burden, then correlate residuals.
# ============================================================

m_apobec = smf.ols(
    "log_APOBEC ~ log_BACKGROUND",
    data=df
).fit()

m_sbs18 = smf.ols(
    "log_SBS18 ~ log_BACKGROUND",
    data=df
).fit()

df["APOBEC_residual"] = m_apobec.resid
df["SBS18_residual"] = m_sbs18.resid

results.append(
    corr(
        df["APOBEC_residual"],
        df["SBS18_residual"],
        "Partial_association_adjusted_for_background_burden"
    )
)


# ============================================================
# 5. MULTIVARIABLE MODEL:
#    APOBEC ~ SBS18 + BACKGROUND BURDEN
# ============================================================

model_all = smf.ols(
    "log_APOBEC ~ log_SBS18 + log_BACKGROUND",
    data=df
).fit(cov_type="HC3")

all_table = save_model(
    model_all,
    "R1_SBS18_background_adjusted_model"
)


# ============================================================
# 6. SAME MODEL AMONG SBS18-POSITIVE TUMORS
# ============================================================

positive["log_APOBEC"] = np.log1p(
    positive["APOBEC"]
)

positive["log_SBS18"] = np.log1p(
    positive["SBS18"]
)

positive["log_BACKGROUND"] = np.log1p(
    positive["BACKGROUND"]
)

model_positive = smf.ols(
    "log_APOBEC ~ log_SBS18 + log_BACKGROUND",
    data=positive
).fit(cov_type="HC3")

positive_table = save_model(
    model_positive,
    "R1_SBS18_positive_background_adjusted_model"
)


# ============================================================
# 7. PRESENCE/ABSENCE ANALYSIS
#
# This checks whether the result is being driven mainly by
# tumors having no SBS18 activity at all.
# ============================================================

df["SBS18_present"] = (
    df["SBS18"] > 0
).astype(int)

present_summary = (
    df.groupby("SBS18_present")["APOBEC"]
      .agg(
          N="size",
          Median="median",
          Mean="mean"
      )
      .reset_index()
)

present_summary.to_csv(
    TABLES / "R1_SBS18_presence_APOBEC_summary.csv",
    index=False
)


# ============================================================
# 8. SAVE CORRELATION RESULTS
# ============================================================

correlation_table = pd.DataFrame(results)

correlation_table.to_csv(
    TABLES / "R1_SBS18_robustness_correlations.csv",
    index=False
)


# ============================================================
# 9. CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("SBS18 ROBUSTNESS ANALYSIS")
print("=" * 80)

print("\nSAMPLE COUNTS")
print("-" * 80)

print(f"All tumors: {len(df)}")
print(
    f"SBS18-positive tumors: "
    f"{(df['SBS18'] > 0).sum()}"
)
print(
    f"SBS18-zero tumors: "
    f"{(df['SBS18'] == 0).sum()}"
)


print("\nCORRELATION / PARTIAL ASSOCIATION RESULTS")
print("-" * 80)

print(
    correlation_table.to_string(
        index=False
    )
)


print("\nBACKGROUND-ADJUSTED MODEL — ALL TUMORS")
print("-" * 80)

print(
    all_table.to_string(
        index=False
    )
)

print(
    f"\nR-squared = "
    f"{model_all.rsquared:.4f}"
)


print("\nBACKGROUND-ADJUSTED MODEL — SBS18-POSITIVE ONLY")
print("-" * 80)

print(
    positive_table.to_string(
        index=False
    )
)

print(
    f"\nR-squared = "
    f"{model_positive.rsquared:.4f}"
)


print("\nSBS18 PRESENCE/ABSENCE SUMMARY")
print("-" * 80)

print(
    present_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"""
Results saved to:

{TABLES / 'R1_SBS18_robustness_correlations.csv'}

{TABLES / 'R1_SBS18_background_adjusted_model_coefficients.csv'}

{TABLES / 'R1_SBS18_positive_background_adjusted_model_coefficients.csv'}

{TABLES / 'R1_SBS18_presence_APOBEC_summary.csv'}
"""
)
