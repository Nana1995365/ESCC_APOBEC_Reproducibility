#!/usr/bin/env python3

"""
Reviewer 1 - Major Concern 1
Compositional-data sensitivity analysis of APOBEC-associated
mutagenesis in 552 ESCC tumors.

Analyses:
1. Absolute COSMIC signature counts
2. Original proportional analysis
3. Count-versus-proportion comparison
4. CLR/log-ratio compositional sensitivity analysis
5. Focused SBS18/SBS3/SBS5 comparison
6. Count-based multivariable regression
7. Regression diagnostics (VIF)
8. Publication-quality sensitivity figures

Original manuscript outputs are NOT overwritten.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ACTIVITY_FILE = (
    ROOT
    / "results"
    / "signatures_fast"
    / "SBS96"
    / "Suggested_Solution"
    / "COSMIC_SBS96_Decomposed_Solution"
    / "Activities"
    / "COSMIC_SBS96_Activities.txt"
)

OUT = ROOT / "results" / "revision_R1"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("REVIEWER 1: COMPOSITIONAL SENSITIVITY ANALYSIS")
print("=" * 80)

print(f"\nInput file:\n{ACTIVITY_FILE}")
print(f"\nOutput directory:\n{OUT}")


# ============================================================
# 2. LOAD COSMIC ACTIVITY MATRIX
# ============================================================

if not ACTIVITY_FILE.exists():
    raise FileNotFoundError(
        f"Activity file not found:\n{ACTIVITY_FILE}"
    )

df = pd.read_csv(ACTIVITY_FILE, sep="\t")

signature_cols = [
    c for c in df.columns
    if c.startswith("SBS")
]

required = [
    "SBS2",
    "SBS13",
    "SBS18",
    "SBS3",
    "SBS5"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required signature columns: {missing}"
    )

print("\nDATASET")
print("-" * 80)
print(f"Number of tumors: {len(df)}")
print(f"Number of COSMIC signatures: {len(signature_cols)}")
print(f"Signatures: {signature_cols}")

if len(df) != 552:
    print(
        f"\nWARNING: Manuscript reports 552 tumors, "
        f"but input contains {len(df)}."
    )


# ============================================================
# 3. NUMERIC VALIDATION
# ============================================================

for col in signature_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

if df[signature_cols].isna().any().any():
    bad = df[
        df[signature_cols].isna().any(axis=1)
    ]

    raise ValueError(
        "Missing/non-numeric signature activities detected.\n"
        f"Affected rows:\n{bad.head()}"
    )

if (df[signature_cols] < 0).any().any():
    raise ValueError(
        "Negative COSMIC signature activities detected."
    )


# ============================================================
# 4. DERIVED VARIABLES
# ============================================================

df["APOBEC_count"] = (
    df["SBS2"] +
    df["SBS13"]
)

df["Total_SBS_count"] = (
    df[signature_cols]
    .sum(axis=1)
)

if (df["Total_SBS_count"] <= 0).any():
    raise ValueError(
        "At least one tumor has zero total SBS burden."
    )

df["APOBEC_prop"] = (
    df["APOBEC_count"] /
    df["Total_SBS_count"]
)

df["Non_APOBEC_count"] = (
    df["Total_SBS_count"] -
    df["APOBEC_count"]
)

for sig in signature_cols:
    df[f"{sig}_prop"] = (
        df[sig] /
        df["Total_SBS_count"]
    )


# ============================================================
# 5. QC SUMMARY
# ============================================================

qc_rows = []

qc_rows.append(
    ["Number_of_tumors", len(df)]
)

qc_rows.append(
    ["Number_of_signatures", len(signature_cols)]
)

qc_rows.append(
    ["Median_total_SBS", df["Total_SBS_count"].median()]
)

qc_rows.append(
    ["Mean_total_SBS", df["Total_SBS_count"].mean()]
)

qc_rows.append(
    ["Median_APOBEC_count", df["APOBEC_count"].median()]
)

qc_rows.append(
    ["Mean_APOBEC_count", df["APOBEC_count"].mean()]
)

qc_rows.append(
    ["Median_APOBEC_proportion", df["APOBEC_prop"].median()]
)

qc_rows.append(
    ["Mean_APOBEC_proportion", df["APOBEC_prop"].mean()]
)

qc_rows.append(
    ["Zero_APOBEC_tumors", int((df["APOBEC_count"] == 0).sum())]
)

for sig in ["SBS18", "SBS3", "SBS5"]:
    qc_rows.append(
        [
            f"Zero_{sig}_tumors",
            int((df[sig] == 0).sum())
        ]
    )

qc = pd.DataFrame(
    qc_rows,
    columns=["Variable", "Value"]
)

qc.to_csv(
    TABLES / "R1_QC_summary.csv",
    index=False
)

print("\nQC SUMMARY")
print("-" * 80)
print(qc.to_string(index=False))


# ============================================================
# 6. CORRELATION FUNCTION
# ============================================================

def correlation_stats(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "N": len(x),
            "Pearson_r": np.nan,
            "Pearson_P": np.nan,
            "Spearman_rho": np.nan,
            "Spearman_P": np.nan
        }

    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "N": len(x),
            "Pearson_r": np.nan,
            "Pearson_P": np.nan,
            "Spearman_rho": np.nan,
            "Spearman_P": np.nan
        }

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    return {
        "N": len(x),
        "Pearson_r": pr,
        "Pearson_P": pp,
        "Spearman_rho": sr,
        "Spearman_P": sp
    }


# ============================================================
# 7. ABSOLUTE-COUNT ASSOCIATIONS
# ============================================================

print("\n" + "=" * 80)
print("ABSOLUTE-COUNT ASSOCIATIONS")
print("=" * 80)

comparison_signatures = [
    sig for sig in signature_cols
    if sig not in ["SBS2", "SBS13"]
]

absolute_rows = []

for sig in comparison_signatures:

    stats = correlation_stats(
        df["APOBEC_count"],
        df[sig]
    )

    absolute_rows.append({
        "Signature": sig,
        **stats
    })

absolute_results = pd.DataFrame(
    absolute_rows
)

absolute_results["Pearson_FDR"] = multipletests(
    absolute_results["Pearson_P"].fillna(1.0),
    method="fdr_bh"
)[1]

absolute_results["Spearman_FDR"] = multipletests(
    absolute_results["Spearman_P"].fillna(1.0),
    method="fdr_bh"
)[1]

absolute_results = absolute_results.sort_values(
    ["Spearman_P", "Signature"]
)

absolute_results.to_csv(
    TABLES / "R1_absolute_count_correlations.csv",
    index=False
)

print(
    absolute_results[
        [
            "Signature",
            "Pearson_r",
            "Pearson_P",
            "Pearson_FDR",
            "Spearman_rho",
            "Spearman_P",
            "Spearman_FDR"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. ORIGINAL PROPORTIONAL ASSOCIATIONS
# ============================================================

print("\n" + "=" * 80)
print("PROPORTIONAL ASSOCIATIONS")
print("=" * 80)

proportion_rows = []

for sig in comparison_signatures:

    stats = correlation_stats(
        df["APOBEC_prop"],
        df[f"{sig}_prop"]
    )

    proportion_rows.append({
        "Signature": sig,
        **stats
    })

proportion_results = pd.DataFrame(
    proportion_rows
)

proportion_results["Pearson_FDR"] = multipletests(
    proportion_results["Pearson_P"].fillna(1.0),
    method="fdr_bh"
)[1]

proportion_results["Spearman_FDR"] = multipletests(
    proportion_results["Spearman_P"].fillna(1.0),
    method="fdr_bh"
)[1]

proportion_results = proportion_results.sort_values(
    ["Spearman_P", "Signature"]
)

proportion_results.to_csv(
    TABLES / "R1_proportional_correlations.csv",
    index=False
)

print(
    proportion_results[
        [
            "Signature",
            "Pearson_r",
            "Pearson_P",
            "Pearson_FDR",
            "Spearman_rho",
            "Spearman_P",
            "Spearman_FDR"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. COUNT VS PROPORTION COMPARISON
# ============================================================

comparison = absolute_results.merge(
    proportion_results,
    on="Signature",
    suffixes=("_count", "_prop")
)

comparison["Direction_count"] = np.select(
    [
        comparison["Spearman_rho_count"] > 0,
        comparison["Spearman_rho_count"] < 0
    ],
    [
        "positive",
        "negative"
    ],
    default="zero"
)

comparison["Direction_prop"] = np.select(
    [
        comparison["Spearman_rho_prop"] > 0,
        comparison["Spearman_rho_prop"] < 0
    ],
    [
        "positive",
        "negative"
    ],
    default="zero"
)

comparison["Direction_changed"] = (
    comparison["Direction_count"] !=
    comparison["Direction_prop"]
)

comparison.to_csv(
    TABLES / "R1_count_vs_proportion_comparison.csv",
    index=False
)

print("\nCOUNT VS PROPORTION")
print("-" * 80)

print(
    comparison[
        [
            "Signature",
            "Spearman_rho_count",
            "Spearman_FDR_count",
            "Spearman_rho_prop",
            "Spearman_FDR_prop",
            "Direction_changed"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. CLR COMPOSITIONAL ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CLR / LOG-RATIO SENSITIVITY ANALYSIS")
print("=" * 80)

X = df[signature_cols].astype(float).copy()

# ------------------------------------------------------------
# Zero replacement
#
# CLR requires strictly positive components.
#
# For each tumor, zero components are replaced by half of the
# smallest non-zero signature activity observed in that tumor.
#
# This replacement is used ONLY for the compositional
# sensitivity analysis. Original absolute counts remain
# unchanged for all count-based analyses.
# ------------------------------------------------------------

X_replaced = X.copy()

replacement_values = []

for idx in X_replaced.index:

    row = X_replaced.loc[idx].to_numpy(
        dtype=float
    )

    positive = row[row > 0]

    if len(positive) == 0:
        raise ValueError(
            f"Sample {df.loc[idx, 'Samples']} "
            "has no positive signature activity."
        )

    delta = positive.min() / 2.0

    row[row == 0] = delta

    X_replaced.loc[idx] = row

    replacement_values.append(delta)

df["CLR_zero_replacement"] = replacement_values


# ============================================================
# 11. CLR TRANSFORMATION
# ============================================================

composition = X_replaced.div(
    X_replaced.sum(axis=1),
    axis=0
)

log_composition = np.log(
    composition
)

clr_matrix = log_composition.sub(
    log_composition.mean(axis=1),
    axis=0
)

clr_matrix.columns = [
    f"CLR_{c}"
    for c in signature_cols
]

df = pd.concat(
    [
        df,
        clr_matrix
    ],
    axis=1
)


# ============================================================
# 12. APOBEC LOG-RATIO
# ============================================================

non_apobec_signatures = [
    sig for sig in signature_cols
    if sig not in ["SBS2", "SBS13"]
]

apobec_replaced = (
    X_replaced["SBS2"] +
    X_replaced["SBS13"]
)

non_apobec_matrix = (
    X_replaced[
        non_apobec_signatures
    ]
)

geometric_mean_non_apobec = np.exp(
    np.log(
        non_apobec_matrix
    ).mean(axis=1)
)

df["APOBEC_logratio"] = np.log(
    apobec_replaced /
    geometric_mean_non_apobec
)


# ============================================================
# 13. CLR ASSOCIATIONS
# ============================================================

clr_rows = []

for sig in non_apobec_signatures:

    stats = correlation_stats(
        df["APOBEC_logratio"],
        df[f"CLR_{sig}"]
    )

    clr_rows.append({
        "Signature": sig,
        **stats
    })

clr_results = pd.DataFrame(
    clr_rows
)

clr_results["Pearson_FDR"] = multipletests(
    clr_results["Pearson_P"].fillna(1.0),
    method="fdr_bh"
)[1]

clr_results["Spearman_FDR"] = multipletests(
    clr_results["Spearman_P"].fillna(1.0),
    method="fdr_bh"
)[1]

clr_results = clr_results.sort_values(
    ["Spearman_P", "Signature"]
)

clr_results.to_csv(
    TABLES / "R1_CLR_correlations.csv",
    index=False
)

print(
    clr_results[
        [
            "Signature",
            "Pearson_r",
            "Pearson_P",
            "Pearson_FDR",
            "Spearman_rho",
            "Spearman_P",
            "Spearman_FDR"
        ]
    ].to_string(index=False)
)


# ============================================================
# 14. KEY SBS18/SBS3/SBS5 SUMMARY
# ============================================================

key_signatures = [
    "SBS18",
    "SBS3",
    "SBS5"
]

summary_rows = []

for sig in key_signatures:

    abs_row = absolute_results.loc[
        absolute_results["Signature"] == sig
    ].iloc[0]

    prop_row = proportion_results.loc[
        proportion_results["Signature"] == sig
    ].iloc[0]

    clr_row = clr_results.loc[
        clr_results["Signature"] == sig
    ].iloc[0]

    summary_rows.append({

        "Signature": sig,

        "Count_Pearson_r":
            abs_row["Pearson_r"],

        "Count_Pearson_P":
            abs_row["Pearson_P"],

        "Count_Pearson_FDR":
            abs_row["Pearson_FDR"],

        "Count_Spearman_rho":
            abs_row["Spearman_rho"],

        "Count_Spearman_P":
            abs_row["Spearman_P"],

        "Count_Spearman_FDR":
            abs_row["Spearman_FDR"],

        "Proportion_Pearson_r":
            prop_row["Pearson_r"],

        "Proportion_Pearson_P":
            prop_row["Pearson_P"],

        "Proportion_Pearson_FDR":
            prop_row["Pearson_FDR"],

        "Proportion_Spearman_rho":
            prop_row["Spearman_rho"],

        "Proportion_Spearman_P":
            prop_row["Spearman_P"],

        "Proportion_Spearman_FDR":
            prop_row["Spearman_FDR"],

        "CLR_Pearson_r":
            clr_row["Pearson_r"],

        "CLR_Pearson_P":
            clr_row["Pearson_P"],

        "CLR_Pearson_FDR":
            clr_row["Pearson_FDR"],

        "CLR_Spearman_rho":
            clr_row["Spearman_rho"],

        "CLR_Spearman_P":
            clr_row["Spearman_P"],

        "CLR_Spearman_FDR":
            clr_row["Spearman_FDR"]
    })

key_summary = pd.DataFrame(
    summary_rows
)

key_summary.to_csv(
    TABLES / "R1_key_signature_sensitivity_summary.csv",
    index=False
)

print("\nKEY SIGNATURE SENSITIVITY SUMMARY")
print("-" * 80)
print(key_summary.to_string(index=False))


# ============================================================
# 15. COUNT-BASED MULTIVARIABLE MODEL
# ============================================================

print("\n" + "=" * 80)
print("COUNT-BASED MULTIVARIABLE MODEL")
print("=" * 80)

# Counts are strongly right-skewed, so log1p transformation
# is used as a sensitivity analysis.

reg = pd.DataFrame({

    "log_APOBEC":
        np.log1p(
            df["APOBEC_count"]
        ),

    "log_Total_SBS":
        np.log1p(
            df["Total_SBS_count"]
        ),

    "log_SBS18":
        np.log1p(
            df["SBS18"]
        ),

    "log_SBS3":
        np.log1p(
            df["SBS3"]
        ),

    "log_SBS5":
        np.log1p(
            df["SBS5"]
        )
})

model = smf.ols(
    formula=(
        "log_APOBEC ~ "
        "log_Total_SBS + "
        "log_SBS18 + "
        "log_SBS3 + "
        "log_SBS5"
    ),
    data=reg
).fit(
    cov_type="HC3"
)

print(model.summary())

with open(
    TABLES /
    "R1_absolute_count_regression_summary.txt",
    "w"
) as handle:

    handle.write(
        model.summary().as_text()
    )

ci = model.conf_int()

regression_table = pd.DataFrame({

    "Term":
        model.params.index,

    "Beta":
        model.params.values,

    "SE_HC3":
        model.bse.values,

    "CI_low":
        ci[0].values,

    "CI_high":
        ci[1].values,

    "P_value":
        model.pvalues.values
})

regression_table.to_csv(
    TABLES /
    "R1_absolute_count_regression_coefficients.csv",
    index=False
)


# ============================================================
# 16. VIF DIAGNOSTICS
# ============================================================

predictor_names = [
    "log_Total_SBS",
    "log_SBS18",
    "log_SBS3",
    "log_SBS5"
]

X_vif = sm.add_constant(
    reg[predictor_names]
)

vif_rows = []

for i, col in enumerate(
    X_vif.columns
):

    vif_rows.append({
        "Variable": col,
        "VIF": variance_inflation_factor(
            X_vif.values,
            i
        )
    })

vif_table = pd.DataFrame(
    vif_rows
)

vif_table.to_csv(
    TABLES /
    "R1_absolute_count_model_VIF.csv",
    index=False
)

print("\nVIF")
print("-" * 80)
print(vif_table.to_string(index=False))


# ============================================================
# 17. SIMPLE TOTAL-BURDEN ASSOCIATION
# ============================================================

total_stats = correlation_stats(
    df["APOBEC_count"],
    df["Total_SBS_count"]
)

total_table = pd.DataFrame(
    [{
        "Comparison":
            "APOBEC_count_vs_Total_SBS_count",

        **total_stats
    }]
)

total_table.to_csv(
    TABLES /
    "R1_APOBEC_count_total_burden_correlation.csv",
    index=False
)

print("\nAPOBEC COUNT VS TOTAL SBS BURDEN")
print("-" * 80)
print(total_table.to_string(index=False))


# ============================================================
# 18. INTERPRETATION FLAGS
# ============================================================

interpretation_rows = []

for sig in key_signatures:

    abs_row = absolute_results.loc[
        absolute_results["Signature"] == sig
    ].iloc[0]

    prop_row = proportion_results.loc[
        proportion_results["Signature"] == sig
    ].iloc[0]

    clr_row = clr_results.loc[
        clr_results["Signature"] == sig
    ].iloc[0]

    abs_direction = (
        "positive"
        if abs_row["Spearman_rho"] > 0
        else "negative"
        if abs_row["Spearman_rho"] < 0
        else "zero"
    )

    prop_direction = (
        "positive"
        if prop_row["Spearman_rho"] > 0
        else "negative"
        if prop_row["Spearman_rho"] < 0
        else "zero"
    )

    clr_direction = (
        "positive"
        if clr_row["Spearman_rho"] > 0
        else "negative"
        if clr_row["Spearman_rho"] < 0
        else "zero"
    )

    interpretation_rows.append({

        "Signature": sig,

        "Absolute_direction":
            abs_direction,

        "Absolute_FDR_significant":
            bool(
                abs_row["Spearman_FDR"] < 0.05
            ),

        "Proportion_direction":
            prop_direction,

        "Proportion_FDR_significant":
            bool(
                prop_row["Spearman_FDR"] < 0.05
            ),

        "CLR_direction":
            clr_direction,

        "CLR_FDR_significant":
            bool(
                clr_row["Spearman_FDR"] < 0.05
            ),

        "Count_vs_proportion_direction_agree":
            abs_direction == prop_direction,

        "Count_vs_CLR_direction_agree":
            abs_direction == clr_direction
    })

interpretation_table = pd.DataFrame(
    interpretation_rows
)

interpretation_table.to_csv(
    TABLES /
    "R1_interpretation_flags.csv",
    index=False
)

print("\nINTERPRETATION FLAGS")
print("-" * 80)
print(
    interpretation_table.to_string(
        index=False
    )
)


# ============================================================
# 19. FIGURE FUNCTION
# ============================================================

def save_scatter(
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename,
    correlation_x=None,
    correlation_y=None
):

    if correlation_x is None:
        correlation_x = x

    if correlation_y is None:
        correlation_y = y

    rho, p = spearmanr(
        correlation_x,
        correlation_y
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    ax.scatter(
        x,
        y,
        s=30,
        alpha=0.55,
        edgecolors="none"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=11
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11
    )

    ax.set_title(
        title,
        fontsize=12
    )

    annotation = (
        f"Spearman ρ = {rho:.3f}\n"
        f"P = {p:.2e}"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.7"
        )
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{filename}.png",
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURES / f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 20. SBS18 SENSITIVITY FIGURES
# ============================================================

save_scatter(
    x=np.log1p(
        df["APOBEC_count"]
    ),
    y=np.log1p(
        df["SBS18"]
    ),
    xlabel="log(1 + APOBEC mutation count)",
    ylabel="log(1 + SBS18 mutation count)",
    title="Absolute APOBEC and SBS18 mutation counts",
    filename="R1_APOBEC_count_vs_SBS18_count",
    correlation_x=df["APOBEC_count"],
    correlation_y=df["SBS18"]
)

save_scatter(
    x=df["APOBEC_prop"],
    y=df["SBS18_prop"],
    xlabel="APOBEC proportion",
    ylabel="SBS18 proportion",
    title="Proportional APOBEC and SBS18 activities",
    filename="R1_APOBEC_prop_vs_SBS18_prop"
)

save_scatter(
    x=df["APOBEC_logratio"],
    y=df["CLR_SBS18"],
    xlabel="APOBEC log-ratio",
    ylabel="CLR-transformed SBS18",
    title="APOBEC-SBS18 compositional sensitivity",
    filename="R1_APOBEC_logratio_vs_CLR_SBS18"
)


# ============================================================
# 21. SBS3 AND SBS5 SENSITIVITY FIGURES
# ============================================================

for sig in ["SBS3", "SBS5"]:

    save_scatter(
        x=np.log1p(
            df["APOBEC_count"]
        ),
        y=np.log1p(
            df[sig]
        ),
        xlabel="log(1 + APOBEC mutation count)",
        ylabel=f"log(1 + {sig} mutation count)",
        title=f"Absolute APOBEC and {sig} mutation counts",
        filename=f"R1_APOBEC_count_vs_{sig}_count",
        correlation_x=df["APOBEC_count"],
        correlation_y=df[sig]
    )


# ============================================================
# 22. SAVE COMPLETE ANALYTIC DATASET
# ============================================================

df.to_csv(
    OUT /
    "R1_compositional_sensitivity_full_data.csv",
    index=False
)


# ============================================================
# 23. WRITE ANALYSIS README
# ============================================================

readme = f"""
Reviewer 1 Major Concern 1
Compositional sensitivity analysis

Input:
{ACTIVITY_FILE}

N tumors:
{len(df)}

COSMIC SBS signatures:
{", ".join(signature_cols)}

APOBEC absolute activity:
SBS2 + SBS13

APOBEC proportional activity:
(SBS2 + SBS13) / total SBS activity

Analyses performed:
1. Absolute-count correlations
2. Proportional correlations
3. Benjamini-Hochberg FDR correction
4. CLR/log-ratio sensitivity analysis
5. Focused SBS18, SBS3 and SBS5 analysis
6. Log-transformed absolute-count multivariable regression
7. VIF diagnostics

Zero handling for CLR:
Within each tumor, zero signature activities were replaced
by one-half of that tumor's smallest observed non-zero
signature activity. This replacement was used only for
the CLR/log-ratio sensitivity analysis.

Important:
The original COSMIC activity counts were not altered for
absolute-count analyses.

No original manuscript result files were overwritten.
"""

with open(
    OUT / "README_R1_analysis.txt",
    "w"
) as handle:

    handle.write(readme)


# ============================================================
# 24. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"""
Outputs saved in:

{OUT}

KEY FILES

{TABLES / 'R1_key_signature_sensitivity_summary.csv'}

{TABLES / 'R1_interpretation_flags.csv'}

{TABLES / 'R1_absolute_count_correlations.csv'}

{TABLES / 'R1_proportional_correlations.csv'}

{TABLES / 'R1_CLR_correlations.csv'}

{TABLES / 'R1_count_vs_proportion_comparison.csv'}

{TABLES / 'R1_absolute_count_regression_coefficients.csv'}

{TABLES / 'R1_absolute_count_model_VIF.csv'}

IMPORTANT:
Do not revise the biological interpretation of SBS18,
SBS3 or SBS5 until the count, proportional and
compositional sensitivity results have been compared.
"""
)
