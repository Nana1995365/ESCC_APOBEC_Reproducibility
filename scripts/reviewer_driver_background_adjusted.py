#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "results/APOBEC_driver_merged.csv"
OUT = ROOT / "results/revision_R1/tables"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

print("=" * 100)
print("COLUMNS")
print("=" * 100)
print(df.columns.tolist())

# ------------------------------------------------------------
# Identify driver columns
# ------------------------------------------------------------

genes = [
    "TP53",
    "NOTCH1",
    "PIK3CA",
    "NFE2L2",
    "CDKN2A",
    "FAT1",
    "KMT2D",
    "ZNF750",
    "AJUBA",
    "CUL3",
    "KEAP1",
]

# ------------------------------------------------------------
# Check required mutation-burden variables
# ------------------------------------------------------------

required = ["APOBEC_burden", "Total_SBS_burden"]

for c in required:
    if c not in df.columns:
        raise ValueError(
            f"Required column {c!r} not found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

# Non-APOBEC/background mutation burden
df["BACKGROUND_burden"] = (
    df["Total_SBS_burden"] - df["APOBEC_burden"]
)

if (df["BACKGROUND_burden"] < 0).any():
    raise ValueError("Negative background burden detected.")

df["log_APOBEC"] = np.log1p(df["APOBEC_burden"])
df["log_BACKGROUND"] = np.log1p(df["BACKGROUND_burden"])

# ------------------------------------------------------------
# Work out mutation indicator columns
# ------------------------------------------------------------

def find_gene_column(gene):
    candidates = [
        gene,
        f"{gene}_mut",
        f"{gene}_mutation",
        f"{gene}_status",
        f"mut_{gene}",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def convert_to_binary(series):
    """
    Convert common driver encodings to 0/1.
    """
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())

        if vals.issubset({0, 1}):
            return series.astype(float)

    s = series.astype(str).str.strip().str.lower()

    mutant = {
        "1", "true", "yes", "mut", "mutant",
        "mutated", "altered", "mutation"
    }

    wild = {
        "0", "false", "no", "wt", "wildtype",
        "wild-type", "wild type", "none", "nan"
    }

    out = pd.Series(np.nan, index=series.index)

    out[s.isin(mutant)] = 1
    out[s.isin(wild)] = 0

    return out


results = []

print("\n" + "=" * 100)
print("BACKGROUND-ADJUSTED DRIVER MODELS")
print("Model: log1p(APOBEC burden) ~ mutation status + log1p(non-APOBEC burden)")
print("HC3 robust standard errors")
print("=" * 100)

for gene in genes:

    col = find_gene_column(gene)

    if col is None:
        print(f"\nWARNING: no mutation column found for {gene}")
        continue

    tmp = df[
        [
            "log_APOBEC",
            "log_BACKGROUND",
            col
        ]
    ].copy()

    tmp["MUT"] = convert_to_binary(tmp[col])
    tmp = tmp.dropna()

    n = len(tmp)
    n_mut = int(tmp["MUT"].sum())
    n_wt = int(n - n_mut)

    if n_mut == 0 or n_wt == 0:
        print(f"\n{gene}: insufficient groups")
        continue

    model = smf.ols(
        "log_APOBEC ~ MUT + log_BACKGROUND",
        data=tmp
    ).fit(cov_type="HC3")

    beta = model.params["MUT"]
    se = model.bse["MUT"]
    p = model.pvalues["MUT"]

    ci = model.conf_int().loc["MUT"]
    ci_low = ci.iloc[0]
    ci_high = ci.iloc[1]

    # Multiplicative interpretation on (APOBEC + 1)
    ratio = np.exp(beta)
    ratio_low = np.exp(ci_low)
    ratio_high = np.exp(ci_high)

    results.append({
        "Gene": gene,
        "N": n,
        "Mutant_N": n_mut,
        "Wildtype_N": n_wt,
        "Beta_log_APOBEC": beta,
        "HC3_SE": se,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "P_value": p,
        "Exp_beta": ratio,
        "Exp_CI_low": ratio_low,
        "Exp_CI_high": ratio_high,
        "Model_R2": model.rsquared,
        "Adjusted_R2": model.rsquared_adj
    })

    print(f"\n{gene}")
    print("-" * 60)
    print(f"N = {n}; mutant = {n_mut}; WT = {n_wt}")
    print(
        f"Mutation beta = {beta:.6f} "
        f"(95% CI {ci_low:.6f} to {ci_high:.6f})"
    )
    print(f"P = {p:.6g}")
    print(
        f"exp(beta) = {ratio:.3f} "
        f"(95% CI {ratio_low:.3f} to {ratio_high:.3f})"
    )
    print(f"R2 = {model.rsquared:.4f}")


res = pd.DataFrame(results)

# ------------------------------------------------------------
# FDR correction across the same 11-gene testing family
# ------------------------------------------------------------

if len(res):

    res["BH_FDR"] = multipletests(
        res["P_value"],
        method="fdr_bh"
    )[1]

    res = res.sort_values(
        ["BH_FDR", "P_value"]
    ).reset_index(drop=True)

    outfile = (
        OUT /
        "R1_driver_background_adjusted_models.csv"
    )

    res.to_csv(outfile, index=False)

    print("\n" + "=" * 100)
    print("FINAL RESULTS WITH BH-FDR")
    print("=" * 100)

    print(
        res[
            [
                "Gene",
                "Mutant_N",
                "Beta_log_APOBEC",
                "CI_low",
                "CI_high",
                "P_value",
                "BH_FDR",
                "Exp_beta",
                "Exp_CI_low",
                "Exp_CI_high"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(outfile)

    # --------------------------------------------------------
    # Focus on the three genes significant in unadjusted
    # absolute-count analysis
    # --------------------------------------------------------

    focus = res[
        res["Gene"].isin(
            ["ZNF750", "KMT2D", "PIK3CA"]
        )
    ].copy()

    focusfile = (
        OUT /
        "R1_driver_background_adjusted_focus.csv"
    )

    focus.to_csv(focusfile, index=False)

    print("\n" + "=" * 100)
    print("FOCUS GENES")
    print("=" * 100)

    print(
        focus[
            [
                "Gene",
                "Mutant_N",
                "Beta_log_APOBEC",
                "CI_low",
                "CI_high",
                "P_value",
                "BH_FDR",
                "Exp_beta",
                "Exp_CI_low",
                "Exp_CI_high"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(focusfile)

