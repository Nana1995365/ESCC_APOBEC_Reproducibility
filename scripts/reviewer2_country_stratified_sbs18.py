#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/APOBEC_PCA_with_country.csv"

OUT = ROOT / "results/revision_R1"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = [
    "Country",
    "SBS2",
    "SBS13",
    "SBS18"
]

missing = [x for x in required if x not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["APOBEC_count"] = (
    pd.to_numeric(df["SBS2"], errors="coerce") +
    pd.to_numeric(df["SBS13"], errors="coerce")
)

df["SBS18"] = pd.to_numeric(
    df["SBS18"],
    errors="coerce"
)

dat = df[
    ["Country", "APOBEC_count", "SBS18"]
].dropna().copy()


def fisher_ci(r, n):

    if n <= 3 or abs(r) >= 1:
        return np.nan, np.nan

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)

    low = np.tanh(z - 1.96 * se)
    high = np.tanh(z + 1.96 * se)

    return low, high


rows = []

for country, g in dat.groupby("Country"):

    n = len(g)

    if (
        n >= 3 and
        g["APOBEC_count"].nunique() > 1 and
        g["SBS18"].nunique() > 1
    ):

        pr, pp = pearsonr(
            g["APOBEC_count"],
            g["SBS18"]
        )

        sr, sp = spearmanr(
            g["APOBEC_count"],
            g["SBS18"]
        )

        p_low, p_high = fisher_ci(pr, n)
        s_low, s_high = fisher_ci(sr, n)

    else:

        pr = pp = sr = sp = np.nan
        p_low = p_high = s_low = s_high = np.nan

    rows.append({
        "Country": country,
        "N": n,
        "SBS18_zero_N": int((g["SBS18"] == 0).sum()),
        "SBS18_zero_percent": (
            (g["SBS18"] == 0).mean() * 100
        ),
        "Pearson_r": pr,
        "Pearson_CI_low": p_low,
        "Pearson_CI_high": p_high,
        "Pearson_P": pp,
        "Spearman_rho": sr,
        "Spearman_CI_low": s_low,
        "Spearman_CI_high": s_high,
        "Spearman_P": sp,
        "Median_APOBEC_count": g["APOBEC_count"].median(),
        "Median_SBS18_count": g["SBS18"].median()
    })

res = pd.DataFrame(rows).sort_values(
    "Country"
)

res.to_csv(
    TABLES /
    "R2_country_stratified_APOBEC_SBS18_correlations.csv",
    index=False
)

# ============================================================
# Direction consistency
# ============================================================

valid = res.dropna(
    subset=["Spearman_rho"]
)

direction = pd.DataFrame([{
    "Countries_evaluable":
        len(valid),

    "Countries_negative_Spearman":
        int((valid["Spearman_rho"] < 0).sum()),

    "Countries_positive_Spearman":
        int((valid["Spearman_rho"] > 0).sum()),

    "Countries_Spearman_P_lt_0_05":
        int((valid["Spearman_P"] < 0.05).sum())
}])

direction.to_csv(
    TABLES /
    "R2_country_stratified_APOBEC_SBS18_direction_summary.csv",
    index=False
)

print("=" * 100)
print("COUNTRY-STRATIFIED APOBEC–SBS18 ASSOCIATIONS")
print("=" * 100)

print(
    res.to_string(index=False)
)

print("\n" + "=" * 100)
print("DIRECTION SUMMARY")
print("=" * 100)

print(
    direction.to_string(index=False)
)

print("\nSaved:")
print(
    TABLES /
    "R2_country_stratified_APOBEC_SBS18_correlations.csv"
)
