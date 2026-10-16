#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/APOBEC_PCA_with_country.csv"

OUT = ROOT / "results/revision_R1/geography_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

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

dat = df[required].dropna().copy()

# ============================================================
# EXCLUDE UK
# ============================================================

print("=" * 80)
print("ADJUSTED COUNTRY SENSITIVITY ANALYSIS — UK EXCLUDED")
print("=" * 80)

print("\nN before UK exclusion:", len(dat))

uk_n = (dat["Country"] == "UK").sum()
print("UK observations identified:", uk_n)

dat = dat.loc[dat["Country"] != "UK"].copy()

print("N after UK exclusion:", len(dat))

print("\nCountry sample sizes:")
print(dat["Country"].value_counts().sort_index().to_string())

# ============================================================
# SAME MODELS AS ORIGINAL ANALYSIS
# ============================================================

formula_reduced = (
    "APOBEC_prop ~ "
    "Total_SBS_burden + SBS1 + SBS3 + SBS5 + SBS18"
)

formula_country = (
    'APOBEC_prop ~ '
    'Total_SBS_burden + SBS1 + SBS3 + SBS5 + SBS18 + '
    'C(Country, Treatment(reference="Brazil"))'
)

m0 = smf.ols(formula_reduced, data=dat).fit()
m1 = smf.ols(formula_country, data=dat).fit()

# ============================================================
# COUNTRY COEFFICIENTS
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

country_coef = coef[
    coef["Term"].str.contains("Country")
].copy()

country_coef.to_csv(
    OUT / "adjusted_country_coefficients_excluding_UK.csv",
    index=False
)

# ============================================================
# NESTED F TEST
# ============================================================

comparison = anova_lm(m0, m1)

F, p, df_diff = m1.compare_f_test(m0)

global_test = pd.DataFrame([{
    "Test": "Joint_country_effect_nested_F_test_excluding_UK",
    "Reference_country": "Brazil",
    "N": len(dat),
    "F_statistic": F,
    "df_difference": df_diff,
    "P_value": p,
    "Reduced_R2": m0.rsquared,
    "Country_model_R2": m1.rsquared,
    "Delta_R2": m1.rsquared - m0.rsquared,
    "Reduced_adjusted_R2": m0.rsquared_adj,
    "Country_adjusted_R2": m1.rsquared_adj
}])

global_test.to_csv(
    OUT / "adjusted_country_global_test_excluding_UK.csv",
    index=False
)

comparison.to_csv(
    OUT / "adjusted_country_nested_F_test_excluding_UK.csv"
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COUNTRY COEFFICIENTS — BRAZIL REFERENCE")
print("=" * 80)

print(country_coef.to_string(index=False))

print("\n" + "=" * 80)
print("GLOBAL COUNTRY TEST — UK EXCLUDED")
print("=" * 80)

print(global_test.to_string(index=False))

print("\nNested-model ANOVA:")
print(comparison.to_string())

print("\n" + "=" * 80)
print("KEY VALUES")
print("=" * 80)
print(f"N = {len(dat)}")
print(f"F = {F:.6f}")
print(f"df_difference = {int(df_diff)}")
print(f"residual_df_full = {int(m1.df_resid)}")
print(f"P = {p:.12g}")
print(f"Reduced R2 = {m0.rsquared:.6f}")
print(f"Country model R2 = {m1.rsquared:.6f}")
print(f"Delta R2 = {m1.rsquared - m0.rsquared:.6f}")

print("\nSaved results to:", OUT)
