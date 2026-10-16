#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/APOBEC_PCA_with_country.csv"

OUT = ROOT / "results/revision_R1"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

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

# Explicitly show country levels
countries = sorted(dat["Country"].unique())

print("=" * 80)
print("COUNTRY MODEL DIAGNOSTICS")
print("=" * 80)

print("\nCountry sample sizes:")
print(dat["Country"].value_counts().sort_index().to_string())

print("\nAlphabetical country levels:")
print(countries)

# Use Brazil explicitly as reference so coding is transparent.
# The scientific conclusion of the global country test does not
# depend on which country is chosen as the reference category.

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
    TABLES / "R2_country_coefficients_Brazil_reference.csv",
    index=False
)

# ============================================================
# NESTED F TEST:
# Does adding country as a whole improve model fit?
# ============================================================

comparison = anova_lm(
    m0,
    m1
)

comparison.to_csv(
    TABLES / "R2_country_global_nested_F_test.csv"
)

# Direct compare_f_test as cross-check
F, p, df_diff = m1.compare_f_test(m0)

global_test = pd.DataFrame([{
    "Test": "Joint_country_effect_nested_F_test",
    "Reference_country": "Brazil",
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
    TABLES / "R2_country_global_test_summary.csv",
    index=False
)

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("COUNTRY COEFFICIENTS — BRAZIL REFERENCE")
print("=" * 80)

print(
    country_coef.to_string(index=False)
)

print("\n" + "=" * 80)
print("GLOBAL COUNTRY TEST")
print("=" * 80)

print(
    global_test.to_string(index=False)
)

print("\nNested-model ANOVA:")
print(comparison.to_string())

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
