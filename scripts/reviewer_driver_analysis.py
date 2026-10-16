#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

MERGED = ROOT / "results/APOBEC_driver_merged.csv"
MUTATIONS = ROOT / "results/driver_mutations_from_vcf.csv"

OUT = ROOT / "results/revision_R1"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

GENES = [
    "TP53",
    "KMT2D",
    "CDKN2A",
    "NOTCH1",
    "PIK3CA",
    "FAT1",
    "NFE2L2",
    "KEAP1",
    "ZNF750",
    "CUL3",
    "AJUBA"
]

df = pd.read_csv(MERGED)
mut = pd.read_csv(MUTATIONS)

# Ensure numeric
df["APOBEC_burden"] = pd.to_numeric(
    df["APOBEC_burden"],
    errors="coerce"
)

df["APOBEC_prop"] = pd.to_numeric(
    df["APOBEC_prop"],
    errors="coerce"
)

for gene in GENES:
    df[gene] = pd.to_numeric(
        df[gene],
        errors="coerce"
    )

print("=" * 110)
print("REVIEWER DRIVER-GENE ANALYSIS")
print("=" * 110)

print("\nN tumors:", len(df))
print("Genes tested:", len(GENES))


# ============================================================
# EFFECT SIZE
# Rank-biserial correlation derived from Mann-Whitney U.
#
# Positive value = APOBEC burden tends to be higher
# in mutant than WT tumors.
# ============================================================

def rank_biserial_from_u(U, n_mut, n_wt):
    return (2 * U) / (n_mut * n_wt) - 1


# ============================================================
# ANALYZE ALL 11 GENES
# ============================================================

rows = []

for gene in GENES:

    g = df[
        [gene, "APOBEC_burden",
         "APOBEC_prop", "APOBEC_group"]
    ].dropna(subset=[gene]).copy()

    mutant = g[g[gene] == 1]
    wt = g[g[gene] == 0]

    mut_count = mutant["APOBEC_burden"].dropna()
    wt_count = wt["APOBEC_burden"].dropna()

    mut_prop = mutant["APOBEC_prop"].dropna()
    wt_prop = wt["APOBEC_prop"].dropna()

    # --------------------------------------------------------
    # Absolute APOBEC burden
    # --------------------------------------------------------

    if len(mut_count) > 0 and len(wt_count) > 0:

        U_count, p_count = mannwhitneyu(
            mut_count,
            wt_count,
            alternative="two-sided"
        )

        rbc_count = rank_biserial_from_u(
            U_count,
            len(mut_count),
            len(wt_count)
        )

    else:
        U_count = p_count = rbc_count = np.nan

    # --------------------------------------------------------
    # APOBEC proportion
    # --------------------------------------------------------

    if len(mut_prop) > 0 and len(wt_prop) > 0:

        U_prop, p_prop = mannwhitneyu(
            mut_prop,
            wt_prop,
            alternative="two-sided"
        )

        rbc_prop = rank_biserial_from_u(
            U_prop,
            len(mut_prop),
            len(wt_prop)
        )

    else:
        U_prop = p_prop = rbc_prop = np.nan

    # --------------------------------------------------------
    # APOBEC high / low enrichment
    # --------------------------------------------------------

    high_mut = int(
        (
            (g[gene] == 1) &
            (g["APOBEC_group"] == "APOBEC-high")
        ).sum()
    )

    low_mut = int(
        (
            (g[gene] == 1) &
            (g["APOBEC_group"] == "APOBEC-low")
        ).sum()
    )

    high_wt = int(
        (
            (g[gene] == 0) &
            (g["APOBEC_group"] == "APOBEC-high")
        ).sum()
    )

    low_wt = int(
        (
            (g[gene] == 0) &
            (g["APOBEC_group"] == "APOBEC-low")
        ).sum()
    )

    table = [
        [high_mut, low_mut],
        [high_wt, low_wt]
    ]

    odds, fisher_p = fisher_exact(
        table,
        alternative="two-sided"
    )

    rows.append({

        "Gene": gene,

        "Mutated_tumors": int((g[gene] == 1).sum()),
        "WT_tumors": int((g[gene] == 0).sum()),

        "Mutation_frequency_percent":
            100 * (g[gene] == 1).mean(),

        # Count analysis
        "Mutant_APOBEC_count_median":
            mut_count.median(),

        "WT_APOBEC_count_median":
            wt_count.median(),

        "Median_count_difference":
            mut_count.median() - wt_count.median(),

        "Count_MannWhitney_U":
            U_count,

        "Count_rank_biserial":
            rbc_count,

        "Count_P_value":
            p_count,

        # Proportion analysis
        "Mutant_APOBEC_prop_median":
            mut_prop.median(),

        "WT_APOBEC_prop_median":
            wt_prop.median(),

        "Prop_rank_biserial":
            rbc_prop,

        "Prop_P_value":
            p_prop,

        # Median-split analysis
        "APOBEC_high_mutated":
            high_mut,

        "APOBEC_low_mutated":
            low_mut,

        "APOBEC_high_WT":
            high_wt,

        "APOBEC_low_WT":
            low_wt,

        "HighLow_odds_ratio":
            odds,

        "HighLow_Fisher_P":
            fisher_p
    })


res = pd.DataFrame(rows)


# ============================================================
# MULTIPLE TESTING
# Separate BH correction for each prespecified analysis family.
# ============================================================

res["Count_FDR_q"] = multipletests(
    res["Count_P_value"],
    method="fdr_bh"
)[1]

res["Prop_FDR_q"] = multipletests(
    res["Prop_P_value"],
    method="fdr_bh"
)[1]

res["HighLow_FDR_q"] = multipletests(
    res["HighLow_Fisher_P"],
    method="fdr_bh"
)[1]


# Sort by absolute-count P value
res = res.sort_values(
    "Count_P_value"
).reset_index(drop=True)


res.to_csv(
    TABLES /
    "R1_driver_all11_APOBEC_associations.csv",
    index=False
)


# ============================================================
# ZNF750 MUTATION-LEVEL CHARACTERIZATION
# ============================================================

znf = mut[
    mut["Gene"].astype(str).str.upper() == "ZNF750"
].copy()

znf = znf.sort_values(
    ["Sample", "Chrom", "Position"]
)

znf.to_csv(
    TABLES /
    "R1_ZNF750_mutation_details.csv",
    index=False
)


# Consequence summary
znf_consequence = (
    znf.groupby("Consequence", dropna=False)
       .agg(
           Mutation_records=("Sample", "size"),
           Tumors=("Sample", "nunique")
       )
       .reset_index()
       .sort_values(
           ["Tumors", "Mutation_records"],
           ascending=False
       )
)

znf_consequence.to_csv(
    TABLES /
    "R1_ZNF750_consequence_summary.csv",
    index=False
)


# Number of distinct ZNF750-mutant tumors
znf_tumors = znf["Sample"].nunique()


# ============================================================
# PRINT RESULTS
# ============================================================

display_cols = [
    "Gene",
    "Mutated_tumors",
    "Mutation_frequency_percent",
    "Mutant_APOBEC_count_median",
    "WT_APOBEC_count_median",
    "Median_count_difference",
    "Count_rank_biserial",
    "Count_P_value",
    "Count_FDR_q",
    "HighLow_odds_ratio",
    "HighLow_Fisher_P",
    "HighLow_FDR_q"
]

print("\n" + "=" * 110)
print("ALL 11 GENES — ABSOLUTE APOBEC COUNTS + HIGH/LOW ANALYSIS")
print("=" * 110)

print(
    res[display_cols].to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("ZNF750 MUTATION DETAILS")
print("=" * 110)

print("ZNF750 mutation records:", len(znf))
print("ZNF750-mutant tumors:", znf_tumors)

if len(znf):
    print(
        znf[
            [
                "Sample",
                "Consequence",
                "Protein_change",
                "Chrom",
                "Position",
                "Ref",
                "Alt"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 110)
print("ZNF750 CONSEQUENCE SUMMARY")
print("=" * 110)

print(
    znf_consequence.to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("FILES SAVED")
print("=" * 110)

for f in [
    "R1_driver_all11_APOBEC_associations.csv",
    "R1_ZNF750_mutation_details.csv",
    "R1_ZNF750_consequence_summary.csv"
]:
    print(TABLES / f)
