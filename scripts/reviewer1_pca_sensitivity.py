#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

df = pd.read_csv(INPUT, sep="\t")

signature_cols = [c for c in df.columns if c.startswith("SBS")]

for c in signature_cols:
    df[c] = pd.to_numeric(df[c], errors="raise")

df["APOBEC_count"] = df["SBS2"] + df["SBS13"]
df["Total_SBS"] = df[signature_cols].sum(axis=1)
df["APOBEC_prop"] = df["APOBEC_count"] / df["Total_SBS"]

non_apobec = [
    c for c in signature_cols
    if c not in ["SBS2", "SBS13"]
]

print("=" * 80)
print("PCA SENSITIVITY ANALYSIS")
print("=" * 80)

print(f"\nTumors: {len(df)}")
print(f"All signatures: {len(signature_cols)}")
print(f"APOBEC-excluded signatures: {len(non_apobec)}")
print(non_apobec)


def association(x, y, label):
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


def run_pca(matrix, variables, prefix):

    pca = PCA()
    scores = pca.fit_transform(matrix)

    variance = pd.DataFrame({
        "PC": [
            f"PC{i+1}"
            for i in range(len(pca.explained_variance_ratio_))
        ],
        "Variance_explained":
            pca.explained_variance_ratio_,
        "Percent_variance_explained":
            pca.explained_variance_ratio_ * 100,
        "Cumulative_percent":
            np.cumsum(pca.explained_variance_ratio_) * 100
    })

    variance.to_csv(
        TABLES / f"{prefix}_variance_explained.csv",
        index=False
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=variables,
        columns=[
            f"PC{i+1}"
            for i in range(pca.components_.shape[0])
        ]
    )

    loadings.index.name = "Signature"

    loadings.reset_index().to_csv(
        TABLES / f"{prefix}_loadings.csv",
        index=False
    )

    score_df = pd.DataFrame({
        "Samples": df["Samples"]
    })

    for i in range(scores.shape[1]):
        score_df[f"PC{i+1}"] = scores[:, i]

    score_df.to_csv(
        TABLES / f"{prefix}_scores.csv",
        index=False
    )

    return pca, scores, variance, loadings


# ============================================================
# ANALYSIS 1
# ORIGINAL PCA RECONSTRUCTION
#
# Unscaled raw signature activities.
# This should reproduce the basic structure of the original PCA.
# ============================================================

X_all = df[signature_cols].to_numpy(dtype=float)

pca_all, scores_all, var_all, load_all = run_pca(
    X_all,
    signature_cols,
    "R1_PCA_all_signatures_unscaled"
)


# ============================================================
# ANALYSIS 2
# APOBEC-EXCLUDED PCA
#
# Remove SBS2 and SBS13 before PCA.
# ============================================================

X_no_apobec = df[non_apobec].to_numpy(dtype=float)

pca_no, scores_no, var_no, load_no = run_pca(
    X_no_apobec,
    non_apobec,
    "R1_PCA_APOBEC_excluded_unscaled"
)


# ============================================================
# ANALYSIS 3
# STANDARDIZED APOBEC-EXCLUDED PCA
#
# Standardize each signature to mean 0 and SD 1 so that
# high-variance signatures do not automatically dominate PCA.
# ============================================================

scaler = StandardScaler()

X_no_scaled = scaler.fit_transform(
    df[non_apobec]
)

pca_scaled, scores_scaled, var_scaled, load_scaled = run_pca(
    X_no_scaled,
    non_apobec,
    "R1_PCA_APOBEC_excluded_scaled"
)


# ============================================================
# ASSOCIATIONS WITH APOBEC
# ============================================================

rows = []

for pc_index in [0, 1]:

    pc = f"PC{pc_index + 1}"

    rows.append(
        association(
            df["APOBEC_prop"],
            scores_all[:, pc_index],
            f"Original_all_signatures_unscaled_APOBEC_prop_vs_{pc}"
        )
    )

    rows.append(
        association(
            df["APOBEC_count"],
            scores_all[:, pc_index],
            f"Original_all_signatures_unscaled_APOBEC_count_vs_{pc}"
        )
    )

    rows.append(
        association(
            df["APOBEC_prop"],
            scores_no[:, pc_index],
            f"APOBEC_excluded_unscaled_APOBEC_prop_vs_{pc}"
        )
    )

    rows.append(
        association(
            df["APOBEC_count"],
            scores_no[:, pc_index],
            f"APOBEC_excluded_unscaled_APOBEC_count_vs_{pc}"
        )
    )

    rows.append(
        association(
            df["APOBEC_prop"],
            scores_scaled[:, pc_index],
            f"APOBEC_excluded_scaled_APOBEC_prop_vs_{pc}"
        )
    )

    rows.append(
        association(
            df["APOBEC_count"],
            scores_scaled[:, pc_index],
            f"APOBEC_excluded_scaled_APOBEC_count_vs_{pc}"
        )
    )


assoc = pd.DataFrame(rows)

assoc.to_csv(
    TABLES / "R1_PCA_APOBEC_associations.csv",
    index=False
)


# ============================================================
# TOP LOADINGS
# ============================================================

def top_loadings(loadings, name):

    rows = []

    for pc in ["PC1", "PC2"]:

        temp = loadings[[pc]].copy()

        temp["Absolute_loading"] = (
            temp[pc].abs()
        )

        temp = temp.sort_values(
            "Absolute_loading",
            ascending=False
        )

        for rank, (sig, row) in enumerate(
            temp.head(10).iterrows(),
            start=1
        ):

            rows.append({
                "PCA": name,
                "PC": pc,
                "Rank": rank,
                "Signature": sig,
                "Loading": row[pc],
                "Absolute_loading":
                    row["Absolute_loading"]
            })

    return rows


top_rows = []

top_rows.extend(
    top_loadings(
        load_all,
        "All_signatures_unscaled"
    )
)

top_rows.extend(
    top_loadings(
        load_no,
        "APOBEC_excluded_unscaled"
    )
)

top_rows.extend(
    top_loadings(
        load_scaled,
        "APOBEC_excluded_scaled"
    )
)

top_table = pd.DataFrame(top_rows)

top_table.to_csv(
    TABLES / "R1_PCA_top_loadings.csv",
    index=False
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL PCA: ALL SIGNATURES, UNSCALED")
print("=" * 80)

print("\nVariance explained:")
print(
    var_all.head(5).to_string(index=False)
)

print("\nPC1 loadings:")
print(
    load_all["PC1"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)

print("\nPC2 loadings:")
print(
    load_all["PC2"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)


print("\n" + "=" * 80)
print("APOBEC-EXCLUDED PCA: UNSCALED")
print("=" * 80)

print("\nVariance explained:")
print(
    var_no.head(5).to_string(index=False)
)

print("\nPC1 loadings:")
print(
    load_no["PC1"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)

print("\nPC2 loadings:")
print(
    load_no["PC2"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)


print("\n" + "=" * 80)
print("APOBEC-EXCLUDED PCA: STANDARDIZED")
print("=" * 80)

print("\nVariance explained:")
print(
    var_scaled.head(5).to_string(index=False)
)

print("\nPC1 loadings:")
print(
    load_scaled["PC1"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)

print("\nPC2 loadings:")
print(
    load_scaled["PC2"]
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)


print("\n" + "=" * 80)
print("APOBEC ASSOCIATIONS WITH PCA SCORES")
print("=" * 80)

print(
    assoc.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"""
Key outputs:

{TABLES / 'R1_PCA_APOBEC_associations.csv'}

{TABLES / 'R1_PCA_all_signatures_unscaled_variance_explained.csv'}

{TABLES / 'R1_PCA_all_signatures_unscaled_loadings.csv'}

{TABLES / 'R1_PCA_APOBEC_excluded_unscaled_variance_explained.csv'}

{TABLES / 'R1_PCA_APOBEC_excluded_unscaled_loadings.csv'}

{TABLES / 'R1_PCA_APOBEC_excluded_scaled_variance_explained.csv'}

{TABLES / 'R1_PCA_APOBEC_excluded_scaled_loadings.csv'}

{TABLES / 'R1_PCA_top_loadings.csv'}
"""
)
