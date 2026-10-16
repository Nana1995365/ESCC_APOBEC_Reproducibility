#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MUT = ROOT / "results/revision_R1/tables/R1_ZNF750_mutation_details.csv"
SEQ = ROOT / "input/ESCC/output/vcf_files/SNV/17_seqinfo.txt"

OUT = ROOT / "results/revision_R1/tables"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD ZNF750 MUTATIONS
# ============================================================

znf = pd.read_csv(MUT)

znf["Sample"] = znf["Sample"].astype(str)
znf["Chrom"] = znf["Chrom"].astype(str)
znf["Position"] = pd.to_numeric(znf["Position"])

# ============================================================
# LOAD SIGPROFILER CHR17 SEQINFO
#
# Columns:
# Sample Chrom Position Context Strand
#
# Example:
# SAMPLE001 17 12345 B:TC[C>T]AG 1  (illustrative example)
# ============================================================

seq = pd.read_csv(
    SEQ,
    sep=r"\s+",
    header=None,
    names=[
        "Sample",
        "Chrom",
        "Position",
        "SigProfiler_context",
        "Strand"
    ]
)

seq["Sample"] = seq["Sample"].astype(str)
seq["Chrom"] = seq["Chrom"].astype(str)
seq["Position"] = pd.to_numeric(seq["Position"])

# ============================================================
# JOIN BY SAMPLE + CHROM + POSITION
# ============================================================

x = znf.merge(
    seq,
    on=["Sample", "Chrom", "Position"],
    how="left",
    validate="one_to_one"
)

# ============================================================
# PARSE SIGPROFILER CONTEXT
#
# Example:
# B:TC[C>T]AG
#
# Five-base sequence = TC C AG
# SBS96 trinucleotide uses:
# immediate 5' base + mutated base + immediate 3' base
#
# => C[C>T]A in this example.
# ============================================================

pattern = re.compile(
    r"^[^:]+:([ACGT])([ACGT])\[([CT])>([ACGT])\]([ACGT])([ACGT])$"
)

def parse_context(s):

    if pd.isna(s):
        return pd.Series({
            "Left2": None,
            "Left1": None,
            "Ref_normalized": None,
            "Alt_normalized": None,
            "Right1": None,
            "Right2": None,
            "SBS96_context": None,
            "APOBEC_TCW": None,
            "APOBEC_signature_compatible": None
        })

    m = pattern.match(str(s))

    if not m:
        return pd.Series({
            "Left2": None,
            "Left1": None,
            "Ref_normalized": None,
            "Alt_normalized": None,
            "Right1": None,
            "Right2": None,
            "SBS96_context": None,
            "APOBEC_TCW": None,
            "APOBEC_signature_compatible": None
        })

    left2, left1, ref, alt, right1, right2 = m.groups()

    sbs96 = f"{left1}[{ref}>{alt}]{right1}"

    # Canonical APOBEC motif:
    # T[C>T]W or T[C>G]W
    # where W = A or T.
    #
    # SBS2: predominantly C>T in TCW
    # SBS13: predominantly C>G in TCW
    tcw = (
        left1 == "T"
        and ref == "C"
        and right1 in {"A", "T"}
    )

    compatible = (
        tcw
        and alt in {"T", "G"}
    )

    return pd.Series({
        "Left2": left2,
        "Left1": left1,
        "Ref_normalized": ref,
        "Alt_normalized": alt,
        "Right1": right1,
        "Right2": right2,
        "SBS96_context": sbs96,
        "APOBEC_TCW": tcw,
        "APOBEC_signature_compatible": compatible
    })


parsed = x["SigProfiler_context"].apply(parse_context)

x = pd.concat(
    [x, parsed],
    axis=1
)

# ============================================================
# LABEL SBS2/SBS13-COMPATIBLE TYPE
# ============================================================

def apobec_type(row):

    if row["APOBEC_signature_compatible"] is not True:
        return "Not canonical APOBEC-compatible"

    if row["Alt_normalized"] == "T":
        return "SBS2-compatible"

    if row["Alt_normalized"] == "G":
        return "SBS13-compatible"

    return "Not canonical APOBEC-compatible"


x["APOBEC_context_class"] = x.apply(
    apobec_type,
    axis=1
)

# ============================================================
# SAVE DETAILS
# ============================================================

detail_cols = [
    "Sample",
    "Consequence",
    "Protein_change",
    "Chrom",
    "Position",
    "Ref",
    "Alt",
    "SigProfiler_context",
    "SBS96_context",
    "APOBEC_TCW",
    "APOBEC_signature_compatible",
    "APOBEC_context_class"
]

details = x[detail_cols].copy()

details.to_csv(
    OUT / "R1_ZNF750_APOBEC_context_details.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

n = len(details)
matched = details["SigProfiler_context"].notna().sum()

n_compatible = (
    details["APOBEC_signature_compatible"]
    .fillna(False)
    .sum()
)

summary = pd.DataFrame({
    "Metric": [
        "ZNF750 mutations",
        "Matched to SigProfiler context",
        "Canonical APOBEC-compatible",
        "Not canonical APOBEC-compatible",
        "Percent canonical APOBEC-compatible"
    ],
    "Value": [
        n,
        matched,
        n_compatible,
        n - n_compatible,
        100 * n_compatible / n if n else float("nan")
    ]
})

summary.to_csv(
    OUT / "R1_ZNF750_APOBEC_context_summary.csv",
    index=False
)

# Context-class summary
class_summary = (
    details["APOBEC_context_class"]
    .value_counts(dropna=False)
    .rename_axis("APOBEC_context_class")
    .reset_index(name="N")
)

class_summary.to_csv(
    OUT / "R1_ZNF750_APOBEC_context_classes.csv",
    index=False
)

# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("ZNF750 MUTATION CONTEXT ANALYSIS")
print("=" * 110)

print(
    details.to_string(
        index=False
    )
)

print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)

print(summary.to_string(index=False))

print("\n" + "=" * 110)
print("CONTEXT CLASSES")
print("=" * 110)

print(class_summary.to_string(index=False))

print("\nSaved:")
print(OUT / "R1_ZNF750_APOBEC_context_details.csv")
print(OUT / "R1_ZNF750_APOBEC_context_summary.csv")
print(OUT / "R1_ZNF750_APOBEC_context_classes.csv")
