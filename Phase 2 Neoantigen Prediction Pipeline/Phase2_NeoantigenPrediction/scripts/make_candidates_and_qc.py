#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent.parent
work = base / "work"
inputs = base / "inputs"

pred = pd.read_csv(work / "mhcflurry_binding.tsv")  # has affinity + presentation cols
mut = pd.read_csv(inputs / "variant_info" / "mutations_with_expression.tsv", sep="\t")
expr = pd.read_csv(inputs / "expression" / "expression_tpm.tsv", sep="\t")

# ---- normalize columns
# expected: pred has columns: allele, peptide, mhcflurry_affinity, mhcflurry_presentation_score, ...
pred.rename(columns={
    "mhcflurry_affinity": "affinity_nm",
    "mhcflurry_presentation_score": "presentation_score",
    "mhcflurry_presentation_percentile": "presentation_percentile"
}, inplace=True)

# join by peptide where possible; if you have peptide->gene mapping, merge on that.
# For now we carry peptide-level only and attach expression heuristics by gene symbol when possible.
# If your mutations_with_expression has a column 'aa_change' or 'peptide' mapping, use it here.
join_cols = [c.lower() for c in mut.columns]
if "peptide" in join_cols:
    # exact peptide mapping available
    mut.columns = [c.lower() for c in mut.columns]
    merged = pred.merge(mut, on="peptide", how="left")
else:
    merged = pred.copy()
    # TODO: if you provide peptide<->gene mapping later, we’ll upgrade this join.

# ---- simple filters (Phase 2 spec)
# binding < 500 nM  (strong binders)
cand = merged[merged["affinity_nm"] < 500].copy()

# VAF threshold (if available)
if "vaf" in cand.columns:
    cand = cand[cand["vaf"] >= 0.1]  # tweak threshold as needed (10% default)

# expression confirmed (if available)
expr_flag_col = None
for name in ["expr_confirmed", "expression_confirmed", "expressed"]:
    if name in cand.columns:
        expr_flag_col = name
        break
if expr_flag_col:
    cand = cand[cand[expr_flag_col] == True]

# ---- Save candidates
out_candidates = work / "neoantigen_candidates.tsv"
cand.to_csv(out_candidates, sep="\t", index=False)
print(f"[OK] candidates -> {out_candidates} (n={len(cand)})")

# ---- QC plots
# 1) affinity histogram
plt.figure()
pred["affinity_nm"].clip(upper=5000).hist(bins=40)
plt.xlabel("Predicted affinity (nM, clipped at 5000)")
plt.ylabel("Count")
plt.title("Binding affinity distribution")
plt.tight_layout()
plt.savefig(work / "binding_affinity_histogram.png")
print("[OK] binding_affinity_histogram.png")

# 2) peptide length distribution
plt.figure()
pred["peptide"].astype(str).str.len().hist(bins=range(7,15))
plt.xlabel("Peptide length")
plt.ylabel("Count")
plt.title("Peptide length distribution")
plt.tight_layout()
plt.savefig(work / "peptide_length_distribution.png")
print("[OK] peptide_length_distribution.png")

# 3) simple filtering flow image (text-based placeholder)
flow = """Phase 2 Filtering Flow
Total predicted pairs: {}
Affinity < 500 nM: {}
+ VAF >= 0.10 (if available): {}
+ Expression confirmed (if available): {}
""".format(len(pred), sum(pred["affinity_nm"] < 500), len(cand), len(cand))
(work / "peptide_filtering_flowchart.txt").write_text(flow, encoding="utf-8")
print("[OK] peptide_filtering_flowchart.txt (placeholder). For a diagram, we can render later as PNG/SVG.")