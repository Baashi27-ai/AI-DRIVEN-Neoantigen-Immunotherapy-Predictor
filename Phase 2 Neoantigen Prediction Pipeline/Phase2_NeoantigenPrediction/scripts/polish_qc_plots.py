#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz

BASE = Path(__file__).resolve().parent.parent
WORK = BASE / "work"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- Load predictions ----------
pred_path = WORK / "mhcflurry_binding.tsv"
df = pd.read_csv(pred_path)

# Robust column names
aff_col = "mhcflurry_affinity" if "mhcflurry_affinity" in df.columns else "affinity_nm"
pep_col = "peptide" if "peptide" in df.columns else None
if pep_col is None:
    raise SystemExit("[FATAL] Could not find peptide column in predictions TSV")

# ---------- Plot 1: Binding Affinity (log-x) ----------
out_bind_png = RESULTS / "binding_affinity_histogram.png"
vals = df[aff_col].replace([np.inf, -np.inf], np.nan).dropna()
vals_clip = vals.clip(lower=0.1, upper=50000)  # avoid zeros on log scale

plt.figure(figsize=(7,5))
plt.hist(vals_clip, bins=np.logspace(np.log10(0.1), np.log10(50000), 50))
plt.xscale("log")
plt.xlabel("Predicted binding affinity (nM, log scale)")
plt.ylabel("Peptide–allele count")
plt.title("Distribution of Predicted Binding Affinities (nM)")

# 500 nM cutoff
cut = 500
plt.axvline(cut, linestyle="--")
plt.text(cut*1.05, plt.ylim()[1]*0.85, "500 nM cutoff", rotation=90, va="top")

# Tight layout + save
plt.tight_layout()
plt.savefig(out_bind_png, dpi=180)
plt.close()
# Also mirror to work/
(WORK / "binding_affinity_histogram.png").write_bytes(out_bind_png.read_bytes())

print(f"[OK] Polished -> {out_bind_png}")

# ---------- Plot 2: Peptide Length Distribution ----------
out_len_png = RESULTS / "peptide_length_distribution.png"
lengths = df[pep_col].astype(str).str.len()
bins = range(7, 15)  # show 7–14 to frame 8–11 nicely

plt.figure(figsize=(7,5))
plt.hist(lengths, bins=bins, align="left", rwidth=0.8)
plt.xticks(range(7,15))
plt.xlabel("Peptide length (aa)")
plt.ylabel("Count")
plt.title("Peptide Length Distribution")

# Highlight modal length
mode_len = lengths.mode().iloc[0]
plt.axvline(mode_len, linestyle="--")
plt.text(mode_len+0.1, plt.ylim()[1]*0.9, f"mode = {mode_len}-mer")

# Emphasize canonical window
plt.axvspan(8-0.5, 11+0.5, alpha=0.08)  # subtle background band for 8–11

plt.tight_layout()
plt.savefig(out_len_png, dpi=180)
plt.close()
(WORK / "peptide_length_distribution.png").write_bytes(out_len_png.read_bytes())
print(f"[OK] Polished -> {out_len_png}")

# ---------- Plot 3: Filtering Flowchart (Graphviz) ----------
out_flow_png = RESULTS / "peptide_filtering_flowchart.png"
dot = graphviz.Digraph(comment="Peptide Filtering Flow", format="png")
dot.attr(rankdir="TB")
dot.attr("node", fontname="Helvetica", fontsize="12")
dot.attr("edge", arrowsize="0.8")

dot.node("start", "All Derived Peptides", shape="box", style="filled", fillcolor="#D9EDF7")
dot.node("aff", "Filter: Binding affinity < 500 nM", shape="diamond", style="filled", fillcolor="#FCF8E3")
dot.node("vaf", "Filter: VAF ≥ 0.10", shape="diamond", style="filled", fillcolor="#FCF8E3")
dot.node("expr", "Filter: Expression confirmed", shape="diamond", style="filled", fillcolor="#FCF8E3")
dot.node("final", "Final Neoantigen Candidates", shape="box", style="filled", fillcolor="#DFF0D8")

dot.edge("start", "aff")
dot.edge("aff", "vaf")
dot.edge("vaf", "expr")
dot.edge("expr", "final")

# Render
out_stem = out_flow_png.with_suffix("")  # Graphviz adds extension automatically
dot.render(str(out_stem), format="png", cleanup=True)
# Copy into work/ too
(WORK / "peptide_filtering_flowchart.png").write_bytes(out_flow_png.read_bytes())
print(f"[OK] Polished -> {out_flow_png}")

print("\n>>> All polished plots saved in 'results/' and mirrored to 'work/'. Done.")