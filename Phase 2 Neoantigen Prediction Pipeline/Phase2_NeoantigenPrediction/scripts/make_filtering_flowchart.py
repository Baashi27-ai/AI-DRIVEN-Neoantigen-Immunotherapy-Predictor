#!/usr/bin/env python3
import graphviz
from pathlib import Path

base = Path(__file__).resolve().parent.parent
out_png = base / "work" / "peptide_filtering_flowchart.png"

# Define the flow
dot = graphviz.Digraph(comment="Peptide Filtering Flow", format="png")
dot.attr(rankdir="TB", size="8")

dot.node("start", "All Derived Peptides", shape="box", style="filled", fillcolor="lightblue")
dot.node("affinity", "Filter: Binding affinity < 500 nM", shape="diamond", style="filled", fillcolor="lightyellow")
dot.node("vaf", "Filter: VAF ≥ 0.1", shape="diamond", style="filled", fillcolor="lightyellow")
dot.node("expr", "Filter: Expression confirmed", shape="diamond", style="filled", fillcolor="lightyellow")
dot.node("candidates", "Final Neoantigen Candidates", shape="box", style="filled", fillcolor="lightgreen")

# Arrows
dot.edge("start", "affinity")
dot.edge("affinity", "vaf")
dot.edge("vaf", "expr")
dot.edge("expr", "candidates")

# Render
dot.render(out_png.with_suffix(""), format="png", cleanup=True)
print(f"[OK] wrote flowchart -> {out_png}")