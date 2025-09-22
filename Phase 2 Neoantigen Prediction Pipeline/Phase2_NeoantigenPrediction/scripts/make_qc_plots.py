from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.cwd()
pep = BASE / "inputs" / "peptides" / "derived_peptides.tsv"
bind = BASE / "work" / "mhcflurry_binding.tsv"
pres = BASE / "work" / "mhcflurry_presentation.tsv"
out_candidates = BASE / "results" / "neoantigen_candidates.tsv"
out_candidates.parent.mkdir(parents=True, exist_ok=True)

# Load data
df_pep = pd.read_csv(pep, sep="\t")
df_bind = pd.read_csv(bind)   # mhcflurry CSV
df_pres = pd.read_csv(pres)

# Merge
df = df_pep.merge(df_bind, how="left", on="peptide")
df = df.merge(df_pres[["allele","peptide","presentation_score"]], how="left", on=["allele","peptide"])

# Filters
df["pass_binding"] = df["affinity"] < 500
df["pass_expr"] = df["expression_tpm"].fillna(0) > 0
df["pass_pres"] = df["presentation_score"].fillna(0) > 0.5
df["pass_any"] = (df["pass_binding"] | df["pass_pres"]) & df["pass_expr"]

df_out = df[df["pass_any"]].copy().sort_values(by=["affinity","presentation_score"], ascending=[True, False])
df_out.to_csv(out_candidates, sep="\t", index=False)
print(f"[OK] wrote candidates -> {out_candidates} (n={len(df_out)})")

# Plots
plt.figure()
df["affinity"].dropna().plot(kind="hist", bins=50)
plt.xlabel("Predicted affinity (nM)"); plt.ylabel("Count"); plt.title("Binding Affinity Histogram")
plt.tight_layout(); plt.savefig("binding_affinity_histogram.png", dpi=150)

plt.figure()
df_pep["length"].plot(kind="hist", bins=range(7,13))
plt.xlabel("Peptide length"); plt.ylabel("Count"); plt.title("Peptide Length Distribution")
plt.tight_layout(); plt.savefig("peptide_length_distribution.png", dpi=150)

steps = ["All 8–11mer peptides", "Filter: expression > 0", "Filter: affinity < 500 or pres > 0.5", "Final candidates"]
counts = [
    len(df_pep),
    int(df.merge(df[["peptide","pass_expr"]].drop_duplicates(), on="peptide", how="left")["pass_expr"].fillna(False).sum()),
    int((df["pass_binding"] | df["pass_pres"]).sum()),
    len(df_out)
]
plt.figure()
plt.plot(range(len(steps)), counts, marker="o")
plt.xticks(range(len(steps)), steps, rotation=20)
plt.ylabel("Count"); plt.title("Peptide Filtering Flow (counts)")
plt.tight_layout(); plt.savefig("peptide_filtering_flowchart.png", dpi=150)

print("[OK] Plots saved: binding_affinity_histogram.png, peptide_length_distribution.png, peptide_filtering_flowchart.png")
