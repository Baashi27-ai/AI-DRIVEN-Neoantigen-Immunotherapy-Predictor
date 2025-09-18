#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Project root = Phase1_DataFoundation
root = Path(_file_).resolve().parents[1]

infile = root / "downloads" / "hugo_supp6.xlsx"   # Supplement 6 (mutations)
outfile = root / "results" / "somatic_mutations.tsv"

print(f"Reading: {infile}")
xl = pd.ExcelFile(infile)

frames = []
for sh in xl.sheet_names:
    df = xl.parse(sh)
    if df.empty:
        continue
    # case-insensitive column resolver
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*keys):
        for k in keys:
            if k in cols: return cols[k]
        return None
    gene   = pick("hugo_symbol","gene","symbol","gene_name")
    sample = pick("tumor_sample_barcode","sample","tumor","patient","sample_id","tumor sample barcode")
    prot   = pick("hgvsp_short","protein_change","protein change","aa_change","hgvs.p","variant")
    if gene and sample and prot:
        sub = df[[sample, gene, prot]].copy()
        sub.columns = ["sample_name","gene_symbol","protein_change"]
        frames.append(sub)

if not frames:
    first_cols = list(xl.parse(xl.sheet_names[0]).columns.astype(str))
    raise SystemExit("No mutation-like sheet found. First sheet columns: " + ", ".join(first_cols))

mut = pd.concat(frames, ignore_index=True)
mut["PtTag"] = mut["sample_name"].astype(str).str.extract(r"(Pt\\d+)", expand=False)
mut["sample_id"] = mut["PtTag"].fillna(mut["sample_name"].astype(str)) + ".baseline"

out = mut[["sample_id","gene_symbol","protein_change"]].dropna().drop_duplicates()
outfile.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(outfile, sep="\t", index=False)
print(f"Saved -> {outfile.resolve()}")
