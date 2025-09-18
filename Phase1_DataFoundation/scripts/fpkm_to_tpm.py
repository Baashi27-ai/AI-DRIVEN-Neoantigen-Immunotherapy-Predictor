import pandas as pd
from pathlib import Path

infile  = Path("../expression/expression_matrix.tsv")
outfile = Path("../expression/expression_tpm.tsv")

print(f"Reading FPKM: {infile.resolve()}")
df = pd.read_csv(infile, sep="\t")

# assume first column is gene identifier; keep it separate
gene_col = df.columns[0]
genes = df[gene_col]
expr = df.drop(columns=[gene_col])

# convert all to numeric where possible
expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# TPM per sample: tpm = fpkm / sum(fpkm) * 1e6
col_sums = expr.sum(axis=0).replace(0, 1.0)  # avoid divide-by-zero
tpm = expr.div(col_sums, axis=1) * 1_000_000

# stitch gene column back
out = pd.concat([genes, tpm], axis=1)
out.to_csv(outfile, sep="\t", index=False)
print(f"Saved TPM: {outfile.resolve()}")
