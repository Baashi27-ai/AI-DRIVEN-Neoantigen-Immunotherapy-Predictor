import pandas as pd
from pathlib import Path

root = Path("..")
mut = pd.read_csv(root/"vcf/somatic_mutations.tsv", sep="\t")
expr = pd.read_csv(root/"expression/expression_tpm.tsv", sep="\t")

# assume first column of expr is gene_id
gene_col = expr.columns[0]
expr = expr.set_index(gene_col)

# fetch TPM for each mutation's gene/sample
def get_tpm(row):
    g = row["gene_symbol"]
    s = row["sample_id"]
    try:
        return float(expr.at[g, s])
    except Exception:
        return 0.0

mut["expression_TPM"] = mut.apply(get_tpm, axis=1)
mut["expr_confirmed"] = mut["expression_TPM"] > 1.0

out = root/"results/mutations_with_expression.tsv"
mut.to_csv(out, sep="\t", index=False)
print(f"Saved -> {out.resolve()}")
