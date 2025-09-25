import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Helpers ---
def read_expression(path):
    df = pd.read_csv(path, sep='\t')
    # Expect columns: gene_id (or gene), sample_id columns or a long format
    # Try to detect wide vs long
    if {"gene","sample_id","tpm"}.issubset(df.columns):
        # long -> pivot
        df = df.pivot_table(index="gene", columns="sample_id", values="tpm", aggfunc="mean").fillna(0.0)
    else:
        # wide: first column is gene name, others are samples
        first = df.columns[0]
        df = df.set_index(first)
    df.index = df.index.str.upper()
    return df

def cyt_score(expr_wide):
    genes = ["GZMA","PRF1","CD8A"]
    have = [g for g in genes if g in expr_wide.index]
    if not have:
        # fallback: zeros
        return pd.Series(0.0, index=expr_wide.columns, name="CYT")
    log_expr = np.log2(expr_wide.loc[have].clip(lower=0)+1.0)
    return log_expr.mean(axis=0).rename("CYT")

def neoantigen_load(path):
    neo = pd.read_csv(path, sep='\t')
    # expect a sample column
    sample_col = None
    for c in ["sample","sample_id","tumor_sample","patient_id"]:
        if c in neo.columns: sample_col = c; break
    if sample_col is None:
        # try infer single sample file
        neo["sample"] = "S1"
        sample_col = "sample"
    return neo.groupby(sample_col).size().rename("neoantigen_load")

def tmb_per_mb(path, exome_mb=38.0):
    muts = pd.read_csv(path, sep='\t')
    sample_col = None
    for c in ["sample","sample_id","tumor_sample","patient_id"]:
        if c in muts.columns: sample_col = c; break
    if sample_col is None:
        muts["sample"] = "S1"
        sample_col = "sample"
    tmb = muts.groupby(sample_col).size() / exome_mb
    return tmb.rename("TMB")

def hla_zygosity_flag(path):
    hla = pd.read_csv(path, sep='\t')
    # Expect columns like sample and allele entries; we’ll collapse unique alleles
    sample_col = None
    for c in ["sample","sample_id","patient_id"]:
        if c in hla.columns: sample_col = c; break
    if sample_col is None:
        hla["sample"] = "S1"
        sample_col = "sample"
    # Collect all alleles per sample across loci
    allele_cols = [c for c in hla.columns if c != sample_col]
    def homozygous(row):
        # Count unique alleles in A/B/C; if any locus has only one unique -> homozygous
        # This is heuristic; adapt if you have explicit A1/A2 etc.
        alleles = []
        for c in allele_cols:
            v = str(row[c]).strip()
            if v and v != "nan": alleles += [a.strip() for a in v.replace(",", ";").split(";") if a.strip()]
        # If fewer than 6 unique across A,B,C, likely some homozygosity
        return int(len(set(alleles)) < 6)
    # If rows are already one per sample with all loci in columns:
    if sample_col in hla.columns and len(allele_cols) >= 2 and hla[sample_col].is_unique:
        z = hla.apply(homozygous, axis=1)
        return pd.Series(z.values, index=hla[sample_col].values, name="HLA_homozygous")
    # Else, aggregate by sample
    g = hla.groupby(sample_col)
    vals = []
    idx = []
    for s, df in g:
        alleles = set()
        for c in df.columns:
            if c == sample_col: continue
            for v in df[c].astype(str):
                if v and v != "nan":
                    alleles.update([a.strip() for a in v.replace(",", ";").split(";") if a.strip()])
        idx.append(s)
        vals.append(int(len(alleles) < 6))
    return pd.Series(vals, index=idx, name="HLA_homozygous")

def main(args):
    base = os.path.abspath(os.path.join(args.phase3_dir))
    inp = os.path.join(base, "inputs")
    work = os.path.join(base, "work")
    qc   = os.path.join(base, "qc")
    os.makedirs(work, exist_ok=True)
    os.makedirs(qc, exist_ok=True)

    expr = read_expression(os.path.join(inp, "expression_tpm.tsv"))
    cyt  = cyt_score(expr)

    neo  = neoantigen_load(os.path.join(inp, "neoantigen_candidates.tsv"))
    tmb  = tmb_per_mb(os.path.join(inp, "somatic_mutations.tsv"))
    hla  = hla_zygosity_flag(os.path.join(inp, "hla_types.tsv"))

    # Combine
    df = pd.concat([cyt, neo, tmb, hla], axis=1)
    df.index.name = "sample_id"
    df = df.fillna(0.0)

    # QC quick histograms
    for col in ["CYT","neoantigen_load","TMB"]:
        try:
            plt.figure()
            df[col].astype(float).hist(bins=30)
            plt.title(col)
            plt.xlabel(col); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(qc, f"{col.lower()}_hist.png"), dpi=160)
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not plot {col}: {e}")

    # Save features
    out = os.path.join(work, "features.tsv")
    df.to_csv(out, sep='\t')
    print(f"[OK] Features -> {out}")
    print(f"[OK] Samples: {len(df)}  Columns: {list(df.columns)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase3_dir", default=".", help="Path to 'Phase 3 AI&ML Modeling'")
    args = p.parse_args()
    main(args)
