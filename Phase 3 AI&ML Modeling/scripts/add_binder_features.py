# -- coding: utf-8 --
"""
Add binder-derived features to Phase 3 feature matrix.

Robustly reads Phase-2 binding/presentation tables (auto-detect ',' vs '\t'),
computes per-sample stats:
  - binders_total / strong (<50 nM) / weak (<500 nM)  [affinity-like scores: lower is better]
  - fractions (strong/weak)
  - (optional) mean expression column if present

Outputs:
  - work/features_binder.tsv
"""

import os, argparse, pandas as pd, numpy as np

# Column name candidates
SAMPLE_CANDIDATES = ["sample","sample_id","tumor_sample","patient_id"]
IC50_CANDIDATES   = ["mhcflurry_affinity","ic50","IC50","affinity","predicted_affinity","ic50_affinity","ba"]
PRES_CANDIDATES   = ["mhcflurry_presentation_score","presentation_score","score","score_present","presentation"]
EXPR_CANDIDATES   = ["expression","expr","tpm","TPM"]

def sniff_delim(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if line:
                c = line.count(",")
                t = line.count("\t")
                if c==0 and t==0:
                    return ","  # fallback
                return "," if c>=t else "\t"
    return ","  # default

def read_smart(path):
    sep = sniff_delim(path)
    return pd.read_csv(path, sep=sep)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_binder_stats(df, sample_col, score_col, strong_thr, weak_thr, expr_col=None, higher_is_better=False):
    df = df.copy()
    df[score_col] = num(df[score_col])
    if expr_col and expr_col in df.columns:
        df[expr_col] = num(df[expr_col])

    rows = []
    for sid, sub in df.groupby(sample_col):
        n_total = len(sub)
        sc = sub[score_col].dropna()
        if higher_is_better:
            n_strong = int((sc >= strong_thr).sum())
            n_weak   = int((sc >= weak_thr).sum())
        else:
            n_strong = int((sc <  strong_thr).sum())
            n_weak   = int((sc <  weak_thr).sum())
        rows.append({
            "sample_id": sid,
            "binders_total": int(n_total),
            "binders_strong": n_strong,
            "binders_weak": n_weak,
            "binders_frac_strong": (n_strong/n_total) if n_total else 0.0,
            "binders_frac_weak":   (n_weak  /n_total) if n_total else 0.0,
            **({"binders_expr_mean": float(sub[expr_col].dropna().mean())} if expr_col and expr_col in sub.columns else {})
        })
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "sample_id","binders_total","binders_strong","binders_weak",
            "binders_frac_strong","binders_frac_weak","binders_expr_mean"
        ])
    return out

def main(args):
    base = os.path.abspath(args.dir)
    work = os.path.join(base, "work")
    os.makedirs(work, exist_ok=True)

    # Base Phase-3 features
    feat_path = os.path.join(work, "features.tsv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing base features at {feat_path}. Run build_features.py first.")
    feat = pd.read_csv(feat_path, sep="\t")
    feat = feat.rename(columns={feat.columns[0]: "sample_id"})

    # Phase-2 paths
    p2 = os.path.join(base, "..", "Phase 2 Neoantigen Prediction Pipeline", "work")
    bind_path = os.path.join(p2, "mhcflurry_binding.tsv")
    pres_path = os.path.join(p2, "mhcflurry_presentation.tsv")
    if not os.path.exists(bind_path): raise FileNotFoundError(f"Binding file not found: {bind_path}")
    if not os.path.exists(pres_path): raise FileNotFoundError(f"Presentation file not found: {pres_path}")

    # Read with delimiter sniffing
    bind = read_smart(bind_path)
    pres = read_smart(pres_path)

    # Detect columns
    sample_col = pick_col(bind, SAMPLE_CANDIDATES) or pick_col(pres, SAMPLE_CANDIDATES)
    if sample_col is None:
        bind["sample"] = "S1"; pres["sample"] = "S1"; sample_col = "sample"

    ic50_col = pick_col(bind, IC50_CANDIDATES)
    if ic50_col is None:
        raise ValueError(f"No IC50/affinity-like column in binding file.\nColumns: {list(bind.columns)}")

    pres_col = pick_col(pres, PRES_CANDIDATES) or pick_col(pres, IC50_CANDIDATES)
    if pres_col is None:
        raise ValueError(f"No presentation score-like column in presentation file.\nColumns: {list(pres.columns)}")

    expr_col = pick_col(bind, EXPR_CANDIDATES)

    # Binding stats (affinity in nM: lower better)
    bstats = compute_binder_stats(bind, sample_col, ic50_col,
                                  strong_thr=50, weak_thr=500,
                                  expr_col=expr_col, higher_is_better=False)

    # Presentation stats (higher better) with data-driven thresholds
    pres_vals = num(pres[pres_col])
    if pres_vals.notna().sum() > 0:
        strong_thr = float(pres_vals.quantile(0.95))  # top 5%
        weak_thr   = float(pres_vals.quantile(0.80))  # top 20%
    else:
        strong_thr, weak_thr = -np.inf, -np.inf
    pstats = compute_binder_stats(pres, sample_col, pres_col,
                                  strong_thr=strong_thr, weak_thr=weak_thr,
                                  expr_col=None, higher_is_better=True)

    # Merge & write
    feat2 = (feat.merge(bstats, on="sample_id", how="left")
                .merge(pstats, on="sample_id", how="left", suffixes=("_bind","_pres"))
                .fillna(0))
    out_path = os.path.join(work, "features_binder.tsv")
    feat2.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote binder-enriched features -> {out_path}")
    print(f"[OK] Shape: {feat2.shape}")
    added = [c for c in feat2.columns if c.endswith(("bind","_pres")) or c.startswith("binders")]
    print("[INFO] Added columns:", added)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Path to 'Phase 3 AI&ML Modeling'")
    args = ap.parse_args()
    main(args)