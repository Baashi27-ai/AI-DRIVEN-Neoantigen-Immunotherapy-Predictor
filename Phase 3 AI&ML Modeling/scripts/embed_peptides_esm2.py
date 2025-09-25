# -- coding: utf-8 --
"""
Peptide embeddings with ESM-2 (small, CPU-friendly) + PCA → merge into features.
Outputs:
  - work/features_with_pep.tsv
  - results/peptide_embedding_heatmap.png  (top-variance components)
  - results/peptide_embedding_heatmap.csv  (their variances)
Run:
  conda activate neo-phase1
  python scripts/embed_peptides_esm2.py --dir . --max_per_sample 200
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"   # tiny, works on CPU

# -------------------- utilities --------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_peptides(tsv_path):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing peptides file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    pep_col = find_col(df, ["peptide","Peptide","seq","sequence"])
    if pep_col is None:
        raise ValueError("No peptide column found (expected one of: peptide/Peptide/seq/sequence).")
    samp_col = find_col(df, ["sample","sample_id","tumor_sample","patient_id"])
    if samp_col is None:
        df["sample"] = "S1"; samp_col = "sample"

    out = df[[samp_col, pep_col]].rename(columns={samp_col: "sample", pep_col: "peptide"})
    # basic cleanup
    out["peptide"] = out["peptide"].astype(str).str.strip()
    out = out[out["peptide"].str.len() > 0].copy()
    return out

def mean_pool_last_hidden(mdl, tok, seq, device):
    enc = tok(seq, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state  # [1, L, D]
        v = out.mean(dim=1).squeeze(0).cpu().numpy()  # [D]
    return v

def compute_embeddings(peps: pd.DataFrame, max_per_sample=200, device="cpu"):
    """Return a DataFrame (index = sample_id, columns = embedding dims)."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME).to(device)
    mdl.eval()

    sample_vecs = {}
    for s, sub in peps.groupby("sample"):
        # cap peptides per sample for speed/variance control
        sub = sub.sample(min(len(sub), max_per_sample), random_state=42)
        vecs = []
        for seq in sub["peptide"]:
            try:
                vecs.append(mean_pool_last_hidden(mdl, tok, seq, device))
            except Exception as e:
                # skip bad sequences gracefully
                print(f"[WARN] Skipping peptide for sample {s}: {e}")
        if len(vecs) == 0:
            # fallback zero vector of model hidden size (try to infer)
            hidden = getattr(mdl.config, "hidden_size", 32)
            vecs = [np.zeros(hidden, dtype=np.float32)]
        sample_vecs[s] = np.vstack(vecs).mean(axis=0)

    emb = pd.DataFrame.from_dict(sample_vecs, orient="index")
    emb.index.name = "sample_id"
    return emb

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------- main --------------------
def main(args):
    base = os.path.abspath(args.dir)
    inp  = os.path.join(base, "inputs")
    work = os.path.join(base, "work")
    res  = os.path.join(base, "results")
    ensure_dirs(work, res)

    # 1) load peptides
    pep_file = os.path.join(inp, "derived_peptides.tsv")
    peps = load_peptides(pep_file)

    # 2) device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 3) embeddings (mean over tokens, then mean over peptides per sample)
    emb = compute_embeddings(peps, max_per_sample=args.max_per_sample, device=device)
    print(f"[OK] Raw embedding matrix: {emb.shape[0]} samples x {emb.shape[1]} dims")

    # 4) PCA compress (dynamic n_components so it never errors)
    n_comp = int(min(32, emb.shape[1], max(2, emb.shape[0]-1)))  # at least 2, at most dims & samples-1
    pca = PCA(n_components=n_comp, random_state=42)
    Z = pca.fit_transform(emb.values)
    cols = [f"pep_emb35M_{i+1}" for i in range(Z.shape[1])]
    Z = pd.DataFrame(Z, index=emb.index, columns=cols)

    # 5) merge into features.tsv → features_with_pep.tsv
    feat_path = os.path.join(work, "features.tsv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing base features: {feat_path} (run build_features.py first)")
    feat = pd.read_csv(feat_path, sep="\t")
    feat = feat.rename(columns={feat.columns[0]: "sample_id"}).set_index("sample_id")
    feat2 = feat.join(Z, how="left").fillna(0).reset_index()
    out_path = os.path.join(work, "features_with_pep.tsv")
    feat2.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote embeddings → {out_path}")

    # 6) quick QC: top-variance components as heatmap-like visualization
    var = feat2[cols].var().sort_values(ascending=False)
    var.head(25).to_csv(os.path.join(res, "peptide_embedding_heatmap.csv"))
    try:
        top_cols = var.head(min(25, len(cols))).index.tolist()
        plt.figure()
        plt.imshow(feat2[top_cols].values, aspect="auto", interpolation="nearest")
        plt.title("Peptide Embedding (top-variance components)")
        plt.xlabel("components"); plt.ylabel("samples")
        plt.tight_layout()
        plt.savefig(os.path.join(res, "peptide_embedding_heatmap.png"), dpi=160)
        plt.close()
        print("[OK] Saved QC heatmap -> results/peptide_embedding_heatmap.png")
    except Exception as e:
        print(f"[WARN] Could not plot heatmap: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Path to 'Phase 3 AI&ML Modeling'")
    ap.add_argument("--max_per_sample", type=int, default=200)
    args = ap.parse_args()
    main(args)