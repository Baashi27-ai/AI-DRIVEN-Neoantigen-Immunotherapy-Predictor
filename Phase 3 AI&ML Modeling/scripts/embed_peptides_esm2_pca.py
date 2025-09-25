#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

SAMPLE_CANDS = ["sample","sample_id","tumor_sample","patient_id"]
PEP_CANDS    = ["peptide","sequence","pep","aa_seq"]
AFF_CANDS    = ["mhcflurry_affinity","ic50","IC50","affinity","predicted_affinity","ic50_affinity","ba"]
PRES_CANDS   = ["mhcflurry_presentation_score","presentation_score","score","score_present","presentation"]

def sniff_delim(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                return "," if ln.count(",") >= ln.count("\t") else "\t"
    return "\t"

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def load_phase2_best_peptide(phase3_dir):
    p2_work = os.path.join(phase3_dir, "..", "Phase 2 Neoantigen Prediction Pipeline", "work")
    bind_p  = os.path.join(p2_work, "mhcflurry_binding.tsv")
    pres_p  = os.path.join(p2_work, "mhcflurry_presentation.tsv")
    if not os.path.exists(bind_p) and not os.path.exists(pres_p):
        raise FileNotFoundError("Could not find Phase 2 binding/presentation files in '.../Phase 2 .../work/'")

    use_bind = os.path.exists(bind_p)
    df = pd.read_csv(bind_p if use_bind else pres_p, sep=sniff_delim(bind_p if use_bind else pres_p))

    samp = pick_col(df, SAMPLE_CANDS) or "sample"
    pep  = pick_col(df, PEP_CANDS)
    aff  = pick_col(df, AFF_CANDS)
    prs  = pick_col(df, PRES_CANDS)

    if pep is None:
        raise ValueError("No peptide column found in binding/presentation files.")

    if samp not in df: df[samp] = "S1"

    # Choose top peptide per sample:
    if aff is not None:
        df = df.sort_values(aff, ascending=True).groupby(samp).first().reset_index()
    elif prs is not None:
        df = df.sort_values(prs, ascending=True).groupby(samp).last().reset_index()  # highest score
    else:
        df = df.groupby(samp).first().reset_index()

    best = df[[samp, pep]].dropna().drop_duplicates()
    best.columns = ["sample_id","peptide"]
    return best

def embed_peptides(seq_list, model_name="facebook/esm2_t33_650M_UR50D", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model     = AutoModel.from_pretrained(model_name)
    model.to(device); model.eval()
    vecs = []
    with torch.no_grad():
        for s in seq_list:
            toks = tokenizer(s, return_tensors="pt", add_special_tokens=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks).last_hidden_state  # [1, L, d]
            v = out.mean(dim=1).squeeze(0).cpu().numpy()  # [d]
            vecs.append(v)
    return np.vstack(vecs)

def main(args):
    base = os.path.abspath(args.dir)
    work = os.path.join(base, "work")
    os.makedirs(work, exist_ok=True)

    # Load existing features (prefer topk_noemb, else topk)
    feat_path = os.path.join(work, "features_topk_noemb.tsv")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(work, "features_topk.tsv")
    feat = pd.read_csv(feat_path, sep="\t")
    if feat.columns[0] != "sample_id":
        feat = feat.rename(columns={feat.columns[0]:"sample_id"})

    # Best peptide per sample from Phase 2
    best = load_phase2_best_peptide(base)
    samp_ids = feat["sample_id"].tolist()
    best = best[best["sample_id"].isin(samp_ids)]
    # Ensure we cover all samples; fallback to 'M' if missing
    pep_map = dict(zip(best["sample_id"], best["peptide"]))
    seqs = [pep_map.get(sid, "M") for sid in samp_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Embedding {len(seqs)} peptides on {device} ...")
    E = embed_peptides(seqs, device=device)  # shape [N, d]

    print("[INFO] PCA -> 32 dims")
    pca = PCA(n_components=32, random_state=42).fit(E)
    Z  = pca.transform(E)
    Zdf = pd.DataFrame(Z, columns=[f"pep650_pca{i+1}" for i in range(32)])
    Zdf.insert(0, "sample_id", samp_ids)

    feat2 = feat.merge(Zdf, on="sample_id", how="left").fillna(0)
    out = os.path.join(work, "features_topk_esm650_pca32.tsv")
    feat2.to_csv(out, sep="\t", index=False)
    print(f"[OK] Wrote -> {out}  shape={feat2.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    main(ap.parse_args())
