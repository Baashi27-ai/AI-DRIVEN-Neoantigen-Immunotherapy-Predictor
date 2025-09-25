# -- coding: utf-8 --
import os, argparse, pandas as pd, numpy as np
SAMPLE=["sample","sample_id","tumor_sample","patient_id"]
AFF=["mhcflurry_affinity","ic50","IC50","affinity","predicted_affinity","ic50_affinity","ba"]
PRES=["mhcflurry_presentation_score","presentation_score","score","score_present","presentation"]
PEP=["peptide","sequence","pep","aa_seq"]
def sniff(p):
    with open(p,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            ln=ln.strip()
            if ln: return "," if ln.count(",")>=ln.count("\t") else "\t"
    return ","
def pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None
def topk_stats(df, col, higher, ks=(5,10)):
    out={}; x=pd.to_numeric(df[col],errors="coerce").dropna().values
    if x.size==0:
        for k in ks: out[f"mean_top{k}"]=0.0
        out["min"]=0.0; out["max"]=0.0; return out
    xs=np.sort(x)[::-1] if higher else np.sort(x)
    for k in ks: out[f"mean_top{k}"]=float(xs[:min(k,xs.size)].mean()) if xs.size else 0.0
    out["min"]=float(x.min()); out["max"]=float(x.max()); return out
def per_sample(df, samp, score, pep, higher, pref):
    rows=[]
    for sid,sub in df.groupby(samp):
        r={"sample_id":sid,
           f"{pref}_n":int(len(sub)),
           f"{pref}_uniq_pep":int(sub[pep].nunique()) if pep in sub else int(len(sub))}
        r.update(topk_stats(sub, score, higher))
        r={ (k if k=="sample_id" else f"{pref}_{k}"):v for k,v in r.items()}
        rows.append(r)
    return pd.DataFrame(rows)
def main(args):
    base=os.path.abspath(args.dir); work=os.path.join(base,"work")
    os.makedirs(work,exist_ok=True)
    feat=pd.read_csv(os.path.join(work,"features.tsv"),sep="\t").rename(columns=lambda c:"sample_id" if c==_.columns[0] else c) if ( _:=pd.read_csv(os.path.join(work,"features.tsv"),sep="\t") ) is not None else None
    feat=feat.rename(columns={feat.columns[0]:"sample_id"})
    p2=os.path.join(base,"..","Phase 2 Neoantigen Prediction Pipeline","work")
    b=os.path.join(p2,"mhcflurry_binding.tsv"); pr=os.path.join(p2,"mhcflurry_presentation.tsv")
    bind=pd.read_csv(b,sep=sniff(b)); pres=pd.read_csv(pr,sep=sniff(pr))
    samp=pick(bind,SAMPLE) or pick(pres,SAMPLE) or "sample"
    if samp not in bind: bind[samp]="S1"
    if samp not in pres: pres[samp]="S1"
    aff=pick(bind,AFF); pepb=pick(bind,PEP)
    prescol=pick(pres,PRES) or pick(pres,AFF); pepp=pick(pres,PEP)
    if aff is None: raise ValueError(f"no affinity col in bind: {list(bind.columns)}")
    if prescol is None: raise ValueError(f"no presentation col in pres: {list(pres.columns)}")
    bstats=per_sample(bind,samp,aff,pepb,False,"bind")
    pstats=per_sample(pres,samp,prescol,pepp,True,"pres")
    out=feat.merge(bstats,on="sample_id",how="left").merge(pstats,on="sample_id",how="left").fillna(0)
    outp=os.path.join(work,"features_topk.tsv"); out.to_csv(outp,sep="\t",index=False)
    print("[OK] Wrote topK features ->",outp); print("[OK] Shape:",out.shape)
