#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

RES={"sample_id","response","age","stage"}
def load_xy(d):
    feat=pd.read_csv(os.path.join(d,"work","features.tsv"),sep="\t")
    if feat.columns[0]!="sample_id": feat=feat.rename(columns={feat.columns[0]:"sample_id"})
    lab =pd.read_csv(os.path.join(d,"inputs","labels.tsv"),sep="\t")
    df  =lab.merge(feat,on="sample_id",how="inner").fillna(0)
    y=df["response"].astype(int).values
    cols=[c for c in df.columns if c not in RES]
    X=df[cols].astype(float); return X,y,cols

def main(args):
    X,y,cols=load_xy(args.dir)
    neg=(y==0).sum(); pos=(y==1).sum(); spw=float(neg)/max(1.0,float(pos))
    model=lgb.LGBMClassifier(objective="binary",n_estimators=1200,learning_rate=0.03,
                             num_leaves=31,subsample=0.9,colsample_bytree=0.9,
                             reg_lambda=1.0,random_state=42,n_jobs=-1,scale_pos_weight=spw)
    model.fit(X,y)
    expl=shap.TreeExplainer(model)
    vals=expl.shap_values(X)
    shap.summary_plot(vals, X, feature_names=cols, show=False, plot_size=(10,6))
    os.makedirs(os.path.join(args.dir,"results"),exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","shap_summary.png"),dpi=160)
    print("[OK] Wrote results/shap_summary.png")
