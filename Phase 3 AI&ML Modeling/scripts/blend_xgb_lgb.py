#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

RES = {"sample_id","response","age","stage"}

def load_xy(d):
    feat = pd.read_csv(os.path.join(d,"work","features.tsv"), sep="\t")
    if feat.columns[0]!="sample_id": feat=feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(d,"inputs","labels.tsv"), sep="\t")
    df   = lab.merge(feat, on="sample_id", how="inner").fillna(0)
    y    = df["response"].astype(int).values
    X    = df[[c for c in df.columns if c not in RES]].astype(float).values
    return X,y

def main(args):
    X,y = load_xy(args.dir)
    from xgboost import XGBClassifier
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_x = np.zeros_like(y, dtype=float)
    oof_l = np.zeros_like(y, dtype=float)
    for tr,te in skf.split(X,y):
        ytr=y[tr]; neg=(ytr==0).sum(); pos=(ytr==1).sum(); spw=float(neg)/max(1.0,float(pos))
        xgb = XGBClassifier(objective="binary:logistic",eval_metric="auc",tree_method="auto",
                            random_state=42,n_jobs=-1,max_depth=3,learning_rate=0.03,subsample=0.8,
                            colsample_bytree=1.0,min_child_weight=3,reg_lambda=1.0,n_estimators=800,
                            scale_pos_weight=spw).fit(X[tr],ytr)
        lgbm = lgb.LGBMClassifier(objective="binary",n_estimators=1200,learning_rate=0.03,
                                  num_leaves=31,subsample=0.9,colsample_bytree=0.9,
                                  reg_lambda=1.0,random_state=42,n_jobs=-1,
                                  scale_pos_weight=spw).fit(X[tr],ytr)
        oof_x[te]=xgb.predict_proba(X[te])[:,1]
        oof_l[te]=lgbm.predict_proba(X[te])[:,1]
    best=(0,None)
    for w in np.linspace(0,1,21):
        p=w*oof_x+(1-w)*oof_l
        auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
        if auc>best[0]: best=(auc,(w,ap))
    auc,(w,ap)=best
    # save chosen blend as final_model.pkl-compatible object
    import pickle
    bundle=dict(kind="blend", weight=w)
    with open(os.path.join(args.dir,"models","final_model.pkl"),"wb") as f: pickle.dump(bundle,f)
    # log
    perf=os.path.join(args.dir,"results","model_performance.tsv")
    row=pd.DataFrame([["omics",f"blend_xgb{w:.2f}_lgb{1-w:.2f}",auc,ap]],columns=["feature_set","model","roc_auc","pr_auc"])
    out=pd.concat([pd.read_csv(perf,sep="\t"),row],ignore_index=True) if os.path.exists(perf) else row
    out.to_csv(perf,sep="\t",index=False)
    print(f"[OK] Blend best w={w:.2f}  ROC-AUC={auc:.3f}  AP={ap:.3f}")
