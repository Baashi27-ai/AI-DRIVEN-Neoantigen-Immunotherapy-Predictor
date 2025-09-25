#!/usr/bin/env python3
import os, argparse, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

RESERVED = {"sample_id","response","age","stage"}

def load_xy(d):
    feat = pd.read_csv(os.path.join(d,"work","features.tsv"), sep="\t")
    if feat.columns[0] != "sample_id":
        feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(d,"inputs","labels.tsv"), sep="\t")
    df   = pd.merge(lab, feat, on="sample_id", how="inner").fillna(0)
    y    = df["response"].astype(int).values
    Xc   = [c for c in df.columns if c not in RESERVED]
    X    = df[Xc].astype(float).values
    return df["sample_id"].tolist(), X, y, Xc

def get_xgb():
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="auto",
        random_state=42, n_jobs=-1,
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=1.0,
        min_child_weight=3, reg_lambda=1.0, n_estimators=800
    )

def get_lgbm():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        max_depth=-1, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, n_jobs=-1
    )

def cv_stack(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_xgb = np.zeros_like(y, dtype=float)
    oof_lgb = np.zeros_like(y, dtype=float)
    models = []

    for tr, te in skf.split(X, y):
        ytr = y[tr]
        neg, pos = (ytr==0).sum(), (ytr==1).sum()
        spw = float(neg) / max(1.0, float(pos))

        xgbm = get_xgb();  xgbm.set_params(scale_pos_weight=spw)
        lgbm = get_lgbm(); lgbm.set_params(scale_pos_weight=spw)

        xgbm.fit(X[tr], ytr)
        lgbm.fit(X[tr], ytr)

        oof_xgb[te] = xgbm.predict_proba(X[te])[:,1]
        oof_lgb[te] = lgbm.predict_proba(X[te])[:,1]
        models.append((xgbm, lgbm))

    Z = np.vstack([oof_xgb, oof_lgb]).T
    meta = LogisticRegression(max_iter=200, solver="liblinear")
    meta.fit(Z, y)

    # full-fit base models for inference
    full_xgb = get_xgb();  full_xgb.fit(X, y)
    full_lgb = get_lgbm(); full_lgb.fit(X, y)

    return (oof_xgb, oof_lgb, meta, full_xgb, full_lgb)

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dir,"models"), exist_ok=True)

    sids, X, y, cols = load_xy(args.dir)
    oof_xgb, oof_lgb, meta, full_xgb, full_lgb = cv_stack(X, y, n_splits=5)

    # Stacked OOF predictions
    Z = np.vstack([oof_xgb, oof_lgb]).T
    p  = meta.predict_proba(Z)[:,1]

    auc = roc_auc_score(y, p)
    ap  = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    pr_p, pr_r, _ = precision_recall_curve(y, p)

    # Curves
    plt.figure(); plt.plot([0,1],[0,1],'--'); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Stack (cal) ROC AUC={auc:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","roc_curves.png"), dpi=150); plt.close()

    plt.figure(); plt.plot(pr_r, pr_p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Stack (cal) AP={ap:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","pr_curves.png"), dpi=150); plt.close()

    # Save model bundle
    bundle = dict(meta=meta, xgb=full_xgb, lgbm=full_lgb, feature_names=cols)
    with open(os.path.join(args.dir,"models","final_model.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    # Log performance
    perf = os.path.join(args.dir,"results","model_performance.tsv")
    row  = pd.DataFrame([["omics","stack_cal", auc, ap]], columns=["feature_set","model","roc_auc","pr_auc"])
    if os.path.exists(perf):
        out = pd.concat([pd.read_csv(perf, sep="\t"), row], ignore_index=True)
    else:
        out = row
    out.to_csv(perf, sep="\t", index=False)

    print(f"[OK] STACK calibrated ROC-AUC={auc:.3f}  AP={ap:.3f}")
    print(f"[OK] Saved -> models/final_model.pkl")
    print(f"[OK] Updated -> {perf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    main(ap.parse_args())
