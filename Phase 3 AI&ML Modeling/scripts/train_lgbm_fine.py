# -- coding: utf-8 --
"""
Train LightGBM with fine hyperparameter grid search (5-fold CV).
Outputs:
  - results/lgbm_fine_roc.png
  - results/lgbm_fine_pr.png
  - results/model_performance.tsv (updated with lgbm_fine row)
  - models/lgbm_fine_best.pkl
"""

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import lightgbm as lgb
import joblib

def load_xy(dir_):
    feat = pd.read_csv(os.path.join(dir_,"work","features.tsv"), sep="\t")
    feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(dir_,"inputs","labels.tsv"), sep="\t")
    df = pd.merge(lab, feat, on="sample_id", how="inner").fillna(0)
    y = df["response"].astype(int).values
    X = df[[c for c in df.columns if c not in ["sample_id","response","age","stage"]]].astype(float).values
    cols = [c for c in df.columns if c not in ["sample_id","response","age","stage"]]
    return X, y, cols

def cv_eval(params, X, y, folds=5, seed=42):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs, aps, mean_fpr, tprs, prs = [], [], np.linspace(0,1,200), [], []
    for tr, te in cv.split(X,y):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dte = lgb.Dataset(X[te], label=y[te])
        bst = lgb.train(params, dtr, num_boost_round=params.get("n_estimators",1200))
        p = bst.predict(X[te])
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        fpr, tpr, _ = roc_curve(y[te], p)
        prec, rec, _ = precision_recall_curve(y[te], p)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        prs.append(np.interp(np.linspace(0,1,200), rec[::-1], prec[::-1]))
    return float(np.mean(aucs)), float(np.mean(aps)), mean_fpr, np.mean(tprs,axis=0), np.mean(prs,axis=0)

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dir,"models"), exist_ok=True)
    X, y, cols = load_xy(args.dir)

    pos, neg = float((y==1).sum()), float((y==0).sum())
    scale_pos_weight = max(1.0, neg/pos) if pos>0 else 1.0

    # grid search space
    grid = [
        dict(objective="binary", metric="auc", boosting_type="gbdt",
             learning_rate=lr, max_depth=md, num_leaves=nl,
             feature_fraction=ff, bagging_fraction=bf, bagging_freq=1,
             min_data_in_leaf=mdil, reg_lambda=rl, n_estimators=ne,
             scale_pos_weight=scale_pos_weight, verbose=-1)
        for lr in [0.03, 0.05, 0.08]
        for md in [3, 4, 6]
        for nl in [15, 31, 63]
        for ff in [0.6, 0.8, 1.0]
        for bf in [0.8, 1.0]
        for mdil in [5, 10, 20]
        for rl in [1, 5, 10]
        for ne in [800, 1200, 1600]
    ]

    best = None
    for i, params in enumerate(grid, 1):
        auc, ap, fpr, tpr, prc = cv_eval(params, X, y)
        if (best is None) or (auc > best[0]):
            best = (auc, ap, params, fpr, tpr, prc)

    auc, ap, params, fpr, tpr, prc = best
    bst = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=params["n_estimators"])
    joblib.dump((bst, params, cols), os.path.join(args.dir,"models","lgbm_fine_best.pkl"))

    # save plots
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"LGBM fine ROC AUC={auc:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","lgbm_fine_roc.png"), dpi=160); plt.close()

    plt.figure(); plt.plot(np.linspace(0,1,200), prc)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"LGBM fine PR AP={ap:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","lgbm_fine_pr.png"), dpi=160); plt.close()

    # update performance table
    perf = os.path.join(args.dir,"results","model_performance.tsv")
    row = pd.DataFrame([["omics","lgbm_fine", auc, ap]], columns=["feature_set","model","roc_auc","pr_auc"])
    if os.path.exists(perf):
        prev = pd.read_csv(perf, sep="\t")
        out = pd.concat([prev, row], ignore_index=True)
    else:
        out = row
    out.to_csv(perf, sep="\t", index=False)

    print(f"[OK] Best LGBM fine ROC-AUC={auc:.3f}")
    print(f"[OK] Updated -> {perf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    args = ap.parse_args()
    main(args)