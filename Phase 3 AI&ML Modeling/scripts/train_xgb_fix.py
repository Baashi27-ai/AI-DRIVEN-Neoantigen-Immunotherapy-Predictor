# -- coding: utf-8 --
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import xgboost as xgb

RESERVED = set(["sample_id","response","age","stage"])

def load_xy(d):
    feat = pd.read_csv(os.path.join(d,"work","features.tsv"), sep="\t")
    if feat.columns[0] != "sample_id":
        feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab = pd.read_csv(os.path.join(d,"inputs","labels.tsv"), sep="\t")
    df = pd.merge(lab, feat, on="sample_id", how="inner").fillna(0)
    y = df["response"].astype(int).values
    Xcols = [c for c in df.columns if c not in RESERVED]
    X = df[Xcols].astype(float).values
    if X.shape[1] == 0:
        raise RuntimeError("No feature columns found after excluding reserved columns.")
    return X, y, Xcols

def cv_eval(params, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        ytr = y[tr]
        neg, pos = (ytr==0).sum(), (ytr==1).sum()
        spw = float(neg) / max(1.0, float(pos))
        p = params.copy()
        p.update(dict(objective="binary:logistic", eval_metric="auc", tree_method="auto",
                      random_state=seed, n_jobs=-1, scale_pos_weight=spw))
        model = xgb.XGBClassifier(**p)
        model.fit(X[tr], ytr)
        preds[te] = model.predict_proba(X[te])[:,1]
    auc = roc_auc_score(y, preds)
    ap  = average_precision_score(y, preds)
    fpr, tpr, _ = roc_curve(y, preds)
    prc_p, prc_r, _ = precision_recall_curve(y, preds)
    return auc, ap, fpr, tpr, prc_p, prc_r

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    X, y, cols = load_xy(args.dir)
    # small param set tuned for tiny cohorts
    grid = [
        dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.0, n_estimators=1200, gamma=0),
        dict(max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=2, reg_lambda=5.0, n_estimators=1400, gamma=0),
        dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=1.0, min_child_weight=3, reg_lambda=1.0, n_estimators=800,  gamma=0),
    ]
    best = (-1, None)
    for g in grid:
        auc, ap, fpr, tpr, pp, rr = cv_eval(g, X, y)
        if auc > best[0]: best = (auc, (g, ap, fpr, tpr, pp, rr))
    auc, (g, ap, fpr, tpr, pp, rr) = best
    # plot curves
    plt.figure(); plt.plot([0,1],[0,1],'--'); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"XGB (fix) ROC AUC={auc:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","xgb_fix_roc.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(rr, pp); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"XGB (fix) AP={ap:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","xgb_fix_pr.png"), dpi=150); plt.close()
    # log row
    perf = os.path.join(args.dir,"results","model_performance.tsv")
    row  = pd.DataFrame([["omics","xgb_fix", auc, ap]], columns=["feature_set","model","roc_auc","pr_auc"])
    if os.path.exists(perf):
        out = pd.concat([pd.read_csv(perf, sep="\t"), row], ignore_index=True)
    else:
        out = row
    out.to_csv(perf, sep="\t", index=False)
    print(f"[OK] XGB_fix ROC-AUC={auc:.3f}  AP={ap:.3f}")
    print(f"[OK] Updated -> {perf}")

if __name__ == "__main_S_":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    main(ap.parse_args())
