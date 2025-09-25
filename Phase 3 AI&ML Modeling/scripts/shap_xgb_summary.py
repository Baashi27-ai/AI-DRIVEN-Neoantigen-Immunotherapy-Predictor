#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

RESERVED = {"sample_id","response","age","stage"}

def load_xy(d):
    feat = pd.read_csv(os.path.join(d,"work","features.tsv"), sep="\t")
    if feat.columns[0] != "sample_id":
        feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(d,"inputs","labels.tsv"), sep="\t")
    df   = lab.merge(feat, on="sample_id", how="inner").fillna(0)
    y    = df["response"].astype(int).values
    cols = [c for c in df.columns if c not in RESERVED]
    X    = df[cols].astype(float)
    return X, y, cols

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    X, y, cols = load_xy(args.dir)

    # class balance → scale_pos_weight
    neg, pos = (y==0).sum(), (y==1).sum()
    spw = float(neg) / max(1.0, float(pos))

    # your best XGB setup from runs (AUC≈0.782)
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="auto",
        random_state=42, n_jobs=-1, scale_pos_weight=spw,
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=1.0,
        min_child_weight=3, reg_lambda=1.0, n_estimators=800, gamma=0
    )
    model.fit(X.values, y)

    expl = shap.TreeExplainer(model)
    vals = expl.shap_values(X)

    # 1) beeswarm summary
    plt.figure(figsize=(10,6))
    shap.summary_plot(vals, X, feature_names=cols, show=False)
    plt.tight_layout()
    out1 = os.path.join(args.dir, "results", "shap_summary.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    # 2) top-10 mean(|SHAP|) bar chart (nice for reports)
    imp = np.abs(vals).mean(axis=0)
    ord_idx = np.argsort(imp)[::-1][:10]
    top_feats = [cols[i] for i in ord_idx]
    top_vals  = imp[ord_idx]
    plt.figure(figsize=(8,5))
    plt.barh(range(len(top_feats))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_feats))[::-1], top_feats[::-1])
    plt.xlabel("mean(|SHAP value|)")
    plt.title("XGB feature importance (top 10)")
    plt.tight_layout()
    out2 = os.path.join(args.dir, "results", "shap_top10.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    print("[OK] Wrote", out1)
    print("[OK] Wrote", out2)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    main(ap.parse_args())