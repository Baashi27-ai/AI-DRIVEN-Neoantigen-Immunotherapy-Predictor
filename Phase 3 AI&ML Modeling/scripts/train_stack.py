# -- coding: utf-8 --
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib, warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_xy(dir_):
    feat = pd.read_csv(os.path.join(dir_,"work","features.tsv"), sep="\t")
    feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(dir_,"inputs","labels.tsv"), sep="\t")
    df   = pd.merge(lab, feat, on="sample_id", how="inner").fillna(0)
    y    = df["response"].astype(int).values
    Xcols = [c for c in df.columns if c not in ["sample_id","response","age","stage"]]
    X = df[Xcols].astype(float).values
    return X, y, Xcols

def make_models(spw):
    import xgboost as xgb, lightgbm as lgb
    rf = RandomForestClassifier(
        n_estimators=800, max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    )
    xgbc = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="auto",
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=1.0,
        min_child_weight=3, gamma=0, reg_lambda=1, n_estimators=900,
        scale_pos_weight=spw, n_jobs=-1, random_state=42
    )
    lgbm = lgb.LGBMClassifier(
        objective="binary", metric="auc", boosting_type="gbdt",
        learning_rate=0.05, max_depth=4, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=10, reg_lambda=5, n_estimators=1400,
        class_weight=None, n_jobs=-1, random_state=42
    )
    # We’ll pass class weight by scale_pos_weight via fit_params for LGBM per fold
    return rf, xgbc, lgbm

def stack_cv(X, y, n_splits=5, n_repeats=3, seeds=(42,7,2025)):
    n = len(y)
    P_rf_all = np.zeros(n); P_xgb_all = np.zeros(n); P_lgb_all = np.zeros(n); cnt = 0

    for seed in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for tr, te in rskf.split(X, y):
            ytr = y[tr]
            pos = (ytr==1).sum(); neg = (ytr==0).sum()
            spw = float(neg)/max(1.0,float(pos))
            rf, xgbc, lgbm = make_models(spw)

            # RF
            rf.fit(X[tr], ytr)
            P_rf_all[te] += rf.predict_proba(X[te])[:,1]

            # XGB (already has scale_pos_weight)
            xgbc.fit(X[tr], ytr)
            P_xgb_all[te] += xgbc.predict_proba(X[te])[:,1]

            # LGBM with per-fold scale_pos_weight
            lgbm.fit(X[tr], ytr, **{"sample_weight": np.where(ytr==1, spw, 1.0)})
            P_lgb_all[te] += lgbm.predict_proba(X[te])[:,1]

            cnt += 1

    # average OOF preds across all folds and seeds
    P_rf  = P_rf_all  / cnt
    P_xgb = P_xgb_all / cnt
    P_lgb = P_lgb_all / cnt

    # meta learner
    Z = np.vstack([P_rf, P_xgb, P_lgb]).T
    meta = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=400, class_weight="balanced"))])
    meta.fit(Z, y)
    P_meta = meta.predict_proba(Z)[:,1]

    auc = roc_auc_score(y, P_meta)
    ap  = average_precision_score(y, P_meta)
    fpr, tpr, _ = roc_curve(y, P_meta)
    prec, rec, _ = precision_recall_curve(y, P_meta)

    # also fit bases on full data for export
    pos = (y==1).sum(); neg = (y==0).sum()
    spw_full = float(neg)/max(1.0,float(pos))
    rf, xgbc, lgbm = make_models(spw_full)
    rf.fit(X, y); xgbc.fit(X, y); lgbm.fit(X, y, **{"sample_weight": np.where(y==1, spw_full, 1.0)})

    return auc, ap, fpr, tpr, prec, rec, meta, {"rf": rf, "xgb": xgbc, "lgbm": lgbm}

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dir,"models"), exist_ok=True)
    X, y, cols = load_xy(args.dir)

    auc, ap, fpr, tpr, prec, rec, meta, bases = stack_cv(X, y)

    # plots
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"STACK (balanced, repeats) ROC AUC={auc:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","stack_roc.png"), dpi=160); plt.close()
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"STACK PR AP={ap:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","stack_pr.png"), dpi=160); plt.close()

    # log table
    perf = os.path.join(args.dir,"results","model_performance.tsv")
    row = pd.DataFrame([["omics","stack_balanced_repeat", auc, ap]], columns=["feature_set","model","roc_auc","pr_auc"])
    out = pd.concat([pd.read_csv(perf, sep="\t"), row], ignore_index=True) if os.path.exists(perf) else row
    out.to_csv(perf, sep="\t", index=False)

    joblib.dump({"meta": meta, "bases": bases, "features": cols}, os.path.join(args.dir,"models","stack.pkl"))
    print(f"[OK] STACK balanced repeats ROC-AUC={auc:.3f}  AP={ap:.3f}")
    print(f"[OK] Updated -> {perf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    main(ap.parse_args())
