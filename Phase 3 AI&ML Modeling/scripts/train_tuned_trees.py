import os, argparse, warnings, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB=True
except Exception:
    HAVE_XGB=False

def load_features(dir_):
    df = pd.read_csv(os.path.join(dir_, "work", "features.tsv"), sep="\t")
    df = df.rename(columns={df.columns[0]: "sample_id"})
    lab = pd.read_csv(os.path.join(dir_, "inputs", "labels.tsv"), sep="\t")
    df = pd.merge(lab, df, on="sample_id", how="inner").fillna(0)
    y = df["response"].astype(int).values
    X = df[[c for c in df.columns if c not in ["sample_id","response","age","stage"]]].astype(float).values
    return df["sample_id"].values, X, y

def mean_cv_auc(model, X, y, splits=5, seed=42):
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    aucs, prs = [], []
    mean_fpr = np.linspace(0,1,200); tprs=[]; prs_curve=[]
    for tr, te in cv.split(X,y):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
        fpr, tpr, _ = roc_curve(y[te], p)
        prec, rec, _ = precision_recall_curve(y[te], p)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        prs_curve.append(np.interp(np.linspace(0,1,200), rec[::-1], prec[::-1]))
    return float(np.mean(aucs)), float(np.mean(prs)), mean_fpr, np.mean(tprs,axis=0), np.mean(prs_curve,axis=0)

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dir,"models"),  exist_ok=True)
    sids, X, y = load_features(args.dir)

    results = []

    # ---- Random Forest grid ----
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [400, 800, 1200],
        "max_depth": [None, 6, 8, 12],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_gs = GridSearchCV(rf, rf_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    rf_gs.fit(X, y)
    rf_best = rf_gs.best_estimator_
    rf_auc, rf_ap, fpr, tpr, prc = mean_cv_auc(rf_best, X, y)
    results.append(["omics","rf_tuned", rf_auc, rf_ap])
    joblib.dump(rf_best, os.path.join(args.dir,"models","rf_tuned.pkl"))

    # plots
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"RF tuned ROC AUC={rf_auc:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","rf_tuned_roc.png"), dpi=160); plt.close()
    plt.figure(); plt.plot(np.linspace(0,1,200), prc); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"RF tuned PR AP={rf_ap:.3f}"); plt.tight_layout()
    plt.savefig(os.path.join(args.dir,"results","rf_tuned_pr.png"), dpi=160); plt.close()

    # ---- XGBoost grid (if available) ----
    if HAVE_XGB:
        xgb_clf = xgb.XGBClassifier(n_jobs=-1, eval_metric="logloss", random_state=42, tree_method="auto")
        xgb_grid = {
            "n_estimators": [400, 800, 1200],
            "max_depth": [3,4,6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5]
        }
        xgb_gs = GridSearchCV(xgb_clf, xgb_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        xgb_gs.fit(X, y)
        xgb_best = xgb_gs.best_estimator_
        xgb_auc, xgb_ap, fpr, tpr, prc = mean_cv_auc(xgb_best, X, y)
        results.append(["omics","xgb_tuned", xgb_auc, xgb_ap])
        joblib.dump(xgb_best, os.path.join(args.dir,"models","xgb_tuned.pkl"))
        plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"XGB tuned ROC AUC={xgb_auc:.3f}"); plt.tight_layout()
        plt.savefig(os.path.join(args.dir,"results","xgb_tuned_roc.png"), dpi=160); plt.close()
        plt.figure(); plt.plot(np.linspace(0,1,200), prc); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"XGB tuned PR AP={xgb_ap:.3f}"); plt.tight_layout()
        plt.savefig(os.path.join(args.dir,"results","xgb_tuned_pr.png"), dpi=160); plt.close()

    # save table
    df = pd.DataFrame(results, columns=["feature_set","model","roc_auc","pr_auc"])
    df_path = os.path.join(args.dir,"results","model_performance.tsv")
    if os.path.exists(df_path):
        prev = pd.read_csv(df_path, sep="\t")
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(df_path, sep="\t", index=False)
    print("[OK] Updated ->", df_path)
    best = df.sort_values("roc_auc", ascending=False).head(1).iloc[0]
    print(f"[OK] Best so far: {best['model']}  ROC-AUC={best['roc_auc']:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="Phase 3 folder")
    args = p.parse_args()
    main(args)
