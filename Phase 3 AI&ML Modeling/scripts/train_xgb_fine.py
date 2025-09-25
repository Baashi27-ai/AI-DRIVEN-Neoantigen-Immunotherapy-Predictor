import os, argparse, joblib, warnings, itertools, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore")
import xgboost as xgb

def load_xy(dir_):
    feat = pd.read_csv(os.path.join(dir_,"work","features.tsv"), sep="\t")
    feat = feat.rename(columns={feat.columns[0]:"sample_id"})
    lab  = pd.read_csv(os.path.join(dir_,"inputs","labels.tsv"), sep="\t")
    df = pd.merge(lab, feat, on="sample_id", how="inner").fillna(0)
    y = df["response"].astype(int).values
    X = df[[c for c in df.columns if c not in ["sample_id","response","age","stage"]]].astype(float).values
    return df["sample_id"].values, X, y

def eval_params(params, X, y, folds=5, seed=42):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs, aps = [], []
    mean_fpr = np.linspace(0,1,200); tprs=[]; prs=[]
    for tr, te in cv.split(X,y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        # early stopping by holding out 15% of the train as eval
        rs = np.random.RandomState(42)
        idx = rs.permutation(len(Xtr))
        cut = max(1, int(0.85*len(idx)))
        tr_idx, ev_idx = idx[:cut], idx[cut:]
        dtrain = xgb.DMatrix(Xtr[tr_idx], label=ytr[tr_idx])
        deval  = xgb.DMatrix(Xtr[ev_idx], label=ytr[ev_idx])
        dtest  = xgb.DMatrix(Xte,       label=yte)
        watch = [(dtrain,"train"), (deval,"eval")]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators",800),
            evals=watch,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        p = bst.predict(dtest, iteration_range=(0, bst.best_iteration+1))
        aucs.append(roc_auc_score(yte, p))
        aps.append(average_precision_score(yte, p))
        fpr, tpr, _ = roc_curve(yte, p)
        prec, rec, _ = precision_recall_curve(yte, p)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        prs.append(np.interp(np.linspace(0,1,200), rec[::-1], prec[::-1]))
    return float(np.mean(aucs)), float(np.mean(aps)), mean_fpr, np.mean(tprs,axis=0), np.mean(prs,axis=0)

def main(args):
    os.makedirs(os.path.join(args.dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dir,"models"), exist_ok=True)
    sids, X, y = load_xy(args.dir)

    pos = float((y==1).sum()); neg = float((y==0).sum())
    spw = max(1.0, neg/pos) if pos>0 else 1.0  # scale_pos_weight

    # focused, stronger grid
    grid = {
        "max_depth":       [3,4,6],
        "learning_rate":   [0.03, 0.05, 0.08],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
        "gamma":           [0, 1],
        "reg_lambda":      [1, 5, 10],
        "n_estimators":    [800, 1200, 1600]
    }

    combos = list(itertools.product(
        grid["max_depth"], grid["learning_rate"], grid["subsample"],
        grid["colsample_bytree"], grid["min_child_weight"], grid["gamma"],
        grid["reg_lambda"], grid["n_estimators"]
    ))

    best = None
    results = []
    for i, (md, lr, ss, cs, mcw, gm, rl, ne) in enumerate(combos, 1):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "auto",
            "random_state": 42,
            "max_depth": md,
            "learning_rate": lr,
            "subsample": ss,
            "colsample_bytree": cs,
            "min_child_weight": mcw,
            "gamma": gm,
            "reg_lambda": rl,
            "n_estimators": ne,
            "scale_pos_weight": spw,
        }
        auc, ap, fpr, tpr, prc = eval_params(params, X, y, folds=5, seed=42)
        results.append(["omics", f"xgb_fine(md={md},lr={lr},ss={ss},cs={cs},mcw={mcw},g={gm},rl={rl},n={ne})", auc, ap])

        if (best is None) or (auc > best[0]):
            best = (auc, ap, params, fpr, tpr, prc)

    # Save best model trained on full data
    best_auc, best_ap, best_params, fpr, tpr, prc = best
    dfull = xgb.DMatrix(X, label=y)
    bst = xgb.train(best_params, dfull, num_boost_round=best_params["n_estimators"])
    joblib.dump((bst, best_params), os.path.join(args.dir,"models","xgb_fine_best.pkl"))

    # Plots
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"XGB fine ROC AUC={best_auc:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","xgb_fine_roc.png"), dpi=160); plt.close()

    plt.figure(); plt.plot(np.linspace(0,1,200), prc)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"XGB fine PR AP={best_ap:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.dir,"results","xgb_fine_pr.png"), dpi=160); plt.close()

    # Append to performance table
    perf_path = os.path.join(args.dir,"results","model_performance.tsv")
    df_new = pd.DataFrame(results, columns=["feature_set","model","roc_auc","pr_auc"])
    if os.path.exists(perf_path):
        prev = pd.read_csv(perf_path, sep="\t")
        df_new = pd.concat([prev, df_new], ignore_index=True)
    df_new.to_csv(perf_path, sep="\t", index=False)

    print(f"[OK] Best XGB fine ROC-AUC={best_auc:.3f}  (params: {best_params})")
    print(f"[OK] Updated -> {perf_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Phase 3 folder")
    args = ap.parse_args()
    main(args)
