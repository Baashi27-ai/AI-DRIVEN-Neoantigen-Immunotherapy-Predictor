import os, argparse, joblib, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def load_data(phase3_dir):
    feat = pd.read_csv(os.path.join(phase3_dir, "work", "features.tsv"), sep="\t")
    feat = feat.rename(columns={feat.columns[0]: "sample_id"})
    lab  = pd.read_csv(os.path.join(phase3_dir, "inputs", "labels.tsv"), sep="\t")
    if "sample_id" not in lab.columns or "response" not in lab.columns:
        raise ValueError("labels.tsv must have columns: sample_id, response[, age, stage]")

    # Merge labels with features
    df = pd.merge(lab, feat, on="sample_id", how="inner")

    # ✅ Fix: fill any missing values
    df = df.fillna(0)

    # Extract response
    y = df["response"].astype(int).values

    # Baseline clinical features
    clinical_cols = [c for c in ["age", "stage"] if c in df.columns]

    # If stage present, convert to numeric
    if "stage" in clinical_cols:
        df["stage"] = df["stage"].astype(str).str.extract(r"(\d+)").astype(float)

    # Separate base vs omics features
    X_base = df[clinical_cols] if clinical_cols else pd.DataFrame(index=df.index)
    feature_cols = [c for c in df.columns if c not in ["sample_id", "response", "age", "stage"]]
    X_main = df[feature_cols]

    return df["sample_id"].values, X_base, X_main, y, feature_cols, clinical_cols


def get_models():
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="liblinear"))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
        )
    }
    if HAVE_XGB:
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=600, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1
        )
    return models


def evaluate_cv(model, X, y, n_splits=5, seed=42, label="model", plot_prefix=None):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    roc_list, pr_list = [], []
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    prs = []
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        roc = roc_auc_score(y[te], p)
        pr  = average_precision_score(y[te], p)
        roc_list.append(roc); pr_list.append(pr)
        fpr, tpr, _ = roc_curve(y[te], p)
        prec, rec, _ = precision_recall_curve(y[te], p)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        prs.append(np.interp(np.linspace(0,1,200), rec[::-1], prec[::-1]))
    roc_mean = np.mean(roc_list); pr_mean = np.mean(pr_list)

    if plot_prefix:
        # ROC
        plt.figure()
        plt.plot([0,1],[0,1], linestyle="--")
        plt.plot(mean_fpr, np.mean(tprs, axis=0))
        plt.title(f"ROC (CV mean) - {label} | AUC={roc_mean:.3f}")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
        plt.savefig(f"{plot_prefix}_roc.png", dpi=160); plt.close()

        # PR
        plt.figure()
        plt.plot(np.linspace(0,1,200), np.mean(prs, axis=0))
        plt.title(f"PR (CV mean) - {label} | AP={pr_mean:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout()
        plt.savefig(f"{plot_prefix}_pr.png", dpi=160); plt.close()

    return roc_mean, pr_mean


def main(args):
    os.makedirs(os.path.join(args.phase3_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.phase3_dir, "models"), exist_ok=True)

    sample_ids, X_base, X_main, y, feature_cols, clinical_cols = load_data(args.phase3_dir)

    perf_rows = []
    results = os.path.join(args.phase3_dir, "results")

    # Baseline clinical model
    if X_base.shape[1] > 0:
        models = get_models()
        for name, mdl in models.items():
            roc, pr = evaluate_cv(
                mdl, X_base.values.astype(float), y,
                n_splits=5, label=f"baseline-{name}",
                plot_prefix=os.path.join(results, f"baseline_{name}")
            )
            perf_rows.append(["baseline", name, roc, pr])

    # Omics features
    models = get_models()
    best_name, best_model, best_roc = None, None, -1
    for name, mdl in models.items():
        roc, pr = evaluate_cv(
            mdl, X_main.values.astype(float), y,
            n_splits=5, label=f"omics-{name}",
            plot_prefix=os.path.join(results, f"omics_{name}")
        )
        perf_rows.append(["omics", name, roc, pr])
        if roc > best_roc:
            best_roc, best_name, best_model = roc, name, mdl

    # Fit best model on full data + calibrate
    cal = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
    cal.fit(X_main.values.astype(float), y)
    joblib.dump({"model": cal, "features": feature_cols},
                os.path.join(args.phase3_dir, "models", "final_model.pkl"))

    # Calibration plot
    prob_pos = cal.predict_proba(X_main.values.astype(float))[:,1]
    frac_pos, mean_pred = calibration_curve(y, prob_pos, n_bins=10)
    plt.figure()
    plt.plot([0,1],[0,1],"--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Calibration - best={best_name} (CV-ROC≈{best_roc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(results, "calibration_plot.png"), dpi=160)
    plt.close()

    # Save performance table
    perf = pd.DataFrame(perf_rows, columns=["feature_set","model","roc_auc","pr_auc"])
    perf.to_csv(os.path.join(results, "model_performance.tsv"), sep="\t", index=False)

    print(f"[OK] Saved performance -> {os.path.join(results, 'model_performance.tsv')}")
    print(f"[OK] Best model: {best_name}  CV-ROC≈{best_roc:.3f}")
    if best_roc >= 0.87:
        print("[GOLD] Target met: ROC-AUC ≥ 0.87")
    else:
        print("[INFO] Target not yet met; we’ll iterate (tuning/embeddings).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase3_dir", default=".", help="Path to 'Phase 3 AI&ML Modeling'")
    args = p.parse_args()
    main(args)