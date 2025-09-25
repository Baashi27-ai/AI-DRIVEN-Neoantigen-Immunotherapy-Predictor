import pandas as pd, os
feat = pd.read_csv("work/features.tsv", sep="\t")
feat = feat.rename(columns={feat.columns[0]: "sample_id"})
tpl = pd.DataFrame({
    "sample_id": feat["sample_id"],
    # Set 0/1 for your outcome (e.g., responder=1, non-responder=0)
    "response": 0,
    # Optional clinical covariates for baseline model; fill if you have them
    "age": pd.NA,
    "stage": pd.NA
})
os.makedirs("inputs", exist_ok=True)
out = "inputs/labels_template.tsv"
tpl.to_csv(out, sep="\t", index=False)
print(f"[OK] Wrote template -> {out}")
