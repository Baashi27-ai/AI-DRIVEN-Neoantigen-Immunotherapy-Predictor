cat > scripts/make_phase3_manifest.py << 'PY'
#!/usr/bin/env python3
import os, glob, pandas as pd

def main():
    base = "."
    out_file = os.path.join(base, "phase3_manifest.tsv")

    # Collect results + models
    files = []
    for sub in ["results","models"]:
        for f in sorted(glob.glob(os.path.join(base, sub, "*"))):
            files.append({"category": sub, "file": os.path.relpath(f, base)})

    # Work features
    for f in sorted(glob.glob(os.path.join(base, "work", "features*.tsv"))):
        files.append({"category": "features", "file": os.path.relpath(f, base)})

    # Plots (PNG)
    for f in sorted(glob.glob(os.path.join(base, "results", "*.png"))):
        files.append({"category": "plots", "file": os.path.relpath(f, base)})

    df = pd.DataFrame(files)
    df.to_csv(out_file, sep="\t", index=False)
    print(f"[OK] Wrote manifest with {len(df)} entries -> {out_file}")

if __name__ == "__main__":
    main()
PY