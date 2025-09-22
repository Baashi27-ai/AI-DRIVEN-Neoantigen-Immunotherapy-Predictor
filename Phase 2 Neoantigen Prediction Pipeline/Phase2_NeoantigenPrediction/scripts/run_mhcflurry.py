#!/usr/bin/env python3
import subprocess
from pathlib import Path
import shutil

def main():
    base = Path(__file__).resolve().parent.parent
    work = base / "work"
    pairs_csv = work / "pairs.csv"
    out_bind = work / "mhcflurry_binding.tsv"
    out_pres = work / "mhcflurry_presentation.tsv"  # compatibility target

    # 1) Binding (includes processing & presentation columns in v2.0.6)
    cmd_bind = [
        "mhcflurry-predict",
        "--allele-column", "allele",
        "--peptide-column", "peptide",
        "--no-flanking",
        "--out", str(out_bind),
        str(pairs_csv)
    ]
    print("[RUN]", " ".join(cmd_bind))
    subprocess.check_call(cmd_bind)
    print(f"[OK] Binding -> {out_bind}")

    # 2) For downstream compatibility, mirror to presentation filename
    shutil.copy2(out_bind, out_pres)
    print(f"[OK] Presentation (copied from binding) -> {out_pres}")

if __name__ == "__main__":
    main()