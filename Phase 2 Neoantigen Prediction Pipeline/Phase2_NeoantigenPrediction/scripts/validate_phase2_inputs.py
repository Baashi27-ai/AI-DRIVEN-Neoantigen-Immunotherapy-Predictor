import os, sys, csv, re
from pathlib import Path

BASE = Path.cwd()  # must be Phase 2 folder
inputs = BASE / "inputs"
paths = {
    "somatic": inputs / "variant_info" / "somatic_mutations.tsv",
    "mut_expr": inputs / "variant_info" / "mutations_with_expression.tsv",
    "expr": inputs / "expression" / "expression_tpm.tsv",
    "hla": inputs / "hla" / "hla_types.tsv",
}

def head(path, n=5):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        for i, row in enumerate(rd):
            rows.append(row)
            if i+1 >= n: break
    return rows

def exists_ok(p):
    if not p.exists():
        print(f"[ERROR] Missing file: {p}")
        return False
    print(f"[OK]    Found: {p}")
    return True

def print_head(label, path):
    print(f"\n=== {label} : HEAD (first 5 rows) ===")
    for r in head(path, 5):
        print("\t".join(r))

def infer_columns(rows):
    if not rows: return []
    return rows[0]

def require_columns(label, cols, required):
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"[WARN] {label}: Missing expected columns: {missing}")
    else:
        print(f"[OK]   {label}: Columns present: {required}")

def sanitize_hla(allele: str) -> str:
    a = allele.strip().upper()
    a = a.replace("HLA ", "HLA-").replace("HLA_", "HLA-")
    a = re.sub(r"^HLA([ABC])", r"HLA-\1", a)
    if not a.startswith("HLA-"):
        if re.match(r"^[ABC]\*", a):
            a = "HLA-" + a
    m = re.match(r"^HLA-([ABC])\*(\d{1,2}):(\d{1,2})$", a)
    if m:
        a = f"HLA-{m.group(1)}*{int(m.group(2)):02d}:{int(m.group(3)):02d}"
    return a

def load_hla_list(hla_path):
    alleles = []
    with open(hla_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            for token in re.split(r"[,\t ]+", line):
                if token:
                    alleles.append(sanitize_hla(token))
    seen, uniq = set(), []
    for a in alleles:
        if a not in seen:
            uniq.append(a); seen.add(a)
    return uniq

def write_sanitized_hla(hla_list, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for a in hla_list:
            f.write(a + "\n")
    print(f"[OK]   Wrote sanitized HLA alleles -> {out_path}")

def main():
    print(">>> Phase 2 Input Validation — Start")

    ok = True
    for k, p in paths.items():
        ok = exists_ok(p) and ok
    if not ok:
        print("[FATAL] Some required inputs are missing. Fix and rerun.")
        sys.exit(1)

    som_head = head(paths["somatic"], 6)
    print_head("somatic_mutations.tsv", paths["somatic"])
    som_cols = infer_columns(som_head)
    expected_any = [
        ["sample_id","chrom","pos","ref","alt","gene","VAF"],
        ["sample_id","chromosome","position","reference","alternate","gene","VAF"],
        ["chrom","pos","ref","alt","gene","VAF"],
    ]
    matched = None
    for exp in expected_any:
        if all(c in som_cols for c in exp):
            matched = exp; break
    if matched:
        print(f"[OK]   somatic_mutations.tsv: recognized schema variant -> {matched}")
    else:
        print(f"[WARN] somatic_mutations.tsv: Could not match standard schema. Columns found: {som_cols}")

    if paths["mut_expr"].exists():
        mex_head = head(paths["mut_expr"], 6)
        print_head("mutations_with_expression.tsv", paths["mut_expr"])
        mex_cols = infer_columns(mex_head)
        require_columns("mutations_with_expression.tsv", mex_cols, ["gene","VAF"])
    else:
        print("[INFO] mutations_with_expression.tsv not present; proceeding without this accelerator.")

    expr_head = head(paths["expr"], 6)
    print_head("expression_tpm.tsv", paths["expr"])
    expr_cols = infer_columns(expr_head)
    if "gene" not in expr_cols:
        print(f"[WARN] expression_tpm.tsv: Expected a 'gene' column. Found: {expr_cols}")
    else:
        print("[OK]   expression_tpm.tsv: 'gene' column found.")
    n_samples = len(expr_cols) - (1 if "gene" in expr_cols else 0)
    print(f"[INFO] expression_tpm.tsv: detected {n_samples} expression column(s).")

    print_head("hla_types.tsv (raw)", paths["hla"])
    hla_list = load_hla_list(paths["hla"])
    print(f"[OK]   Parsed {len(hla_list)} HLA alleles after sanitization.")
    out_hla = inputs / "hla" / "hla_types.sanitized.txt"
    write_sanitized_hla(hla_list, out_hla)

    print("\n>>> Next step readiness check:")
    peptide_file = inputs / "peptides" / "derived_peptides.tsv"
    if peptide_file.exists():
        print(f"[OK] Peptide list detected: {peptide_file} -> we can jump to prediction.")
    else:
        has_protein_change = "protein_change" in som_cols or "aa_change" in som_cols
        print(f"[INFO] Peptide list not found. Can we derive from mutations? protein_change present? {has_protein_change}")
        if not has_protein_change:
            print("[NOTE] If no peptide list and no protein_change, we will generate a conservative peptide set later from reference + variant context, or introduce a helper for 8–11mers.")

    print("\n>>> Phase 2 Input Validation — Done")

main()
