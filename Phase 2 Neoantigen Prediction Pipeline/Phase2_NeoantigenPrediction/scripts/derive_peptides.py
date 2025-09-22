import re, csv
from pathlib import Path

BASE = Path.cwd()
inp = BASE / "inputs" / "variant_info" / "somatic_mutations.tsv"
expr = BASE / "inputs" / "expression" / "expression_tpm.tsv"
out_pep = BASE / "inputs" / "peptides" / "derived_peptides.tsv"
out_pep.parent.mkdir(parents=True, exist_ok=True)

def read_expression(path):
    # Map Gene -> TPM for Pt1.baseline if present, else first expression column after Gene/gene
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        hdr = next(rd)
        gene_col = hdr.index("Gene") if "Gene" in hdr else (hdr.index("gene") if "gene" in hdr else 0)
        if "Pt1.baseline" in hdr:
            exp_col = hdr.index("Pt1.baseline")
        else:
            exp_col = gene_col + 1 if len(hdr) > gene_col + 1 else gene_col
        m = {}
        for row in rd:
            if not row: continue
            g = row[gene_col]
            try:
                tpm = float(row[exp_col])
            except:
                tpm = 0.0
            m[g] = tpm
        return m

def aa_window(mut_aa, flank=15):
    # Keep letters only, drop prefixes like 'p.'; keep a 31-aa window (best effort)
    s = re.sub(r"[^A-Za-z]", "", str(mut_aa)).upper()
    return s[: (2*flank+1)] if s else ""

def kmerize(seq, kmin=8, kmax=11):
    out = []
    for k in range(kmin, kmax+1):
        for i in range(0, max(0, len(seq)-k+1)):
            out.append(seq[i:i+k])
    return out

expr_map = read_expression(expr)

rows = []
with open(inp, "r", encoding="utf-8") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        gene = r.get("gene_symbol") or r.get("gene") or ""
        aa = r.get("protein_change") or r.get("aa_change") or ""
        sample = r.get("sample_id") or ""
        if not gene or not aa:
            continue
        win = aa_window(aa, 15)
        if not win:
            continue
        peps = kmerize(win, 8, 11)
        tpm = expr_map.get(gene, 0.0)
        for p in peps:
            rows.append((sample, gene, aa, p, len(p), tpm))

with open(out_pep, "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f, delimiter="\t")
    wr.writerow(["sample_id","gene","protein_change","peptide","length","expression_tpm"])
    wr.writerows(rows)

print(f"[OK] wrote peptides -> {out_pep}  (n={len(rows)})")
