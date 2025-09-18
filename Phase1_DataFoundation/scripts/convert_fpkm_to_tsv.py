import pandas as pd
from pathlib import Path

# Run this from the scripts/ folder. We go one level up to Phase1_DataFoundation/
infile = Path("../raw/GSE78220_PatientFPKM.xlsx")
outfile = Path("../expression/expression_matrix.tsv")

print(f"Reading: {infile.resolve()}")
df = pd.read_excel(infile, engine="openpyxl")

# tidy column names
df.columns = [str(c).strip() for c in df.columns]

# ensure output folder exists and write TSV
outfile.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(outfile, sep="\t", index=False)
print(f"Saved TSV -> {outfile.resolve()}")
