# AI-DRIVEN Neoantigen & Immunotherapy Predictor

*Author:* Bhaskararao Ch (Baashi27-ai) • *Contact:* bhaskarch.1602@gmail.com

## Abstract
We develop an AI-driven pipeline to predict patient-specific neoantigens and immunotherapy response with explainability. The workflow integrates somatic mutations, RNA expression, HLA typing, and immune microenvironment features, producing a clinician-facing dashboard and reproducible artifacts.

## Repository Scope
This repository is organized *phase-wise*. Each phase is a clean, auditable commit with scripts, manifests, and compact result tables (large raw data are intentionally ignored).

### Phase Checklist
- *Phase 1 – Data Foundation (DONE)*  
  - Somatic mutation table (Hugo2016 S1D parsed)  
  - RNA expression (FPKM→TPM) & metadata curation  
  - HLA typing via OptiType (paired-end, relaxed)  
  - Immune infiltration (template/placeholder)  
  - Manifest logging for reproducibility  
- *Phase 2 – Neoantigen Prediction (NEXT)*  
  - NetMHCpan / MHCflurry binding & presentation  
  - Filtering by VAF, expression; QC plots  
- *Phase 3 – AI Modeling* (response prediction, embeddings, SHAP)  
- *Phase 4 – Explainability & Clinical Mapping*  
- *Phase 5 – Dashboard & External Validation*  
- *Phase 6 – Packaging & Publication*

## Directory Layout
Neo_Antigen_Moonshot/ ├─ Phase1_DataFoundation/ │  ├─ metadata/                     # curated labels, patient metadata │  ├─ scripts/                      # python/R scripts used in Phase 1 │  ├─ results/                      # compact TSV results (tracked) │  ├─ downloads/ raw/ hla_runs/ …   # large assets (ignored by git) │  └─ phase1_manifest.tsv           # provenance log (who/what/when) ├─ figures/                         # (optional) curated plots for paper ├─ docs/                            # (optional) method notes, SOPs ├─ README.md └─ .gitignore

## Phase 1 Key Artifacts (tracked)
- Phase1_DataFoundation/results/hla_types.tsv – HLA genotypes  
- Phase1_DataFoundation/metadata/patient_metadata.tsv – sample linkage  
- Phase1_DataFoundation/metadata/gse78220_labels.tsv – response labels  
- Phase1_DataFoundation/results/somatic_mutations.tsv – (MAF-like minimal)  
- Phase1_DataFoundation/phase1_manifest.tsv – actions & timestamps

> Large raw/intermediate files (FASTQ/BAM/VCF, full optitype runs) are ignored by design. Recreate via scripts when needed.

## Reproducibility (Phase 1)
Example (inside WSL):
bash
# Expression conversion
python Phase1_DataFoundation/scripts/convert_fpkm_to_tsv.py
python Phase1_DataFoundation/scripts/fpkm_to_tpm.py

# Parse Hugo2016 mutations
python Phase1_DataFoundation/scripts/parse_hugo_mutations.py

# HLA typing (OptiType; see manifest for exact commands/versions)
# Results summarized to: Phase1_DataFoundation/results/hla_types.tsv

Citation

If you use this code, please cite:

> Bhaskararao Ch (2025). AI-DRIVEN Neoantigen & Immunotherapy Predictor.
https://github.com/Baashi27-ai/AI-DRIVEN-Neoantigen-Immunotherapy-Predictor



License

TBD (MIT recommended). EOF

---

### 3) (Optional) Add a LICENSE (MIT)
bash
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2025 Bhaskararao Ch

Permission is hereby granted, free of charge, to any person obtaining a copy
... (standard MIT text continues) ...
