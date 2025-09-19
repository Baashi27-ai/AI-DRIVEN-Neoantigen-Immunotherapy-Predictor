# AI-DRIVEN Neoantigen & Immunotherapy Predictor

## Overview
This repository contains a multi-phase pipeline for predicting patient-specific neoantigens and guiding immunotherapy response modeling.  
It integrates *HLA typing, mutation-expression linkage, RNA-seq processing, and immune infiltration analysis* into a reproducible framework.

---

## Repository Structure
AI-DRIVEN-Neoantigen-Immunotherapy-Predictor/ ├── Phase1_DataFoundation/       # Core data + scripts for Phase 1 │   ├── metadata/                # Patient and clinical annotations │   ├── results/                 # Processed outputs (HLA typing, mutations, TPMs) │   ├── scripts/                 # Python and R utilities for Phase 1 │   └── phase1_manifest.tsv      # Log of all Phase 1 outputs ├── docs/                        # Documentation and methods │   └── PHASE1_NOTES.md          # SOP notes for Phase 1 ├── .gitignore                   # Ignore raw/large files └── README.md                    # Project overview (this file)
---

## Phase 1: Data Foundation
- *Metadata curation*
  - metadata/patient_metadata.tsv
  - metadata/gse78220_labels.tsv
- *Expression processing*
  - Convert RNA-seq FPKM → TPM
  - Ensure mutation-expression linkage
- *HLA typing*
  - OptiType standard & relaxed runs on paired-end RNA-seq
- *Outputs*
  - results/hla_types.tsv – HLA types per patient
  - results/mutations_with_expression.tsv – expressed mutations
  - results/immune_infiltration_scores.tsv – infiltration estimates
  - phase1_manifest.tsv – reproducibility log

📄 Detailed SOP: [docs/PHASE1_NOTES.md](docs/PHASE1_NOTES.md)

---

## Future Phases
- *Phase 2:* Neoantigen candidate prediction  
- *Phase 3:* Immunogenicity scoring + integration with immune infiltration  
- *Phase 4:* AI-driven predictive modeling  

---

## Requirements
### Python
- Python ≥3.9  
- Packages: pandas, numpy, pyomo, biopython, matplotlib  
- Specialized tools: [OptiType](https://github.com/FRED-2/OptiType), fastp  

### R
- R ≥4.0  
- Packages: ggplot2, data.table  

(See requirements.txt to reproduce the environment.)

---

## License
This project is released under the [MIT License](LICENSE).

---

## Citation
If you use this pipeline in academic work, please cite:  
*Bhaskararao Ch (Baashi27-ai), *AI-DRIVEN Neoantigen & Immunotherapy Predictor, 2025.
