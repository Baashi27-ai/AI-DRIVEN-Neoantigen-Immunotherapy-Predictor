# AI-DRIVEN Neoantigen & Immunotherapy Predictor

*Author:* Bhaskararao Ch (Baashi27-ai) • *Contact:* bhaskarch.1602@gmail.com

## Abstract
This project develops an AI-driven pipeline to predict patient-specific neoantigens and immunotherapy response.  
The workflow integrates somatic mutations, RNA expression, HLA typing, and immune infiltration, producing reproducible results and publication-ready artifacts.

## Phases
- *Phase 1 – Data Foundation (DONE)*  
  - Mutation parsing, RNA-seq TPM conversion  
  - HLA typing (OptiType full + relaxed)  
  - Metadata curation & manifest logging
- *Phase 2 – Neoantigen Prediction (NEXT)*  
  - NetMHCpan / MHCflurry binding, filtering, QC plots
- *Phase 3 – AI Modeling*  
- *Phase 4 – Explainability & Clinical Mapping*  
- *Phase 5 – Dashboard & Validation*  
- *Phase 6 – Packaging & Publication*

## Directory Layout
Neo_Antigen_Moonshot/ ├─ Phase1_DataFoundation/ │  ├─ metadata/                     # patient labels, metadata │  ├─ scripts/                      # python/R scripts for Phase 1 │  ├─ results/                      # compact TSV results (tracked) │  └─ phase1_manifest.tsv           # provenance log ├─ figures/                         # curated plots ├─ README.md └─ .gitignore
## Phase 1 Key Artifacts
- results/hla_types.tsv – HLA genotypes  
- metadata/patient_metadata.tsv – patient linkage  
- metadata/gse78220_labels.tsv – response labels  
- results/somatic_mutations.tsv – compact mutation table  
- phase1_manifest.tsv – log of commands & outputs

## Citation
Bhaskararao Ch (2025). AI-DRIVEN Neoantigen & Immunotherapy Predictor.  
https://github.com/Baashi27-ai/AI-DRIVEN-Neoantigen-Immunotherapy-Predictor

## License
MIT (see LICENSE)
