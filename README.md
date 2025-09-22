# AI-DRIVEN Neoantigen & Immunotherapy Predictor

## Overview
This repository contains a multi-phase pipeline for predicting patient-specific neoantigens and modeling immunotherapy response. It integrates HLA typing, mutation–expression linkage, RNA-seq processing, immune infiltration quantification, and peptide–MHC prediction into a reproducible framework.

---

## Repository Structure
AI-DRIVEN-Neoantigen-Immunotherapy-Predictor/ ├── Phase1_DataFoundation/              # Core data + scripts for Phase 1 │   ├── metadata/                       # Patient and clinical annotations │   ├── results/                        # Processed outputs (HLA typing, mutations, TPMs) │   ├── scripts/                        # Python and R utilities for Phase 1 │   └── phase1_manifest.tsv             # Log of all Phase 1 outputs ├── Phase 2 Neoantigen Prediction Pipeline/ │   ├── inputs/                         # Variant info, expression, HLA │   ├── work/                           # Intermediates + raw predictions │   ├── results/                        # Final Phase 2 deliverables │   └── scripts/                        # Phase 2 scripts ├── docs/ │   └── PHASE1_NOTES.md                 # SOP notes for Phase 1 └── README.md                           # (this file)

---
---

## Phase 1: Data Foundation
*Metadata curation*
- metadata/patient_metadata.tsv
- metadata/gse78220_labels.tsv

*Expression processing*
- Convert RNA-seq FPKM → TPM
- Ensure mutation–expression linkage

*HLA typing*
- OptiType (standard & relaxed) on paired-end RNA-seq

*Outputs*
- results/hla_types.tsv – HLA types per patient  
- results/mutations_with_expression.tsv – expressed mutations  
- results/immune_infiltration_scores.tsv – infiltration estimates  
- phase1_manifest.tsv – reproducibility log  

📄 Detailed SOP: docs/PHASE1_NOTES.md

---

## Phase 2: Neoantigen Prediction Pipeline
*Goal:* Derive patient-specific neoantigen candidates using peptide–MHC binding and presentation models, then apply biological filters and generate QC outputs.

### Inputs
- variant_info/somatic_mutations.tsv  
- variant_info/mutations_with_expression.tsv  
- expression/expression_tpm.tsv  
- hla/hla_types.tsv  

### Method
1. *Peptide generation* (8–11mers from expressed mutations).  
2. *Prediction* with *MHCflurry* (affinity, processing, presentation scores).  
3. *Filtering* by:
   - Binding < 500 nM  
   - VAF threshold (if available)  
   - Expression confirmed  
4. *QC outputs*:
   - Binding affinity histogram  
   - Peptide length distribution  
   - Filtering flowchart  

### Outputs
- results/neoantigen_candidates.tsv  
- results/binding_affinity_histogram.png  
- results/peptide_length_distribution.png  
- results/peptide_filtering_flowchart.png  

---

## Requirements
*Python*
- Python ≥ 3.9  
- Packages: pandas, numpy, matplotlib, graphviz  
- Specialized: mhcflurry (v2.x + downloaded models)  

*R*
- R ≥ 4.0  
- Packages: ggplot2, data.table  

---

## License
MIT License.

## Citation
If you use this pipeline in academic work, please cite:

> Bhaskararao Ch (Baashi27-ai), *AI-DRIVEN Neoantigen & Immunotherapy Predictor*, 2025.
