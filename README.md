# AI-DRIVEN Neoantigen & Immunotherapy Predictor

🚀 *Precision Immuno-Oncology with Multi-Phase Pipeline*  
This repository hosts a complete AI-driven pipeline for *neoantigen discovery and immunotherapy prediction, structured in 3 phases.

## 🌟 Highlights
- End-to-end *data foundation → neoantigen prediction → AI/ML modeling*
- Phase-wise modular design for clarity & reproducibility
- Integration of *omics data, MHC binding, and AI models*
- Outputs include *QC plots, candidate tables, SHAP explainability, and final models*

---

## 📂 Repository Structure

├── Phase1_DataFoundation/            # Input foundation (VCF, HLA typing, expression) ├── Phase 2 Neoantigen Prediction Pipeline/ │   ├── scripts/                      # Prediction + QC scripts │   └── qc_plots/                     # QC plots ├── Phase 3 AI&ML Modeling/ │   ├── scripts/                      # ML training & explainability │   ├── results/                      # ROC/PR, SHAP, performance logs │   └── models/                       # Saved trained models └── figures/                          # Global figures (pipeline schematic, etc.)

---

## 🔑 Phase Overview
- *[Phase 1: Data Foundation](./Phase1_DataFoundation/README.md)* → Builds foundation (VCF, HLA, expression).  
- *[Phase 2: Neoantigen Prediction](./Phase%202%20Neoantigen%20Prediction%20Pipeline/README.md)* → Predicts candidate peptides with QC.  
- *[Phase 3: AI & ML Modeling](./Phase%203%20AI%26ML%20Modeling/README.md)* → Trains ML models, generates SHAP plots & final predictor.  

---

## 📊 Expected Outputs
- *Phase 1* → Cleaned inputs, HLA results, QC plots  
- *Phase 2* → neoantigen_candidates.tsv, QC plots, filtering flowchart  
- *Phase 3* → ROC/PR plots, SHAP explainability, final model .pkl, manifest TSV  

---

## 🛠 Tech Stack
- Python (pandas, scikit-learn, xgboost, lightgbm, shap)  
- R (QC plots, statistical validation)  
- GROMACS / NetMHCpan / MHCflurry (for peptide-HLA binding)  

---

## 📜 License
MIT License – free to use with attribution.  

---


---

📌 Phase1_DataFoundation/README.md

# Phase 1: Data Foundation  

## 🎯 Goal
Prepare the foundation for neoantigen prediction:  
- Variant calling format (VCF)  
- HLA typing results  
- Expression quantification  

## 📂 Inputs
- vcf/ → Somatic mutations  
- hla_results/ → HLA typing outputs  
- expression/ → RNA-seq expression data  

## ⚙ Steps
1. Collect input data (VCF, HLA, RNA-seq).  
2. Run QC checks (coverage, allele fraction).  
3. Format inputs into standardized tables.  

## 📊 Outputs
- labels.tsv → Sample labels template  
- QC plots in qc/  
- Foundation datasets for Phase 2  

---


---

📌 Phase 2 Neoantigen Prediction Pipeline/README.md

# Phase 2: Neoantigen Prediction Pipeline  

## 🎯 Goal
Predict candidate *neoantigen peptides* and filter them using binding, expression, and QC metrics.  

## 📂 Inputs
- From *Phase 1 foundation* (vcf/, hla_results/, expression/)  

## ⚙ Steps
1. Predict peptide–MHC binding (NetMHCpan-EL, MHCflurry).  
2. Filter peptides with thresholds:  
   - Binding < 500 nM  
   - Variant allele frequency (VAF) confirmed  
   - Expression confirmed  
3. Generate QC plots:  
   - Allele fraction vs expression  
   - Peptide filtering flowchart  
4. (Optional) Validate expression in scRNA-seq  

## 📊 Outputs
- neoantigen_candidates.tsv  
- binding_affinity_histogram.png  
- peptide_length_distribution.png  
- peptide_filtering_flowchart.png  
- (Optional) scRNA_validation_plots.png  

---


---

📌 Phase 3 AI&ML Modeling/README.md

# Phase 3: AI & ML Modeling  

## 🎯 Goal
Develop machine learning models for predicting immunogenicity of neoantigens, with explainability & validation.  

## 📂 Inputs
- Features from *Phase 2* (work/features.tsv, embeddings, Top-K files)  

## ⚙ Steps
1. Feature preparation (binder features, Top-K compact features, embeddings).  
2. Train models:  
   - XGBoost fine-tuned  
   - LightGBM fine-tuned  
   - Ensemble (stacked)  
3. Evaluate metrics: ROC-AUC, Average Precision.  
4. Generate *SHAP plots* for explainability.  
5. Save final model + manifest.  

## 📊 Outputs
- results/model_performance.tsv  
- results/roc_curve.png, results/pr_curve.png  
- results/shap_summary.png, results/shap_top10.png  
- models/final_model.pkl  
- phase3_manifest.tsv  

---


---
