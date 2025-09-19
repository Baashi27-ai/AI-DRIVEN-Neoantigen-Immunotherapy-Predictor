# Phase 1 – Data Foundation (Methods & Notes)

## Metadata
- metadata/patient_metadata.tsv – sample → patient mapping
- metadata/gse78220_labels.tsv – clinical response annotations

---

## Expression Processing
Scripts (scripts/):
- convert_fpkm_to_tsv.py / .bak – reshape FPKM data
- fpkm_to_tpm.py – convert FPKM → TPM
- confirm_mutation_expression.py – ensure mutations are expressed

---

## HLA Typing

### Cleaning Reads & Typing
bash
fastp \
  -i sample1_R1.fastq.gz -I sample1_R2.fastq.gz \
  -o clean_R1.fq.gz -O clean_R2.fq.gz \
  --detect_adapter_for_pe --qualified_quality_phred 15 --length_required 50 \
  --thread 4 --html fastp_report.html --json fastp_report.json


### OptiType (standard)
bash
OptiTypePipeline.py \
  -i clean_R1.fq.gz clean_R2.fq.gz \
  -o results/optitype_sample1_full -p sample1 --dna -v


### OptiType (relaxed config)
bash
OptiTypePipeline.py \
  -i clean_R1.fq.gz clean_R2.fq.gz \
  -o results/optitype_sample1_relaxed -p sample1 --dna \
  -c results/optitype_relaxed.ini -v

