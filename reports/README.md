# Reports

This folder contains **analysis and reporting utilities** used to evaluate, summarize, and visualize the outputs of the detection and segmentation pipelines.

The code in this directory is **not required to run training or inference**.  
It is used for:
- quantitative evaluation
- qualitative visualization
- report / presentation material generation

---

## Purpose

The `reports/` folder serves three main purposes:

1. **Quantitative analysis**  
   - score distributions  
   - class-wise statistics  
   - confidence thresholds

2. **Qualitative analysis**  
   - visualization of predictions  
   - sample inspection for reports

3. **Submission verification**  
   - sanity checks on prediction formats  
   - validation of Codabench-ready outputs

---

## Folder Structure

```text
reports/
├── figures/        # Generated figures (plots, histograms, examples)
├── tables/         # Generated tables (CSV / LaTeX-ready metrics)
├── metrics/        # Aggregated evaluation results
├── detection_*     # Detection-specific reporting scripts
└── segmentation_* # Segmentation-specific reporting scripts
