# Reports folder

## Where to find results
- **Latest run (source of truth):**
  - `reports/outputs/latest/figures/`
  - `reports/outputs/latest/tables/`

- **Final hand-in package (copy of latest):**
  - `reports/final_pack/figures/`
  - `reports/final_pack/tables/`

## Re-running report generation
Run scripts from repo root using `python -m ...`:
- Segmentation metrics:
  - `python -m reports.scripts.segmentation.segmentation_val_metrics ...`
- Detection metrics/plots:
  - `python -m reports.scripts.detection.detection_val_pr_f1 ...`
  - `python -m reports.scripts.detection.detection_val_threshold_sweep ...`
