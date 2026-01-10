# reports/

Everything related to report generation (plots, tables, metrics).

## Structure

- `scripts/` : report scripts (detection / segmentation / common helpers)
- `jobs/`    : SLURM helpers to run report scripts on ULHPC
- `outputs/` :
  - `outputs/latest/figures/` : latest plots used in the report
  - `outputs/latest/tables/`  : latest tables/metrics used in the report
  - `outputs/archive/`        : optional snapshots of older report outputs

## Typical workflow

1) Run inference (detection / segmentation) to produce raw predictions in:
- `inference_results/detection/`
- `inference_results/segmentation/`

2) Generate report material (writes to `reports/outputs/latest/`):
- Detection quick plots/tables:
  - `python -u reports/scripts/detection/detection_report_plots.py --score_thr 0.30`
  - `python -u reports/scripts/detection/detection_report_extras.py --score_thr 0.30`
- Segmentation quick plots/tables:
  - `python -u reports/scripts/segmentation/segmentation_report_plots.py`
  - `python -u reports/scripts/segmentation/segmentation_report_extras.py`
- Segmentation VAL metrics (requires GT masks and predicted masks):
  - `python -u reports/scripts/segmentation/segmentation_val_metrics.py --gt_dir <GT_DIR> --pred_dir <PRED_DIR>`
