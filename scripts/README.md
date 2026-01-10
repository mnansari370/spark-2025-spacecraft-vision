# SPARK 2025 — Spacecraft Vision (Detection + Segmentation)

This repository contains my solution for the SPARK spacecraft vision challenge:
- **Task 1: Detection** (RT-DETRv2)
- **Task 2: Segmentation** (DeepLabV3+)

Large assets (datasets, checkpoints, inference outputs, logs) are stored on scratch on ULHPC and linked into the repo.

---

## Repository structure

- `run/`  
  Entry-point scripts (the commands you execute).

- `scripts/`  
  Reusable scripts and utilities (conversion, inference logic, visualization).

- `models/`  
  Model code:
  - `models/detection/rtdetrv2/` (third-party RT-DETRv2 code, kept unchanged)
  - `models/segmentation/` (my segmentation model factory)

- `jobs/`  
  SLURM job scripts (train / inference / submission building).

- `reports/`  
  Report generation code (plots, tables, metrics).  
  Output figures/tables are not required for running the pipeline.

---

## Data & checkpoints

On ULHPC, the following folders are linked to scratch:
- `data/`
- `checkpoints/`
- `inference_results/`
- `wandb/`

The code assumes these directories exist at repo root.

---

## Environments

Detection environment (RT-DETRv2):
- conda env: `spark_rtdetr`

Segmentation environment (DeepLabV3+):
- conda env: `spark_seg`

---

## Running Detection

### Test inference (creates JSON predictions)
```bash
python run/detection_infer.py --device cuda --limit 0 --score_thr 0.30
