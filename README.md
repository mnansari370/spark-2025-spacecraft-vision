# SPARK 2025 – Spacecraft Detection & Segmentation (CVIA Project)

This repository contains my end-to-end pipeline for **spacecraft detection** and **spacecraft part segmentation** developed for the **CVIA course project / SPARK challenge** at the University of Luxembourg.

It is structured to be **HPC-friendly (SLURM/ULHPC)** and to produce **Codabench-ready submissions** for:
- **Detection** (multi-class bounding boxes)
- **Segmentation** (spacecraft body + solar panels)

---

## Repository Structure

The full repository structure (Level-3) is documented in:

- `PROJECT_TREE_L3.txt`

Key folders:
- `scripts/` – preprocessing, conversion, inference utilities
- `models/` – model code and weights (segmentation weights tracked via Git LFS)
- `jobs/` – SLURM job scripts (ULHPC)
- `logs/` – SLURM outputs (cleaned)
- `inference_results/` – outputs + Codabench submissions

---

## What is tracked vs local-only

### Tracked in GitHub
- Code (`scripts/`, `models/`, `jobs/`)
- Submission artifacts (`inference_results/**/submission_*.zip`, prediction CSV/JSON)
- Segmentation weights (Git LFS):
  - `models/segmentation/deeplabv3plus_baseline/best.pth`
  - `models/segmentation/deeplabv3plus_baseline/last.pth`
- SLURM logs (cleaned): `logs/**/slurm/*.out`

### Not tracked (must exist locally)
- `data/` (datasets)
- training checkpoints under `checkpoints/` (optional local output)
- `wandb/` experiment runs (optional local tracking)
- `_TRASH_DO_NOT_DELETE_YET/` (temporary local folder)

---

## Models

### Detection
- Model: **RT-DETRv2**
- Location: `models/detection/rtdetrv2/`
- Utilities: `scripts/detection/`
- Outputs: `inference_results/detection/`

### Segmentation
- Model: **DeepLabV3+ baseline**
- Location: `models/segmentation/deeplabv3plus_baseline/`
- Weights:
  - `best.pth` (Git LFS)
  - `last.pth` (Git LFS)
- Utilities: `scripts/segmentation/`
- Outputs: `inference_results/segmentation/`

---

## Dataset Layout (local)

Expected local layout (not pushed to GitHub):

```text
data/
├── annotations/
│   ├── spark_train.json
│   └── spark_val.json
├── spark-2024-train-val/
│   ├── images/
│   ├── mask/
│   ├── train.csv
│   └── val.csv
├── spark-2024-detection-test/images/
└── spark-2024-segmentation-test/stream-1-test/
