# Segmentation Scripts

This directory contains all **segmentation-related scripts** used for inference,
visualization, and submission preparation for the SPARK spacecraft segmentation task.

These scripts **do not define the model itself**.
They use the segmentation model defined in:
`models/segmentation/model_factory.py`.

---

## Purpose of this folder

The scripts here handle:
- running segmentation inference (test / validation)
- converting predicted masks into Codabench format
- visualizing segmentation outputs for inspection and reporting

They are written to be:
- reproducible
- runnable from the repository root
- compatible with CPU and GPU execution
- robust to different directory layouts

---

## Scripts Overview

### Inference

- **`infer_segmentation.py`**  
  Runs segmentation inference for:
  - `test`
  - `val`
  - `train` (optional)

  Features:
  - optional flip TTA
  - CPU / GPU support
  - preserves mission/split directory structure for val/train
  - writes PNG masks to `inference_results/segmentation/`

Example:
```bash
python scripts/segmentation/infer_segmentation.py --split test --device cuda
