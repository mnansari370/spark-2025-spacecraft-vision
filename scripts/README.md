# Scripts — Inference, Conversion & Visualization

This folder contains **reusable Python scripts** used across the project for:
- running inference
- converting outputs to challenge formats
- visualizing results for analysis and reporting

All scripts are **path-robust**: they automatically resolve the repository root and can be executed from any working directory.

---

## Folder structure

```text
scripts/
├── detection/
│   ├── convert_predictions_to_submission.py
│   ├── convert_spark_to_coco.py
│   ├── detection_dataset_loader.py
│   └── visualize_detection_results.py
│
├── segmentation/
│   ├── infer_segmentation.py
│   ├── convert_png_to_npz_submission.py
│   └── visualize_segmentation_results.py
