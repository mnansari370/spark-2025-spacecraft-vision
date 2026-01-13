# Run Scripts

This folder contains scripts that contains the parameters used to execute the main stages of the project pipeline.

## Detection

- `detection_infer.py`  
  Runs **RT-DETRv2 detection inference on the test set** and generates prediction JSON files.

- `detection_infer_val.py`  
  Runs **detection inference on the validation set** using COCO annotations (used for validation and analysis).

---

## Segmentation

- `segmentation_infer.py`  
  Runs **DeepLabV3+ segmentation inference on the test set**, producing PNG mask predictions.

- `segmentation_infer_val.py`  
  Runs **segmentation inference on the validation set**, preserving the mission / split directory structure.

- `segmentation_make_submission.py`  
  Converts predicted segmentation masks into the **final Codabench submission format** (`.npz` → `.zip`).

---

All scripts are designed to be executed **from the repository root** using the slurm script:

```bash
python run/<script_name>.py

