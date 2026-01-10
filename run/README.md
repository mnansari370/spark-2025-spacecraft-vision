# Run Scripts

This folder contains entry-point scripts used to execute the project pipeline.

These scripts define *what* is run (test / validation / submission),
while the implementation logic lives in `scripts/` and `models/`.

## Detection
- `detection_infer.py`  
  Runs detection inference on the test set.

- `detection_infer_val.py`  
  Runs detection inference on the validation set using COCO annotations.

## Segmentation
- `segmentation_infer.py`  
  Runs segmentation inference on the test set.

- `segmentation_infer_val.py`  
  Runs segmentation inference on the validation set.

- `segmentation_make_submission.py`  
  Converts predicted masks into the final submission format.

All scripts are designed to be executed from the repository root:
```bash
python run/<script_name>.py
