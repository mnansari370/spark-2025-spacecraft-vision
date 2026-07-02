# Run Scripts

These scripts hold the parameters used to execute the main stages of the pipeline. They are meant to be called from the repository root, usually through the SLURM job scripts in `jobs/`.

## Detection

`detection_infer.py`
Runs RT-DETRv2 inference on the test set and writes the prediction JSON.

`detection_infer_val.py`
Runs detection inference on the validation set using the COCO annotations. Used for validation and analysis.

## Segmentation

`segmentation_make_submission.py`
Converts the predicted segmentation masks into the final Codabench format (`.npz` then `.zip`).

## Usage

Run any of these from the repository root, for example:

```bash
python run/detection_infer.py
```
