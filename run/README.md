# run/

Entry-point scripts (thin wrappers) for the project.

- detection_infer.py: run inference on detection test set and write JSON predictions
- detection_infer_val.py: run inference on validation set (for metrics/plots)

- segmentation_infer.py: run segmentation inference (calls scripts/segmentation/run_segmentation_inference.py)
- segmentation_infer_val.py: run segmentation on val split (for metrics/plots)

- segmentation_make_submission.py: converts predicted PNG masks into Codabench-ready NPZ zip
