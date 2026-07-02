# Segmentation Scripts

These scripts handle inference, submission preparation, and visualisation for the segmentation task. They do not define the model; that lives in `models/segmentation/model_factory.py`.

They are written to run from the repository root, work on CPU or GPU, and stay robust to different directory layouts.

## Scripts

`infer_segmentation.py`
Runs segmentation inference for the `test`, `val`, or `train` split.

Features:

* optional horizontal flip test time augmentation
* CPU or GPU support
* preserves the mission and split directory structure for val and train
* writes PNG masks to `inference_results/segmentation/`

Example:

```bash
python scripts/segmentation/infer_segmentation.py --split test --device cuda
```

`convert_png_to_npz_submission.py`
Converts the predicted PNG masks into the `.npz` format and zips them for Codabench.

`visualize_segmentation_results.py`
Overlays the predicted masks on the images for inspection and reporting.
