# Scripts

Utility scripts for data preparation, inference, submission formatting, and visualisation. The model code itself lives under `models/`.

```
scripts/
├── detection/
│   ├── convert_spark_to_coco.py            # CSV to COCO JSON
│   ├── detection_dataset_loader.py         # PyTorch dataset
│   ├── convert_predictions_to_submission.py # JSON to CSV submission
│   └── visualize_detection_results.py
└── segmentation/
    ├── infer_segmentation.py               # inference
    ├── convert_png_to_npz_submission.py    # PNG to NPZ to ZIP
    └── visualize_segmentation_results.py
```

See the README in each sub-folder for more detail.
