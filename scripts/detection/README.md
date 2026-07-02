# Detection Scripts

These scripts handle data preparation, inference, submission formatting, and visualisation for the detection task. The detector itself (RT-DETRv2) lives in `models/detection/rtdetrv2/`.

They are written to be explicit, reproducible, easy to inspect, and compatible with the ULHPC environment.

## Scripts

`convert_spark_to_coco.py`
Converts the SPARK CSV annotations (`train.csv`, `val.csv`) into COCO format JSON.

Outputs:

```
data/annotations/spark_train.json
data/annotations/spark_val.json
```

`detection_dataset_loader.py`
PyTorch dataset used to load images and their COCO annotations.

`convert_predictions_to_submission.py`
Turns the prediction JSON into the CSV format expected by Codabench. Detections are kept above a confidence threshold of 0.30.

`visualize_detection_results.py`
Draws the predicted bounding boxes and class labels on the images for inspection and reporting.
