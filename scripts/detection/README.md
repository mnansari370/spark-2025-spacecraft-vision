# Detection Scripts

This directory contains all **detection-related scripts** used for data preparation,
inference, visualization, and submission formatting for the SPARK detection task.

The detection model itself (RT-DETRv2) lives in:
`models/detection/rtdetrv2/`.

---

## Purpose of this folder

The scripts here are responsible for:
- converting dataset annotations
- running detection inference
- preparing Codabench submissions
- visualizing detection predictions

They are written to be:
- explicit
- reproducible
- compatible with ULHPC environments
- easy to inspect and debug

---

## Scripts Overview

### Dataset & annotation utilities

- **`convert_spark_to_coco.py`**  
  Converts SPARK CSV annotations (`train.csv`, `val.csv`) into COCO-format JSON files.

Outputs:
```text
data/annotations/spark_train.json
data/annotations/spark_val.json
