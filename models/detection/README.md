# Detection Models

## RT-DETRv2 (third-party)
The folder `models/detection/rtdetrv2/` contains third-party code (RT-DETRv2).
We keep it unchanged and only use it via:

- configs: `models/detection/rtdetrv2/configs/...`
- training entrypoint: `models/detection/rtdetrv2/tools/train.py`
- project wrappers: `run/detection_infer.py`, `run/detection_infer_val.py`
