# Detection Model

This directory holds all the code and configuration for spacecraft detection.

The task is multi class object detection: each image may contain one spacecraft belonging to one of the ten SPARK classes, and the model predicts its bounding box and class.

## RT-DETRv2

RT-DETRv2 (Real-Time Detection Transformer v2) is a transformer based detector that combines convolutional feature extraction with global self attention. Given an image, it outputs a fixed number of object queries, each with a class probability and a bounding box.

We keep the implementation as close as possible to the original codebase so that runs stay reproducible and stable, and so that experiments can be controlled from YAML files.

## Directory Structure

```
models/detection/
└── rtdetrv2/
    ├── configs/           # YAML configuration files
    ├── src/               # Core RT-DETRv2 implementation
    ├── tools/             # Training and evaluation utilities
    ├── requirements.txt   # Python dependencies
    └── README.md          # Original RT-DETRv2 documentation
```
