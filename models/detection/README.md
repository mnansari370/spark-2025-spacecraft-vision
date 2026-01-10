# Detection Models

This directory contains all code and configuration related to **spacecraft detection** for the SPARK challenge.

The detection task is formulated as **multi-class object detection**, where each image may contain one spacecraft belonging to one of the official SPARK classes.

---

## Detection Framework

### RT-DETRv2

- **Architecture**: RT-DETRv2 (Real-Time DEtection TRansformer v2)
- **Task**: Spacecraft detection (bounding boxes + class labels)
- **Framework type**: Transformer-based object detector
- **Primary use**: High-accuracy inference on the SPARK dataset

The RT-DETRv2 implementation used in this project is kept **as close as possible to the original codebase**, to ensure:
- reproducibility
- stability
- ease of configuration using YAML files

---

## Directory Structure

```text
models/detection/
└── rtdetrv2/
    ├── configs/          # YAML configuration files
    ├── src/              # Core RT-DETRv2 implementation
    ├── tools/            # Training / evaluation utilities
    ├── deploy/           # Export / deployment utilities
    └── README.md         # Original RT-DETRv2 documentation
