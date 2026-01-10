# Segmentation Models

This directory contains all code related to **spacecraft part segmentation** for the SPARK challenge.

The segmentation task focuses on **pixel-wise classification** of spacecraft images into:
- background
- spacecraft body
- solar panels

---

## Segmentation Framework

### DeepLabV3+ (Baseline)

- **Architecture**: DeepLabV3+
- **Backbone**: ResNet-50
- **Framework**: PyTorch
- **Task**: Semantic segmentation (3 classes)
- **Primary use**: Accurate segmentation of spacecraft body and solar panels

The implementation is intentionally kept **simple and explicit**, prioritising:
- stability
- reproducibility
- ease of inference on ULHPC

---

## Directory Structure

```text
models/segmentation/
├── model_factory.py     # Model construction and checkpoint loading
└── README.md            # This file
