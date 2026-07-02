# Segmentation Model

This directory holds the code for spacecraft part segmentation.

The task is pixel wise classification of each image into three classes: background, spacecraft body, and solar panels.

## DeepLabV3+

The model is DeepLabV3+ with a ResNet-50 backbone, built through the `segmentation_models_pytorch` library. It predicts a class distribution for every pixel and produces the final mask by taking the argmax over classes.

We keep the implementation simple and explicit, which makes it stable, reproducible, and easy to run on the HPC cluster.

## Directory Structure

```
models/segmentation/
├── model_factory.py   # Model construction and checkpoint loading
└── README.md          # This file
```
