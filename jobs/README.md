# SLURM Job Scripts

This directory contains **SLURM job scripts** used to run training,
inference, and submission preparation on **ULHPC**.

These scripts define:
- requested resources (GPU, CPU, memory, time)
- environment activation
- execution order of Python scripts

---

## Purpose of this folder

The `jobs/` directory answers the question:

> *How is the pipeline executed on ULHPC?*

It contains **only execution logic**, not model or algorithm code.

---

## Directory Structure

```text
jobs/
├── detection/
│   ├── train_detection_*.sh
│   ├── infer_detection_*.sh
│   └── build_detection_submission_*.sh
│
├── segmentation/
│   ├── infer_segmentation_*.sh
│   └── build_segmentation_submission_*.sh
│
└── README.md
