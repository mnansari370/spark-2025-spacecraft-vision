# Models

This directory contains all **model-related code and configuration files** used for the SPARK spacecraft vision project, covering both **detection** and **segmentation** tasks.

The structure is intentionally split by task to keep the repository **clear, modular, and reproducible**.

---

## Overview

```text
models/
├── detection/
│   └── rtdetrv2/          # RT-DETRv2 detection framework (third-party)
│
└── segmentation/
    ├── model_factory.py  # Model construction & checkpoint loading
    └── ...               # Segmentation architecture components
