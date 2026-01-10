# SPARK 2025 – Spacecraft Detection & Segmentation (CVIA Course Project)

This repository contains my **end-to-end computer vision pipeline** for **spacecraft detection** and **spacecraft part segmentation**, developed for the **CVIA (Computer Vision and Image Analysis)** course and the **SPARK challenge** at the **University of Luxembourg**.

The project is designed to run on **ULHPC using SLURM**, and to generate **Codabench-ready submissions** for:

- **Detection** – multi-class spacecraft bounding boxes  
- **Segmentation** – spacecraft body and solar panels  

---

##  Results (Codabench)

Best scores obtained on the official Codabench evaluation:

- **Detection (RT-DETRv2): 0.9865**
- **Segmentation (DeepLabV3+): 0.88**

---

##  Repository Structure

A complete **Level-3 directory tree** of the repository is provided in:

- `PROJECT_TREE_L3.txt`

### Main folders

```text
spark_project/
├── jobs/               # SLURM job scripts (training & inference only)
├── scripts/            # Reusable inference, conversion & visualization utilities
├── models/             # Model implementations and configuration files
├── inference_results/  # Predictions, visualizations, and Codabench submissions
├── logs/               # Cleaned SLURM logs (stdout / stderr)
├── assets/             # Images used in README and reports
└── README.md
