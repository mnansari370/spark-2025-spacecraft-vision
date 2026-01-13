# SPARK 2024 – Spacecraft Detection & Segmentation (CVIA Course Project)

This repository contains the **pipeline** for the **spacecraft detection** and **spacecraft part segmentation**, for the **CVIA (Computer Vision and Image Analysis)** course and the **SPARK challenge** at the **University of Luxembourg**.

The project is designed to run on **ULHPC using SLURM**, and to generate **Codabench-ready submissions** for:

- **Detection** – multi-class spacecraft bounding boxes  
- **Segmentation** – spacecraft body and solar panels  


##  Results (Codabench)

Best scores obtained on the official Codabench evaluation:

- **Detection (RT-DETRv2): 0.9865**
- **Segmentation (DeepLabV3+): 0.88**



## Main folders

```text
spark_project/
├── assets/             # Images used in README and reports
├── jobs/               # SLURM job scripts 
├── models/             # Model implementations
├── reports/            # Scripts for generating reports
├── run/                # Used to pass parameters to the scripts
├── scripts/            # Main scripts to train the models
└── README.md
```


## ML Workflow

The jobs are submitted to the HPC using the slurm scripts found in the jobs directory. The slurm script either directly calls the scripts from "scripts" directory or if there are many parameters, it calls the files in the "run" directory which then calls the script from the "scripts" directory.

Both detection and segmentation follow the standard ML pipeline:

### 1. Training
- Train the model on labeled training data
- Save the best checkpoint (model weights)
- Example: `jobs/detection/train_detection_rtdetrv2_gpu.sh`

### 2. Inference
- Load trained model checkpoint
- Generate predictions on validation or test data
- Save predictions (JSON for detection, PNG masks for segmentation)
- Examples:
  - Detection validation: `jobs/detection/infer_detection_val_gpu.sh`
  - Detection test: `jobs/detection/infer_detection_test_gpu.sh`
  - Segmentation validation: `jobs/segmentation/infer_segmentation_val_gpu.sh`
  - Segmentation test: `jobs/segmentation/infer_segmentation_test_gpu.sh`

### 3. Evaluation (Validation Only)
- Compare predictions against ground truth labels
- Compute metrics (mAP for detection, IoU for segmentation)
- Generate reports and visualizations
- Examples:
  - Detection metrics: `reports/jobs/detection/run_detection_val_infer_gpu.sh`
  - Segmentation metrics: `reports/jobs/segmentation/run_segmentation_val_metrics_slurm.sh`

### Key Differences: Validation vs Test
- **Validation**: Has labels → Infer + Evaluate → Get metrics for model assessment
- **Test**: No labels (unlabeled competition data) → Infer only → Generate submission