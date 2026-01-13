# SLURM Job Scripts

This directory contains **SLURM job scripts** used to submit and run jobs on the **HPC**.


## Directory Structure

```text
jobs/
├── detection/
│   ├── train_detection_*.sh
│   └── infer_detection_*.sh
│   
│
├── segmentation/
│   ├── infer_segmentation_*.sh
│   └── build_segmentation_submission.sh
│
└── README.md
```



## Detection

- `train_detection_rtdetv2_gpu.sh`
Used for training the model

- `infer_detection_test_gpu.sh`
Used for running the model on the test dataset

## Segmentation

- `train_segmentation_gpu.sh`  
  Trains the DeepLabV3+ segmentation model on the training dataset and saves the best checkpoint

- `infer_segmentation_val_gpu.sh`  
  Runs inference on the validation dataset to generate predicted segmentation masks for evaluation

- `infer_segmentation_test_gpu.sh`  
  Runs inference on the test dataset to generate predicted segmentation masks for submission

- `build_segmentation_submission_cpu.sh`  
  Converts the predicted masks from PNG format to the required .npz submission format