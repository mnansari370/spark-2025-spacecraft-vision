# SLURM Job Scripts

This directory holds the SLURM job scripts we use to run training and inference on the ULHPC cluster.

## Directory Structure

```
jobs/
├── detection/
│   ├── train_detection_rtdetrv2_gpu.sh
│   └── infer_detection_test_gpu.sh
└── segmentation/
    ├── infer_segmentation_train.sh
    ├── infer_segmentation_val_gpu.sh
    ├── infer_segmentation_test_gpu.sh
    └── build_segmentation_submission_cpu.sh
```

## Detection

`train_detection_rtdetrv2_gpu.sh`
Trains the RT-DETRv2 detector and saves the best checkpoint.

`infer_detection_test_gpu.sh`
Runs the trained detector on the test set.

## Segmentation

`infer_segmentation_train.sh`
Trains the DeepLabV3+ segmentation model and saves the best checkpoint. (The name is kept for historical reasons; this is the training job.)

`infer_segmentation_val_gpu.sh`
Runs inference on the validation set to produce masks for evaluation.

`infer_segmentation_test_gpu.sh`
Runs inference on the test set to produce masks for the submission.

`build_segmentation_submission_cpu.sh`
Converts the predicted PNG masks into the `.npz` format required by Codabench.
