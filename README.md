# SPARK 2024 – Spacecraft Detection & Segmentation

> Course project for **Computer Vision and Image Analysis (CVIA)** at the **University of Luxembourg**, built around the [SPARK 2024 challenge](https://cvisp.uni.lu/spark/).  
> Trained and evaluated on the **ULHPC cluster** (SLURM), with submissions evaluated on **Codabench**.

---

## Results

| Task | Model | Codabench Score | Val AP@50 | Val AP@50:95 |
|------|-------|:--------------:|:---------:|:------------:|
| **Detection** | RT-DETRv2 (ResNet-50vd) | **0.9865** | **0.997** | **0.969** |
| **Segmentation** | DeepLabV3+ (ResNet-50) | **0.88** | — | — |

---

## Visual Examples

Detection predictions (bounding box + class label):

| Proba2 | Smart1 | VenusExpress |
|--------|--------|--------------|
| ![](assets/README_images/detection_proba2.png) | ![](assets/README_images/detection_smart1.png) | ![](assets/README_images/detection_venusexpress.png) |

Segmentation predictions (red = spacecraft body, blue = solar panels):

| Example 1 | Example 2 |
|-----------|-----------|
| ![](assets/README_images/segmentation_example_1.png) | ![](assets/README_images/segmentation_example_2.png) |

---

## Overview

The **SPARK** (Spacecraft PARts and Kinematics) dataset contains images of spacecraft across multiple missions, captured in simulated orbital conditions.  
This project tackles two computer vision tasks in a single unified pipeline:

- **Detection** – Classify and localise each spacecraft with a bounding box.  
  10 classes: *Cheops, LisaPathfinder, ObservationSat1, Proba2, Proba3, Proba3ocs, Smart1, Soho, VenusExpress, XMM Newton*.

- **Segmentation** – Pixel-wise labelling into three classes:  
  `0 = background`, `1 = spacecraft body`, `2 = solar panels`.

Both pipelines follow the same workflow: **data preparation → training → inference → submission formatting**.

---

## Architecture

### Detection — RT-DETRv2

RT-DETRv2 (Real-Time Detection Transformer v2) is an end-to-end transformer detector that eliminates NMS and achieves state-of-the-art accuracy at real-time speeds.

| Component | Choice |
|-----------|--------|
| Backbone | PResNet-50 (variant D), ImageNet pretrained |
| Encoder | HybridEncoder (hidden dim 256, GELU, 1 encoder layer) |
| Decoder | RTDETRTransformerv2 (6 layers, 300 queries, 3 feature levels) |
| Loss | VariFocal + GIoU + L1 (weights 1 : 2 : 5) |
| Optimizer | AdamW — backbone LR 1e-5, others 1e-4, weight decay 1e-4 |
| LR schedule | MultiStepLR (decay ×0.1 at epochs 28, 36) |
| Input size | 640 × 640 |
| Training | 40 epochs, AMP, EMA, batch size 8 |
| Augmentation | PhotometricDistort, ZoomOut, IoUCrop, HFlip, MultiScale (480–640 px) — first 30 epochs; plain resize only after |

Annotations are converted from SPARK CSV format to **COCO JSON** before training (`scripts/detection/convert_spark_to_coco.py`).  
Experiment tracking via **W&B** and **TensorBoard**.

### Segmentation — DeepLabV3+

| Component | Choice |
|-----------|--------|
| Architecture | DeepLabV3+ |
| Encoder | ResNet-50 (via `segmentation_models_pytorch`) |
| Classes | 3 (background, body, panels) |
| Training | 100 epochs, batch size 8, GPU |
| TTA | Horizontal flip TTA at inference (averages logits of original + flipped image) |
| Output | Per-image PNG label masks → converted to `.npz` → zipped for Codabench |

---

## Dataset Structure

```
data/
├── spark-2024-train-val/
│   ├── images/
│   │   └── <ClassName>/<split>/<image_XXXXX_img.jpg>
│   ├── mask/                          # segmentation ground truth
│   ├── train.csv                      # detection labels
│   └── val.csv
├── spark-2024-detection-test/
│   └── images/                        # ~20 000 unlabelled test images
├── spark-2024-segmentation-test/
│   └── stream-1-test/                 # unlabelled test images
└── annotations/                       # COCO JSON (auto-generated)
    ├── spark_train.json
    └── spark_val.json
```

---

## Project Structure

```
spark_project/
├── assets/                # Images used in this README
├── checkpoints/           # Saved model weights (not tracked in git)
│   ├── detection/detection_model/best.pth
│   └── segmentation/segmentation_model/best.pth
├── data/                  # Dataset root (not tracked in git)
├── inference_results/     # Model outputs (not tracked in git)
├── jobs/                  # SLURM job scripts
│   ├── detection/
│   │   ├── train_detection_rtdetrv2_gpu.sh
│   │   └── infer_detection_test_gpu.sh
│   └── segmentation/
│       ├── infer_segmentation_train.sh
│       ├── infer_segmentation_val_gpu.sh
│       ├── infer_segmentation_test_gpu.sh
│       └── build_segmentation_submission_cpu.sh
├── models/
│   ├── detection/rtdetrv2/            # Full RT-DETRv2 codebase
│   │   ├── configs/                   # YAML training configs
│   │   ├── src/                       # Model, solver, backbone, criterion
│   │   └── tools/train.py
│   └── segmentation/
│       └── model_factory.py           # DeepLabV3+ build + checkpoint loader
├── reports/                           # Evaluation scripts and generated figures
│   ├── scripts/                       # Metric computation + plotting
│   └── latest/                        # Output figures and tables
├── run/                               # Parameter-passing wrappers for job scripts
│   ├── detection_infer.py
│   ├── detection_infer_val.py
│   └── segmentation_make_submission.py
├── scripts/
│   ├── detection/
│   │   ├── convert_spark_to_coco.py       # CSV → COCO JSON
│   │   ├── detection_dataset_loader.py    # PyTorch Dataset
│   │   ├── convert_predictions_to_submission.py  # JSON → CSV submission
│   │   └── visualize_detection_results.py
│   └── segmentation/
│       ├── infer_segmentation.py          # Main inference script
│       ├── convert_png_to_npz_submission.py  # PNG → NPZ → ZIP
│       └── visualize_segmentation_results.py
└── wandb/                             # W&B run logs (not tracked in git)
```

---

## Setup

Two separate conda environments are used (detection and segmentation have conflicting dependencies).

### Detection environment

```bash
conda create -n spark_rtdetr python=3.10 -y
conda activate spark_rtdetr
pip install -r models/detection/rtdetrv2/requirements.txt
```

Requirements include: `torch>=2.0.1`, `torchvision>=0.15.2`, `faster-coco-eval`, `pycocotools`, `tensorboard`, `scipy`, `onnx`, `onnxruntime-gpu`.

### Segmentation environment

```bash
conda create -n spark_seg python=3.10 -y
conda activate spark_seg
pip install torch torchvision segmentation-models-pytorch numpy pillow tqdm
```

---

## Reproducing the Pipeline

All jobs are submitted to ULHPC via SLURM. To run locally, replace `sbatch` with `bash`, or run the Python scripts directly.

### 1. Prepare detection annotations (run once)

```bash
conda activate spark_rtdetr
python scripts/detection/convert_spark_to_coco.py
# Outputs: data/annotations/spark_train.json, spark_val.json
```

### 2. Train detection model

```bash
sbatch jobs/detection/train_detection_rtdetrv2_gpu.sh
# Saves: checkpoints/detection/detection_model/best.pth
# Resources: 1 GPU, 24 GB RAM, up to 48 h
```

To train manually:
```bash
conda activate spark_rtdetr
export PYTHONPATH="$PWD/models/detection/rtdetrv2:$PYTHONPATH"
python -u models/detection/rtdetrv2/tools/train.py \
  -c models/detection/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_spark_40ep.yml \
  --device cuda --use-amp \
  --output-dir checkpoints/detection/detection_model
```

### 3. Train segmentation model

```bash
sbatch jobs/segmentation/infer_segmentation_train.sh
# Saves: checkpoints/segmentation/segmentation_model/best.pth
# Resources: 1 GPU, 24 GB RAM, up to 48 h
```

### 4. Run detection inference (test set)

```bash
sbatch jobs/detection/infer_detection_test_gpu.sh
# Outputs: inference_results/detection/detection_test_predictions.json
```

Then convert to Codabench CSV:
```bash
conda activate spark_rtdetr
python scripts/detection/convert_predictions_to_submission.py
# Outputs: inference_results/detection/detection.csv
```

### 5. Run segmentation inference (test set)

```bash
sbatch jobs/segmentation/infer_segmentation_test_gpu.sh
# Outputs: inference_results/segmentation/predicted_masks/*.png
```

Then build the submission ZIP:
```bash
sbatch jobs/segmentation/build_segmentation_submission_cpu.sh
# Outputs: inference_results/segmentation/final/submission_segmentation.zip
```

---

## Detailed Results

### Detection — Validation COCO Metrics

```
AP  @[ IoU=0.50:0.95 | all ]  = 0.969
AP  @[ IoU=0.50      | all ]  = 0.997
AP  @[ IoU=0.75      | all ]  = 0.994
AR  @[ IoU=0.50:0.95 | all ]  = 0.978
```

### Detection — Per-Class AP@[0.50:0.95] (Validation)

| Class | AP |
|-------|----|
| Proba2 | 0.986 |
| Cheops | 0.984 |
| ObservationSat1 | 0.981 |
| LisaPathfinder | 0.981 |
| Proba3 | 0.980 |
| XMM Newton | 0.974 |
| Proba3ocs | 0.962 |
| VenusExpress | 0.956 |
| Soho | 0.953 |
| Smart1 | 0.936 |
| **Mean** | **0.969** |

### Segmentation — Validation IoU

| Class | IoU |
|-------|-----|
| Background | 0.967 |
| Body | 0.003 |
| Panels | 0.009 |
| **mIoU** | **0.326** |

> Note: The Codabench segmentation score (0.88) differs from the local validation mIoU because Codabench evaluates on the held-out test split using its own metric definition (focused on foreground classes only).  
> The class imbalance is severe: background = 96% of all pixels, body = 3.6%, panels = 1.2%.

---

## Tech Stack

| Area | Tools |
|------|-------|
| Deep learning | PyTorch, torchvision |
| Detection model | RT-DETRv2 (custom YAML config) |
| Segmentation model | `segmentation_models_pytorch` (DeepLabV3+) |
| Annotation format | COCO JSON |
| Experiment tracking | Weights & Biases, TensorBoard |
| Infrastructure | ULHPC HPC cluster, SLURM |
| Data processing | NumPy, Pandas, Pillow |
| Evaluation | pycocotools, faster-coco-eval |

---

## Acknowledgements

- RT-DETRv2 implementation by [lyuwenyu](https://github.com/lyuwenyu/RT-DETR) — adapted with a custom SPARK training configuration.
- DeepLabV3+ via the [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) library.
- SPARK 2024 challenge organised by the [Computer Vision and Image Processing group](https://cvisp.uni.lu/) at the University of Luxembourg.

```bibtex
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
  title   = {RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
  author  = {Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
  year    = {2024},
  eprint  = {2407.17140},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2407.17140},
}
```
