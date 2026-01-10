#!/bin/bash
#SBATCH --job-name=seg_metrics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"

cd /mnt/aiongpfs/users/nmo/spark_project
mkdir -p logs/segmentation/slurm

source ~/.bashrc
conda activate spark_seg

python -m reports.scripts.segmentation.segmentation_val_metrics \
  --gt_dir  data/spark-2024-train-val/mask \
  --pred_dir inference_results/segmentation/val_predicted_masks \
  --num_classes 3 \
  --class_names bg body panels

echo "Done: $(date)"
