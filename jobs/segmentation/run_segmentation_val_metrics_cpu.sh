#!/bin/bash
#SBATCH --job-name=seg_metrics
#SBATCH --partition=cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.err

set -euo pipefail

cd /mnt/aiongpfs/users/nmo/spark_project
mkdir -p logs/segmentation/slurm
source ~/.bashrc
conda activate spark_seg

mkdir -p inference_results/segmentation/_tmp_val_gt
rm -rf inference_results/segmentation/_tmp_val_gt/*

find data/spark-2024-train-val/mask -type f -path "*/val/*" -name "*_layer.*" -print0 \
  | xargs -0 -I{} ln -s {} inference_results/segmentation/_tmp_val_gt/ 2>/dev/null

python -u reports/segmentation_val_metrics.py \
  --gt_dir inference_results/segmentation/_tmp_val_gt \
  --pred_dir inference_results/segmentation/val_predicted_masks \
  --limit 0
