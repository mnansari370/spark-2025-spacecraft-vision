#!/bin/bash
#SBATCH --job-name=seg_metrics
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_metrics_%j.err

set -e
set +u
source ~/.bashrc
set -u

conda activate spark_seg
cd /mnt/aiongpfs/users/nmo/spark_project

python -u reports/segmentation_val_metrics.py \
  --gt_dir inference_results/segmentation/_tmp_val_gt \
  --pred_dir inference_results/segmentation/val_predicted_masks \
  --limit 0
