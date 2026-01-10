#!/bin/bash
#SBATCH --job-name=seg_npz
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_npz_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_npz_%j.err

set -euo pipefail
cd /mnt/aiongpfs/users/nmo/spark_project
mkdir -p logs/segmentation/slurm

source ~/.bashrc
conda activate spark_seg

python -u reports/make_segmentation_submission_bool_npz.py \
  --pred_dir inference_results/segmentation/predicted_masks \
  --out_npz inference_results/segmentation/segmentation_submission_BOOL.npz \
  --limit 0
