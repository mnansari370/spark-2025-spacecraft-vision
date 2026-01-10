#!/bin/bash
#SBATCH --job-name=det_val
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/det_val_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/det_val_%j.err

# IMPORTANT on ULHPC: /etc/bashrc may reference unset vars like BASHRCSOURCED.
# If we use `set -u`, the script can exit immediately.
set -eo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"

# Safe shell init
source ~/.bashrc

conda activate spark_seg
cd /mnt/aiongpfs/users/nmo/spark_project

python -u run/detection_infer_val.py \
  --coco_gt data/annotations/spark_val.json \
  --out_json inference_results/detection/val_predictions.json \
  --limit 0 \
  --score_thr 0.0 \
  --device cuda

echo "End: $(date)"
