#!/bin/bash
#SBATCH --job-name=seg_submit
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segmentation/slurm/%x_%j.out
#SBATCH --error=logs/segmentation/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/segmentation/slurm inference_results/segmentation

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_seg

echo "Start: $(date) on $(hostname)"

# (A) Build single NPZ submission :
python -u reports/make_segmentation_submission_bool_npz.py \
  --pred_dir inference_results/segmentation/predicted_masks \
  --out_npz inference_results/segmentation/segmentation_submission_BOOL.npz \
  --limit 0

# (B) Or if you need the zip of per-image npz:
# python -u run/segmentation_make_submission.py

echo "End: $(date)"
