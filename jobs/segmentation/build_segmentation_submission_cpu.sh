#!/bin/bash
#SBATCH --job-name=seg_sub
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segmentation/slurm/%x_%j.out
#SBATCH --error=logs/segmentation/slurm/%x_%j.err

set -euo pipefail

echo "Start: $(date) on $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs/segmentation/slurm inference_results/segmentation/npz_tmp

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_seg

test -f scripts/segmentation/convert_png_to_npz_submission.py
test -d inference_results/segmentation/predicted_masks

python -u scripts/segmentation/convert_png_to_npz_submission.py \
  --input_dir inference_results/segmentation/predicted_masks \
  --output_dir inference_results/segmentation/npz_tmp \
  --zip_path inference_results/segmentation/submission_segmentation.zip

echo "End: $(date)"
