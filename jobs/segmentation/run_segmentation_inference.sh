#!/bin/bash
#SBATCH --job-name=seg_final
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/segmentation/slurm/seg_final_%j.out
#SBATCH --error=logs/segmentation/slurm/seg_final_%j.err

set -euo pipefail

echo "Start: $(date) on $(hostname)"

source ~/.bashrc
conda activate spark_seg

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
echo "REPO_ROOT=$REPO_ROOT"
cd "$REPO_ROOT"

# Create necessary directories
mkdir -p logs/segmentation/slurm
mkdir -p inference_results/segmentation

# Run segmentation inference
python -u scripts/segmentation/run_segmentation_inference.py

python -u scripts/segmentation/convert_png_to_npz_submission.py \
  --input_dir "inference_results/segmentation/predicted_masks" \
  --output_dir "inference_results/segmentation/npz_tmp" \
  --zip_path "inference_results/segmentation/final/submission_segmentation.zip"

echo "End: $(date)"
