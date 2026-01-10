#!/bin/bash
#SBATCH --job-name=det_test
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/detection/slurm/%x_%j.out
#SBATCH --error=logs/detection/slurm/%x_%j.err

set -euo pipefail

echo "Start: $(date) on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs/detection/slurm inference_results/detection

# Robust conda init (avoid ~/.bashrc issues)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

# Sanity checks
test -f run/detection_infer.py
test -f checkpoints/detection/detection_model/best.pth
test -d data/spark-2024-detection-test/images

# Run full test set: LIMIT=0 means all found images
LIMIT=0
SCORE_THR=0.30

python -u run/detection_infer.py --limit "$LIMIT" --score_thr "$SCORE_THR"

echo "End: $(date)"
