#!/bin/bash
#SBATCH --job-name=det_val
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs/detection/slurm inference_results/detection

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

test -f run/detection_infer_val.py
test -f data/annotations/spark_val.json

LIMIT=0
SCORE_THR=0.0

python -u run/detection_infer_val.py \
  --limit "$LIMIT" \
  --score_thr "$SCORE_THR" \
  --device cuda

echo "End: $(date)"
