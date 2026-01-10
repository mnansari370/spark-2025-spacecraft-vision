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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/detection/slurm inference_results/detection

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

echo "Start: $(date) on $(hostname)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

test -f run/detection_infer.py
test -f checkpoints/detection/detection_model/best.pth
test -d data/spark-2024-detection-test/images

LIMIT=20000
SCORE_THR=0.30

python -u run/detection_infer.py --limit "$LIMIT" --score_thr "$SCORE_THR" --device cuda

# Optional (if submission needs CSV):
# python -u scripts/detection/convert_predictions_to_submission.py

echo "End: $(date)"
