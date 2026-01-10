#!/bin/bash
#SBATCH --job-name=seg_val
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/segmentation/slurm/%x_%j.out
#SBATCH --error=logs/segmentation/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/segmentation/slurm inference_results/segmentation

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_seg

echo "Start: $(date) on $(hostname)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

test -f run/segmentation_infer_val.py

python -u run/segmentation_infer_val.py --device cuda --limit 0 --no_tta

echo "End: $(date)"
