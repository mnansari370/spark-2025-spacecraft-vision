#!/bin/bash
#SBATCH --job-name=det_train
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/detection/slurm/%x_%j.out
#SBATCH --error=logs/detection/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/detection/slurm

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/models/detection/rtdetrv2:$PYTHONPATH"

echo "Start: $(date) on $(hostname)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

python -u models/detection/rtdetrv2/tools/train.py \
  -c models/detection/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_spark_40ep.yml \
  --device cuda \
  --use-amp \
  --output-dir checkpoints/detection/detection_model

echo "End: $(date)"
