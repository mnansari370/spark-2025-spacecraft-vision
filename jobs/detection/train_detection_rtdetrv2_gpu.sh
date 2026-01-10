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

echo "Start: $(date) on $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs/detection/slurm checkpoints/detection/detection_model

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

# W&B (keep; change to offline if needed)
export WANDB_DIR="$REPO_ROOT/wandb/detection"
mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="CV_Detection"
export WANDB_ENTITY="nafees-workspace"
export WANDB_MODE="online"
export WANDB_NAME="rtdetr40_${SLURM_JOB_ID}"
export WANDB_SILENT=true
export WANDB_CONSOLE=wrap

export PYTHONPATH="$REPO_ROOT/models/detection/rtdetrv2:$PYTHONPATH"

python -u models/detection/rtdetrv2/tools/train.py \
  -c models/detection/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_spark_40ep.yml \
  --device cuda \
  --use-amp \
  --output-dir "$REPO_ROOT/checkpoints/detection/detection_model"

echo "End: $(date)"
