#!/bin/bash
#SBATCH --job-name=det_train
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/detection_train_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/detection_train_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"

source ~/.bashrc
conda activate spark_rtdetr
export PYTHONUNBUFFERED=1

# Repo root based on this script location (works in sbatch)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
echo "REPO_ROOT=$REPO_ROOT"

# W&B (keep your settings)
export WANDB_DIR="$REPO_ROOT/logs/wandb/detection"
mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="CV_Detection"
export WANDB_ENTITY="nafees-workspace"
export WANDB_MODE="online"
export WANDB_NAME="rtdetr40_${SLURM_JOB_ID}"
export WANDB_SILENT=true
export WANDB_CONSOLE=wrap

# Run from repo root so YAML paths like data/... resolve correctly
cd "$REPO_ROOT"

# Make RT-DETR code importable
export PYTHONPATH="$REPO_ROOT/models/detection/rtdetrv2:$PYTHONPATH"

python -u models/detection/rtdetrv2/tools/train.py \
  -c models/detection/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_spark_40ep.yml \
  --device cuda \
  --use-amp \
  --output-dir "$REPO_ROOT/checkpoints/detection/detection_model"

echo "End: $(date)"
