#!/bin/bash
#SBATCH --job-name=seg_val
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_val_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_val_%j.err

# IMPORTANT: avoid /etc/bashrc crash when "set -u" is active somewhere
set +u
set -e

echo "Start: $(date) on $(hostname)"
echo "PWD: $(pwd)"

# Load conda safely
source ~/.bashrc
conda activate spark_seg

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

nvidia-smi || true

cd /mnt/aiongpfs/users/nmo/spark_project

# Run segmentation val inference (GT-style naming expected in val set)
python -u run/segmentation_infer_val.py --device cuda --limit 0 --no_tta

echo "End: $(date)"
