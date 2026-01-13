#!/bin/bash
#SBATCH --job-name=seg_train
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/segmentation/slurm/%x_%j.out
#SBATCH --error=logs/segmentation/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/segmentation/slurm

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_seg

export PYTHONUNBUFFERED=1

echo "Start: $(date) on $(hostname)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

test -f scripts/segmentation/train_segmentation.py
test -d data/spark-2024-train-val/images
test -d data/annotations

python -u scripts/segmentation/train_segmentation.py \
  --images_root data/spark-2024-train-val/images \
  --annotations_dir data/annotations \
  --output_dir checkpoints/segmentation/segmentation_model \
  --device cuda \
  --num_epochs 100 \
  --batch_size 8

echo "End: $(date)"
