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

set -e  # stop on error
# NOTE: do NOT "set -u" here; it breaks /etc/bashrc on this cluster.

echo "Start: $(date) on $(hostname)"
echo "PWD before cd: $(pwd)"

cd /mnt/aiongpfs/users/nmo/spark_project

# ---- Conda without sourcing ~/.bashrc (avoids /etc/bashrc BASHRCSOURCED issue) ----
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate spark_seg
# -------------------------------------------------------------------------------

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

nvidia-smi || true

python -u run/segmentation_infer_val.py --device cuda --limit 0 --no_tta

echo "End: $(date)"
