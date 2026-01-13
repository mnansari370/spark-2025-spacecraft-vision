#!/bin/bash
#SBATCH --job-name=seg_conf
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/seg_conf_%j.out
#SBATCH --error=logs/seg_conf_%j.err

# IMPORTANT: avoid system bashrc issues on ULHPC
set -e

# Go to repo
cd /mnt/aiongpfs/users/nmo/spark_project

# Ensure logs dir exists (repo-root logs/)
mkdir -p logs

# Load conda in a robust way (works even in non-interactive Slurm)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found under \$HOME/miniconda3 or \$HOME/anaconda3"
  exit 1
fi

conda activate spark_rtdetr

echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
python -c "import sys; print('PY:', sys.executable)"

python reports/scripts/segmentation/segmentation_confusion.py \
  --gt_root data/spark-2024-train-val/mask \
  --pred_root inference_results/segmentation/val_predicted_masks \
  --split val \
  --out_dir reports/latest

echo "[OK] Done."
