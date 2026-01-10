#!/bin/bash
#SBATCH --job-name=det_val
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/det_val_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/det_val_%j.err

echo "Start: $(date) on $(hostname)"

# Avoid 'unbound variable' issues from system bashrc
set +u

source ~/.bashrc
conda activate spark_rtdetr

cd /mnt/aiongpfs/users/nmo/spark_project

python -u run/detection_infer_val.py \
  --limit 5000 \
  --score_thr 0.0 \
  --device cuda

echo "End: $(date)"
