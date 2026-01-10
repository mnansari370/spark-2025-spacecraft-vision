#!/bin/bash
#SBATCH --job-name=det_infer
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/nmo/spark_project/logs/detection/slurm/det_infer_%j.out
#SBATCH --error=/home/users/nmo/spark_project/logs/detection/slurm/det_infer_%j.err


set -eo pipefail
unset BASH_ENV
export BASHRCSOURCED=1


echo "Start: $(date) on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/users/nmo/spark_project}"
cd "$REPO_ROOT"

mkdir -p logs/detection/slurm reports/figures reports/tables reports/metrics inference_results/detection

# ---- robust conda init for batch ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spark_rtdetr

set -u

# ---- sanity checks (fail fast with clear error) ----
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true

test -f run/detection_infer.py
test -f models/detection/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_spark_40ep.yml
test -f checkpoints/detection/detection_model/best.pth
test -d data/spark-2024-detection-test/images

# ---- choose how many images you want for report stats ----
# 5000 is already a “good amount” for stable histograms.
# For full test-set (20k) use 20000 (but takes longer).
LIMIT=5000
SCORE_THR=0.30

echo "Running detection inference: LIMIT=$LIMIT SCORE_THR=$SCORE_THR"
python -u run/detection_infer.py --limit "$LIMIT" --score_thr "$SCORE_THR"

echo "Generating detection plots/tables"
python -u reports/detection_report_plots.py --score_thr "$SCORE_THR"

echo "End: $(date)"
