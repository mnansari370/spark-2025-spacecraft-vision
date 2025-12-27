#!/bin/bash
#SBATCH --job-name=det_infer
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/detection_infer_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/detection/slurm/detection_infer_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"

source ~/.bashrc
conda activate spark_rtdetr
export PYTHONUNBUFFERED=1

echo "GPU check:"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
echo "REPO_ROOT=$REPO_ROOT"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/models/detection/rtdetrv2:$PYTHONPATH"

# This script writes JSON into inference_results/detection/
python -u models/detection/rtdetrv2/run_inference_on_testset.py

echo "End: $(date)"
