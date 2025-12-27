#!/bin/bash
#SBATCH --job-name=seg_final
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/nmo/spark_project/logs/segmentation/slurm/seg_final_%j.out

set -eo pipefail


echo "Start: $(date) on $(hostname)"
set +u
source ~/.bashrc
set -u 2>/dev/null || true
conda activate spark_seg

# 1) Inference (TTA) -> PNG label masks
python /home/users/nmo/spark_project/scripts/segmentation/run_segmentation_inference.py

# 2) Convert PNG -> NPZ 'data' format and zip it to final folder
python /home/users/nmo/spark_project/scripts/segmentation/convert_png_to_npz_submission.py \
  --input_dir /home/users/nmo/spark_project/inference_results/segmentation/predicted_masks \
  --output_dir /home/users/nmo/spark_project/inference_results/segmentation/npz_tmp \
  --zip_path /home/users/nmo/spark_project/inference_results/segmentation/final/submission_segmentation.zip

echo "End: $(date)"
