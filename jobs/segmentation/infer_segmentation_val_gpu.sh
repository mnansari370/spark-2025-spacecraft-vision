#!/bin/bash
#SBATCH --job-name=seg_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --output=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_val_%j.out
#SBATCH --error=/mnt/aiongpfs/users/nmo/spark_project/logs/segmentation/slurm/seg_val_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"

# Avoid unbound-variable issues from system bashrc
set +u

source ~/.bashrc
conda activate spark_seg

# Debug: confirm GPU is visible
nvidia-smi || true
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"

cd /mnt/aiongpfs/users/nmo/spark_project

# Clean output so we regenerate a complete set
rm -rf inference_results/segmentation/val_predicted_masks
mkdir -p inference_results/segmentation/val_predicted_masks

python -u run/segmentation_infer_val.py \
  --images_root data/spark-2024-train-val/images \
  --out_dir inference_results/segmentation/val_predicted_masks \
  --ckpt checkpoints/segmentation/segmentation_model/best.pth \
  --device cuda

echo "End: $(date)"
