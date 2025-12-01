#!/bin/bash
#SBATCH --job-name=spark_rtdetrv2
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rtdetrv2_%j.out

echo "Job started on $(hostname) at $(date)"

# Load conda
source ~/.bashrc

# Activate the environment for RT-DETRv2
conda activate spark_rtdetr

# Go to the RT-DETRv2 code base
cd /home/users/nmo/spark_project/rtdetr/rtdetrv2_pytorch

python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml \
  --device cuda \
  --use-amp \
  --output-dir /home/users/nmo/spark_project/models/rtdetrv2_spark

echo "Job finished at $(date)"

