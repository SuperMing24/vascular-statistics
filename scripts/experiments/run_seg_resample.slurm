#!/bin/bash
# 分割采样尺寸实验 —— Slurm 提交脚本
# 脚本走 git（本文件在 repo），输出写入 experiments/vs_seg_resample/
#SBATCH --job-name=vs_segres
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_seg_resample/job_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_seg_resample/job_%j.err

source ~/.bashrc
/share/home/sukm/conda_envs/vesseg/bin/python \
    /share/home/sukm/vascular-statistics/scripts/experiments/seg_resample_eval.py --save-masks
