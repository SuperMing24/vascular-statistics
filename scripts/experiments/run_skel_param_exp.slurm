#!/bin/bash
#SBATCH --job-name=skelparam
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_skel_param/job_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_skel_param/job_%j.err

set +u
source ~/.bashrc
set -u
set -eo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate vascstats
cd /share/home/sukm/vascular-statistics
python scripts/experiments/run_skel_param_exp.py \
    --out-dir /share/home/sukm/experiments/vs_skel_param \
    "$@"

echo "=== SKELPARAM ALL DONE ==="
