#!/bin/bash
#SBATCH --job-name=csam_test
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_test_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_test_%j.err

source /share/home/sukm/miniconda3/etc/profile.d/conda.sh
conda activate vesseg
export OMP_NUM_THREADS=16

# 只用最小体积测试: sampling=3.0, speed_param 加速
cd /tmp/test_csam_skel_1190
mkdir -p /tmp/test_csam_skel_1190

python3 << 'PYEOF'
import sys, os, time
sys.path.insert(0, '/share/home/sukm/vascular-statistics/python')
from vascular_statistics.vascgraph import GraphIO, Skeletonize
from VascGraph.Tools.CalcTools import fixG
import tifffile
import numpy as np

seg_path = '/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W1N_20190920_ref_csam_seg.tiff'
print(f'Loading: {seg_path}')
arr = tifffile.imread(seg_path)
if arr.ndim == 3:
    arr = np.transpose(arr, (1, 2, 0))  # [D,H,W] -> [H,W,D]
stack = (arr > 0).astype(int)
print(f'Shape: {stack.shape}, foreground: {stack.sum():,} voxels')

for sampling in [3.0, 4.0, 5.0]:
    print(f'\n--- sampling={sampling} ---')
    t0 = time.time()
    try:
        sk = Skeletonize.Skeleton(label=stack, sampling=sampling,
                                  speed_param=0.1, dist_param=0.5, med_param=0.5)
        sk.Update()
        graph = fixG(sk.GetOutput())
        elapsed = time.time() - t0
        print(f'OK: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {elapsed:.1f}s')
        break
    except Exception as e:
        elapsed = time.time() - t0
        print(f'FAIL after {elapsed:.1f}s: {e}')

print('\nDone.')
PYEOF
rm -rf /tmp/test_csam_skel_1190
