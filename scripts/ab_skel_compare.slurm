#!/bin/bash
#SBATCH --job-name=ab_skel
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/ab_skel_%A_%a.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/ab_skel_%A_%a.err
#SBATCH --array=0-1
#
# A/B 骨架化方案对比 —— 同一样本 W2R_20190903_ws1 (CSAM 分割)
#   方案 A (task 0): 512 原生 + sampling 2.0 + speed 0.2, spacing 1.37
#   方案 B (task 1): 降采样 512->363 + sampling 1.0 + speed 0.2, spacing 1.932
#
set -euo pipefail
source ~/.bashrc
conda activate vascstats
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEG=/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_csam_seg.tiff
OUT_BASE=/share/home/sukm/experiments/vs_2pfm_skelgt/ab_compare
VOL=0.40936   # W2R_ws1 整图组织体积（与金标准同图）
EXE=/share/home/sukm/vascular-statistics/build/vessel_stats

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    METHOD="A_native512_s2.0"
    DOWNSAMPLE=0
    SAMPLING=2.0
    SPACING="1.37 1.37 2.0"
else
    METHOD="B_resize363_s1.0"
    DOWNSAMPLE=363
    SAMPLING=1.0
    SPACING="1.932 1.932 2.0"
fi

RUN=$OUT_BASE/$METHOD
mkdir -p "$RUN"
cd "$RUN"

echo "[$(date)] === 方案 $METHOD ==="

python3 << PYEOF
import sys, os, time, json
sys.path.insert(0, '/share/home/sukm/vascular-statistics/python')
import tifffile, numpy as np
from vascular_statistics.vascgraph import GraphIO, Skeletonize
from vascular_statistics.bridge import pajek_to_cpp_input
from VascGraph.Tools.CalcTools import fixG

arr = tifffile.imread('$SEG')
if arr.ndim == 3:
    arr = np.transpose(arr, (1, 2, 0))   # [D,H,W] -> [H,W,D]
stack = (arr > 0).astype(int)

downsample = $DOWNSAMPLE
if downsample > 0:
    from scipy.ndimage import zoom
    h = stack.shape[0]
    zf = downsample / float(h)
    stack = (zoom(stack.astype(np.uint8), (zf, zf, 1), order=0) > 0).astype(int)

fg = int(stack.sum())
print(f'  shape={stack.shape}, fg={fg:,}', flush=True)

t0 = time.time()
sk = Skeletonize.Skeleton(label=stack, sampling=$SAMPLING,
                          speed_param=0.2, dist_param=0.5, med_param=0.5)
sk.Update()
graph = fixG(sk.GetOutput())
t1 = time.time()
print(f'  skeleton: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {t1-t0:.1f}s', flush=True)

GraphIO.WritePajek(path='', name='skeleton.pajek', graph=graph)
pajek_to_cpp_input('skeleton.pajek', 'skeleton_edges.txt', 'skeleton_vertices.txt')

json.dump({
    'method': '$METHOD', 'downsample': downsample, 'sampling': $SAMPLING,
    'speed_param': 0.2, 'spacing_um': [float(x) for x in '$SPACING'.split()],
    'r_scale': (float('$SPACING'.split()[0]) + float('$SPACING'.split()[1]))/2,
    'stats_unit_mode': 'anisotropic',
    'foreground_voxels': fg, 'skeleton_time_s': round(t1-t0,1),
    'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(),
    'volume_mm3': $VOL,
}, open('run_meta.json','w'), indent=2)
PYEOF

echo "[$(date)] C++ stats (spacing $SPACING)..."
$EXE skeleton_edges skeleton_vertices $VOL $SPACING

# 子范围统计
python3 -c "
import sys; sys.path.insert(0,'/share/home/sukm/vascular-statistics/python')
from vascular_statistics.subrange_stats import write_subrange_stats
write_subrange_stats('.', $VOL)
"

echo "[$(date)] === $METHOD DONE ==="
echo '--- d10+um summary ---'
grep -E '平均直径|中位直径|平均长度|中位长度|段密度|平均弯曲度|中位弯曲度|有效段数' statistics_summary_d10+um.txt
