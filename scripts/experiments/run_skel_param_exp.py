"""
骨架化参数实验（sampling × speed）—— 2PFM 金标准匹配

对 2PFM 清洁金标准样本，用金标准原生网格（W1N/W2N=363，W2R=382）：
  CSAM seg(512) → resize 到金标准网格 → LFD 骨架化(sampling,speed)
  → .pajek + 挂墙耗时 → skeleton_match_3d(物理 μm, δ) vs 金标准 .pajek
  CSAM seg(512) → resize 384(order=1+阈值) → LFD 骨架化(sampling,speed)
  → .pajek + 挂墙耗时 → skeleton_match_3d(物理 μm, δ) vs 金标准 .pajek
  → 健康度 + 匹配度(completeness/correctness/quality)

样本选择（用户 2.1）：排除人工过度连接的问题样本（W1N_ref/W2N_ws1/W2R_ref，
生理畸变），仅用清洁金标准。

匹配口径（算法已对 Chap_4 §4.7 逐式复核）：
  - 预测骨架在 363/382 网格，有效间距 = FOV/363 或 FOV/382（金标准网格）
  - 金标准在原生网格（W1N/W2N=363→1.932；W2R=382→1.836），z=2.0
  - δ=3.7μm（≈金标准 2px，对齐 Chap_4 2 像素容差）

用法（vascstats 环境）：
  python scripts/experiments/run_skel_param_exp.py --out-dir <...> [--samples W1N_ws1 ...] [--limit-config N]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
import numpy as np
import tifffile
from scipy.ndimage import zoom

from skeleton_match_3d import parse_pajek, skeleton_health, skeleton_matching

SEG_DIR = '/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg'
GOLD_DIR = '/share/home/sukm/datasets/2PFM_SkelGT/skeletons'
FOV_UM = 512 * 1.37  # 2PFM 横向 FOV（μm）= 701.44
DELTA_UM = 3.7

# 清洁金标准（用户 2.1：排除 W1N_ref/W2N_ws1/W2R_ref 过度连接畸变样本）
CLEAN = ['W1N_20190920_ws1', 'W2N_20190911_ref', 'W2N_20190911_ws2', 'W2R_20190903_ws1']
# 金标准原生 XY 网格 + 对应分割输入尺寸（用户 1.2）：
#   骨架化半径计算依赖网格正确性——须用金标准原 grid，非 384
#   W1N/W2N = 363 grid(1.932μm/px)，W2R = 382 grid(1.836μm/px)
SAMPLE_GRID = {
    'W1N_20190920_ws1': 363,
    'W2N_20190911_ref': 363,
    'W2N_20190911_ws2': 363,
    'W2R_20190903_ws1': 382,
}

# sampling × speed 配置（覆盖快慢两极 + 单变量对照）
CONFIGS = [
    {'sampling': 1.0, 'speed': 0.05},   # 最慢/最密
    {'sampling': 1.5, 'speed': 0.10},   # 中
    {'sampling': 2.0, 'speed': 0.20},   # 最快/最疏（2PFM 已验证 ~0.9h）
    {'sampling': 2.0, 'speed': 0.10},   # 测 speed 独立
    {'sampling': 1.5, 'speed': 0.20},   # 测 sampling 独立
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--samples', nargs='*', default=CLEAN)
    ap.add_argument('--limit-config', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    configs = CONFIGS[:args.limit_config] if args.limit_config else CONFIGS

    results = []
    for samp in args.samples:
        seg_tif = os.path.join(SEG_DIR, samp + '_csam_seg.tiff')
        if not os.path.exists(seg_tif):
            cand = [f for f in os.listdir(SEG_DIR) if samp in f and f.endswith('.tiff')]
            if not cand:
                print(f'[SKIP] {samp}: 无 CSAM seg'); continue
            seg_tif = os.path.join(SEG_DIR, cand[0])
        gold = os.path.join(GOLD_DIR, samp + '.pajek')
        if not os.path.exists(gold):
            print(f'[SKIP] {samp}: 无金标准 .pajek'); continue

        grid_size = SAMPLE_GRID.get(samp, 363)           # 用户 1.2：须用金标准原 grid
        spacing = [round(FOV_UM / grid_size, 4)] * 2 + [2.0]  # pred 与 gold 同网格
        gc, ge, gmap = parse_pajek(gold)

        # 读 seg(512, [D,H,W])→[H,W,D]，resize XY 到金标准网格
        arr = tifffile.imread(seg_tif)
        seg = np.transpose(arr, (1, 2, 0)) if arr.ndim == 3 else arr[..., None]
        seg = (seg > 0).astype(np.float32)
        f = grid_size / seg.shape[0]
        seg_resized = (zoom(seg, (f, f, 1), order=1) > 0.5).astype(np.uint8) * 255

        sdir = os.path.join(args.out_dir, samp)
        os.makedirs(sdir, exist_ok=True)
        seg_tif_out = os.path.join(sdir, f'seg{grid_size}.tiff')
        tifffile.imwrite(seg_tif_out, np.transpose(seg_resized, (2, 0, 1)), photometric='minisblack')

        for cfg in configs:
            tag = f"s{cfg['sampling']}_sp{cfg['speed']}"
            pajek_out = os.path.join(sdir, f'skel_{tag}.pajek')
            print(f'=== {samp} / {tag} 骨架化 (grid={grid_size}) ===', flush=True)
            t0 = time.time()
            rc = subprocess.run(
                [sys.executable, '-m', 'vascular_statistics.cli', 'skeletonize', seg_tif_out,
                 '-o', pajek_out, '--sampling', str(cfg['sampling']), '--speed', str(cfg['speed'])],
                capture_output=True, text=True)
            elapsed = time.time() - t0
            if rc.returncode != 0 or not os.path.exists(pajek_out):
                print(f'  [FAILED] rc={rc.returncode}: {rc.stderr[-300:]}')
                results.append({'sample': samp, 'config': tag, 'elapsed_s': elapsed,
                                'status': 'failed', 'stderr': rc.stderr[-300:]})
                continue
            pc, pe, pmap = parse_pajek(pajek_out)
            health = skeleton_health(pc, pe, pmap)
            match = skeleton_matching(pc, gc, spacing, spacing, DELTA_UM)
            rec = {'sample': samp, 'config': tag, 'sampling': cfg['sampling'], 'speed': cfg['speed'],
                   'grid_size': grid_size, 'elapsed_s': round(elapsed, 1), 'status': 'ok',
                   'health': health, 'matching': match,
                   'spacing': spacing, 'delta_um': DELTA_UM}
            results.append(rec)
            print(f"  耗时 {elapsed:.0f}s | grid={grid_size} 节点 {health['n_nodes']} | "
                  f"comp={match['completeness']:.3f} corr={match['correctness']:.3f} "
                  f"qual={match['quality']:.3f} | defects={health['total_defects']}", flush=True)

    json.dump(results, open(os.path.join(args.out_dir, 'skel_param_results.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n完成 {len(results)} 条 → {args.out_dir}/skel_param_results.json')


if __name__ == '__main__':
    main()
