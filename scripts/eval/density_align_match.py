"""
密度对齐重评 —— 消除 LFD 节点密度失配对 node-based 匹配的偏置

问题：skeleton_match_3d 的 completeness/correctness/quality 是节点级 KDTree
最近邻，受两边节点密度直接影响（quality 分母含全部预测节点 → 节点越多越吃亏；
completeness 随预测越密越虚高）。LFD 输出节点数是人工金标准的 1.5–3.3×，故原始
绝对匹配度有偏。

方法：对 pred 与 gold 的 .pajek，**先转物理 μm，再沿每条边按固定弧长 step 重采样**
（中点采样，不双计交点），使两边节点密度统一为「每 step μm 一点」，再算匹配。
step ≪ δ 时，node-based 指标逼近「中心线-中心线」连续重叠度，与原始离散密度无关。

⚠️ 非破坏性：只读原 .pajek，重采样点云在内存中算，**不改、不覆盖原骨架文件**。

用法（vascstats 环境，集群）：
  python scripts/eval/density_align_match.py \
      --base-dir /share/home/sukm/experiments/vs_skel_param \
      --gold-dir /share/home/sukm/datasets/2PFM_SkelGT/skeletons \
      --out <base-dir>/density_align_results.json [--step 1.0]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from skeleton_match_3d import parse_pajek, skeleton_matching

# 与 run_skel_param_exp.py 同口径
FOV_UM = 512 * 1.37  # 701.44
DELTA_UM = 3.7
CLEAN = ['W1N_20190920_ws1', 'W2N_20190911_ref', 'W2N_20190911_ws2', 'W2R_20190903_ws1']
SAMPLE_GRID = {
    'W1N_20190920_ws1': 363,
    'W2N_20190911_ref': 363,
    'W2N_20190911_ws2': 363,
    'W2R_20190903_ws1': 382,
}
CONFIGS = ['s1.0_sp0.05', 's1.5_sp0.1', 's2.0_sp0.2', 's2.0_sp0.1', 's1.5_sp0.2']


def resample_edges(coords_um: np.ndarray, edges, pid2idx, step: float) -> np.ndarray:
    """沿每条边按弧长 step 中点重采样 → 均匀密度点云（μm）。

    每条边长 L → n=max(1, round(L/step)) 个点，参数 t=(i+0.5)/n（中点采样，
    不在共享端点处双计）。节点数 ≈ 总血管长 / step，与原始离散密度无关。
    孤立节点（不在任何边上）被排除——只比有效中心线。
    """
    pts = []
    for a, b in edges:
        if a not in pid2idx or b not in pid2idx:
            continue
        ia, ib = pid2idx[a], pid2idx[b]
        if ia == ib:
            continue
        pa, pb = coords_um[ia], coords_um[ib]
        L = float(np.linalg.norm(pb - pa))
        n = max(1, int(round(L / step)))
        for i in range(n):
            t = (i + 0.5) / n
            pts.append(pa + t * (pb - pa))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3))


def to_um(coords: np.ndarray, spacing) -> np.ndarray:
    return coords * np.asarray(spacing, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='/share/home/sukm/experiments/vs_skel_param')
    ap.add_argument('--gold-dir', default='/share/home/sukm/datasets/2PFM_SkelGT/skeletons')
    ap.add_argument('--out', default=None)
    ap.add_argument('--step', type=float, default=1.0, help='重采样弧长 μm（默认 1.0，≪ δ=3.7）')
    ap.add_argument('--samples', nargs='*', default=CLEAN)
    args = ap.parse_args()
    out = args.out or os.path.join(args.base_dir, 'density_align_results.json')

    results = []
    for samp in args.samples:
        grid = SAMPLE_GRID.get(samp, 363)
        spacing = [round(FOV_UM / grid, 4)] * 2 + [2.0]
        gold_path = os.path.join(args.gold_dir, samp + '.pajek')
        if not os.path.exists(gold_path):
            print(f'[SKIP] {samp}: 无 gold {gold_path}'); continue
        gc, ge, gmap = parse_pajek(gold_path)
        gold_um = to_um(gc, spacing)
        gold_rs = resample_edges(gold_um, ge, gmap, args.step)

        for cfg in CONFIGS:
            pred_path = os.path.join(args.base_dir, samp, f'skel_{cfg}.pajek')
            if not os.path.exists(pred_path):
                print(f'[SKIP] {samp}/{cfg}: 无 pred {pred_path}'); continue
            pc, pe, pmap = parse_pajek(pred_path)
            pred_um = to_um(pc, spacing)
            pred_rs = resample_edges(pred_um, pe, pmap, args.step)

            # 已是 μm，spacing=1 退化；δ 不变
            m = skeleton_matching(pred_rs, gold_rs, (1, 1, 1), (1, 1, 1), DELTA_UM)
            rec = {
                'sample': samp, 'config': cfg, 'grid_size': grid, 'step_um': args.step,
                'n_pred_raw': len(pc), 'n_gold_raw': len(gc),
                'n_pred_rs': len(pred_rs), 'n_gold_rs': len(gold_rs),
                'completeness': round(m['completeness'], 4),
                'correctness': round(m['correctness'], 4),
                'quality': round(m['quality'], 4),
            }
            results.append(rec)
            print(f"{samp}/{cfg}: raw {len(pc)}/{len(gc)} → rs {len(pred_rs)}/{len(gold_rs)} | "
                  f"comp={rec['completeness']:.3f} corr={rec['correctness']:.3f} "
                  f"qual={rec['quality']:.3f}", flush=True)

    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\n完成 {len(results)} 条 → {out}')


if __name__ == '__main__':
    main()
