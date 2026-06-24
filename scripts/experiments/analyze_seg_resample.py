"""
分割采样尺寸实验 —— 分析与出图

读取各尺寸 eval_summary.json（含 per_sample），生成：
  - results_all_sizes.csv：跨尺寸汇总（尺寸 × 指标 × 综合评分）
  - results_by_um_per_px.csv：按有效 μm/px 重组（迁移用）
  - 4 张图：评分 vs 像素尺寸 / vs 有效μm·px / 指标分项 / τ 敏感性

综合评分 = HM(Dice, clDice3D, BoundaryScore)，BoundaryScore=1/(1+HD95/τ)。
图用英文标注（避免 CJK 字体缺失）。

用法：
  python analyze_seg_resample.py --data-dir <含 size_*.json 的目录> --out-dir <输出>
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 目标数据集的原生 XY 分辨率（μm/px），用于迁移标注
TARGETS = {'huaien/xiaoqian (~1.09)': 1.09, '2PFM_SkelGT (1.37)': 1.37}
TAUS = [10, 20, 40]


def boundary_score(hd95, tau):
    return 1.0 / (1.0 + hd95 / tau) if hd95 is not None and hd95 >= 0 else 0.0


def composite3(d, c, hd, tau):
    b = boundary_score(hd, tau)
    if min(d, c, b) <= 0:
        return 0.0
    return 3.0 / (1.0 / d + 1.0 / c + 1.0 / b)


def hm2(d, c):
    return 2 * d * c / (d + c) if (d + c) > 0 else 0.0


def load(data_dir):
    out = {}
    for f in glob.glob(os.path.join(data_dir, 'size_*.json')):
        c = json.load(open(f, encoding='utf-8'))
        out[c['size']] = c
    return dict(sorted(out.items(), key=lambda kv: -kv[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sizes = load(args.data_dir)
    SZ = sorted(sizes.keys(), reverse=True)

    def m(s, k):
        return sizes[s]['summary'].get(k, {}).get('mean', 0.0)

    # ── results_all_sizes.csv ──
    keys = ['dice', 'iou', 'cl_dice', 'cl_dice_3d', 'HD95_bound', 'sensitivity', 'composite']
    with open(os.path.join(args.out_dir, 'results_all_sizes.csv'), 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(['size', 'linear_ratio', 'pixel_ratio'] + keys + ['comp_tau10', 'comp_tau40', 'hm2'])
        for s in SZ:
            d, c, hd = m(s, 'dice'), m(s, 'cl_dice_3d'), m(s, 'HD95_bound')
            w.writerow([s, round(s / 512, 3), round((s / 512) ** 2, 3)]
                       + [round(m(s, k), 4) for k in keys]
                       + [round(composite3(d, c, hd, 10), 4), round(composite3(d, c, hd, 40), 4),
                          round(hm2(d, c), 4)])

    # ── per-sample 池化（μm/px 分析）──
    pts = []  # (eff_um_px, composite, size, native_um_px)
    native = []  # size=512 的 (native_um_px, composite)
    for s in SZ:
        for r in sizes[s]['per_sample']:
            ump = r.get('native_um_per_px')
            comp = r.get('composite')
            if ump is None or comp is None:
                continue
            eff = ump * 512 / s
            pts.append((eff, comp, s, ump))
            if s == 512:
                native.append((ump, comp))
    pts = np.array([(p[0], p[1]) for p in pts])
    native = np.array(native)

    with open(os.path.join(args.out_dir, 'results_by_um_per_px.csv'), 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(['eff_um_per_px_bin', 'n', 'composite_mean', 'composite_std'])
        bins = np.arange(0.3, 8.1, 0.5)
        idx = np.digitize(pts[:, 0], bins)
        for b in range(1, len(bins)):
            sel = pts[idx == b, 1]
            if len(sel):
                w.writerow([f'{bins[b-1]:.1f}-{bins[b]:.1f}', len(sel),
                            round(sel.mean(), 4), round(sel.std(), 4)])

    # ══ Fig 1: 综合评分 + Dice + clDice3D vs 像素尺寸 ══
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SZ, [m(s, 'composite') for s in SZ], 'o-', color='C3', lw=2, label='Composite (3-way HM, tau=20)')
    ax.plot(SZ, [m(s, 'dice') for s in SZ], 's--', color='C0', label='Dice')
    ax.plot(SZ, [m(s, 'cl_dice_3d') for s in SZ], '^--', color='C2', label='clDice3D')
    ax.axvline(512, color='gray', ls=':', alpha=0.6)
    ax.annotate('native 512 (best)', (512, m(512, 'composite')), textcoords='offset points',
                xytext=(-10, 12), ha='right', color='C3')
    ax.set_xlabel('Segmentation input size (px)'); ax.set_ylabel('Score')
    ax.set_title('Segmentation quality vs input pixel size (MiniVess n=70, GT-aligned)')
    ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'score_vs_pixelsize.png'), dpi=130); plt.close(fig)

    # ══ Fig 2: 综合评分 vs 有效 μm/px（按输入尺寸着色，展示降采样轨迹）══
    # 注：同一 eff μm/px 混合"原生(分布内)"与"降采样(OOD)"两类点 → 不做池化均值
    #     （会被 OOD 混淆成锯齿）。改按尺寸着色：native 在左上(分布内高分)，
    #     降采样越多越往右下(OOD 退化)。
    pts_full = []
    for s in SZ:
        for r in sizes[s]['per_sample']:
            ump, comp = r.get('native_um_per_px'), r.get('composite')
            if ump is not None and comp is not None:
                pts_full.append((ump * 512 / s, comp, s))
    pf = np.array(pts_full)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sc = ax.scatter(pf[:, 0], pf[:, 1], c=pf[:, 2], s=12, alpha=0.6, cmap='viridis')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('input size (px)')
    if len(native):
        ax.scatter(native[:, 0], native[:, 1], s=45, facecolors='none', edgecolors='red',
                   linewidths=1.2, label='native (size 512, in-distribution)')
    for name, v in TARGETS.items():
        ax.axvline(v, color='C3', ls='--', alpha=0.7)
        ax.annotate(name, (v, 0.20), rotation=90, va='bottom', ha='right', color='C3', fontsize=8)
    ax.set_xlabel('Effective resolution (um/px)  = native_um_px x 512/size')
    ax.set_ylabel('Composite score'); ax.set_xlim(0, 8)
    ax.set_title('Composite vs effective um/px (colored by input size)\n'
                 'native=left/high (in-dist); downsampling shifts right+down (OOD)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'score_vs_um_per_px.png'), dpi=130); plt.close(fig)

    # ══ Fig 3: 指标分项 ══
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, st, lb in [('dice', 's-', 'Dice'), ('cl_dice', 'v-', 'clDice2D'),
                      ('cl_dice_3d', '^-', 'clDice3D'), ('sensitivity', 'd-', 'Sensitivity')]:
        ax.plot(SZ, [m(s, k) for s in SZ], st, label=lb)
    ax.set_xlabel('Input size (px)'); ax.set_ylabel('Overlap / topology'); ax.invert_xaxis()
    ax2 = ax.twinx(); ax2.plot(SZ, [m(s, 'HD95_bound') for s in SZ], 'x:', color='C5', label='HD95 (px)')
    ax2.set_ylabel('HD95 (px)')
    ax.set_title('Metric breakdown vs input size'); ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'metrics_breakdown.png'), dpi=130); plt.close(fig)

    # ══ Fig 4: τ 敏感性 ══
    fig, ax = plt.subplots(figsize=(8, 5))
    for tau, st in zip(TAUS, ['o-', 's-', '^-']):
        ax.plot(SZ, [composite3(m(s, 'dice'), m(s, 'cl_dice_3d'), m(s, 'HD95_bound'), tau) for s in SZ],
                st, label=f'Composite tau={tau}')
    ax.plot(SZ, [hm2(m(s, 'dice'), m(s, 'cl_dice_3d')) for s in SZ], 'k--', label='HM(Dice,clDice3D) [no HD95]')
    ax.set_xlabel('Input size (px)'); ax.set_ylabel('Score'); ax.invert_xaxis()
    ax.set_title('tau sensitivity of composite (ranking stability)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'tau_sensitivity.png'), dpi=130); plt.close(fig)

    print('图与 CSV 已写入', args.out_dir)
    # 打印各 τ 下的最优尺寸（排名稳健性检查）
    for tau in TAUS + ['hm2']:
        if tau == 'hm2':
            vals = {s: hm2(m(s, 'dice'), m(s, 'cl_dice_3d')) for s in SZ}
        else:
            vals = {s: composite3(m(s, 'dice'), m(s, 'cl_dice_3d'), m(s, 'HD95_bound'), tau) for s in SZ}
        best = max(vals, key=vals.get)
        print(f'  {tau}: 最优尺寸={best} (score={vals[best]:.4f})')


if __name__ == '__main__':
    main()
