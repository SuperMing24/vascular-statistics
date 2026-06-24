"""
分割采样尺寸实验驱动

对 MiniVess 全 70 例 × 12 档尺寸：resize XY → 分割 → 上采样回512 → 对原生GT评测
→ 三元综合评分。端到端原生分辨率衡量（nnU-Net 标准）。

GT 处理：GT 永远保持原生 512×512 不动；只把预测概率从 size-X 上采样回 512
（nnU-Net 严格版 order=3 cubic spline），在原生分辨率对真值评测——非缩放 GT。

原始推理保留：每例保存 size-X 推理概率到 infer_prob/（uint8 .nii.gz）。
换上采样方式/阈值/指标可零成本重评，无需再推理。

运行（vesseg 环境，GPU 节点）：
  python seg_resample_eval.py [--sizes 256 ...] [--limit N]
"""
import argparse, glob, json, os, sys
from collections import defaultdict

sys.path.insert(0, '/share/home/sukm/Vascular_Extraction')
sys.path.insert(0, '/share/home/sukm/vascular-statistics/python')

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

from src.scripts.predict import InferenceManager
from src.train_tools.metrics import (
    IoUScore, DiceScore, CLDiceScore, HausdorffDistance,
    AccuracyScore, SensitivityScore, CLDice3DScore, MetricEvaluator,
)
from vascular_statistics.composite_score import composite_score

EXP  = '/share/home/sukm/experiments/ve_phase0_baseline/nnunet_2d_bce_dice_slice3_bs4/seed22'
ROOT = '/share/home/sukm/datasets/MiniVess'
OUT  = '/share/home/sukm/experiments/vs_seg_resample'
ALL_SIZES = [512, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128]
TAU = 20.0   # HD95_bound 为 512 网格像素(典型5-15px)，τ 匹配该尺度；边界作守卫不主导。
             # 原始指标全保存，综合评分可事后按任意 τ 重算（分析时做 τ∈{10,20,40} 敏感性）
EPS = 1e-7


def gather_samples():
    out = []
    for split in ['train', 'val']:
        for img in sorted(glob.glob(f'{ROOT}/{split}/images/*.nii')):
            stem = os.path.basename(img)[:-4]
            mask = f'{ROOT}/{split}/masks/{stem}.nii'
            js = f'{ROOT}/json/{stem}.json'
            if not os.path.exists(mask):
                continue
            umpx = None
            if os.path.exists(js):
                umpx = json.load(open(js)).get('physical size x')
            out.append((stem, img, mask, split, umpx))
    return out


def eval_lite(prob, gt, threshold=0.5):
    """轻量评测（跳过慢的 SkeletonEvaluator）：2D 逐切片均值 + 3D clDice。"""
    H, W, D = prob.shape
    m2d = MetricEvaluator([
        IoUScore(), DiceScore(), CLDiceScore(),
        HausdorffDistance(percentile=95.0, method='boundary'),
        AccuracyScore(), SensitivityScore(),
    ])
    acc = defaultdict(list)
    for d in range(D):
        ps, gs = prob[:, :, d], gt[:, :, d]
        if gs.sum() == 0 and (ps < threshold).all():
            continue
        logit = np.log(np.clip(ps, EPS, 1 - EPS) / np.clip(1 - ps, EPS, 1 - EPS))
        lt = torch.from_numpy(logit).float().unsqueeze(0).unsqueeze(0)
        gtt = torch.from_numpy(gs).float().unsqueeze(0).unsqueeze(0)
        for k, v in m2d.evaluate(lt, gtt).items():
            if v is not None:
                acc[k].append(v)
    res = {k: float(np.mean(v)) for k, v in acc.items() if v}
    pd = np.transpose(prob, (2, 0, 1))
    gd = np.transpose(gt, (2, 0, 1))
    l3 = np.log(np.clip(pd, EPS, 1 - EPS) / np.clip(1 - pd, EPS, 1 - EPS))
    lt3 = torch.from_numpy(l3).float().unsqueeze(0).unsqueeze(0)
    gt3 = torch.from_numpy(gd).float().unsqueeze(0).unsqueeze(0)
    res['cl_dice_3d'] = float(CLDice3DScore(threshold).compute(lt3, gt3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='*', type=int, default=ALL_SIZES)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-masks', action='store_true')
    args = ap.parse_args()

    samples = gather_samples()
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f'样本 {len(samples)}, 尺寸 {args.sizes}', flush=True)

    mgr = InferenceManager(EXP)

    for X in args.sizes:
        sdir = f'{OUT}/per_size/size_{X}'
        os.makedirs(sdir, exist_ok=True)
        if args.save_masks:
            os.makedirs(f'{sdir}/masks', exist_ok=True)
        per = []
        for stem, img, mask, split, umpx in samples:
            v = nib.load(img).get_fdata().astype(np.float32)
            g = (nib.load(mask).get_fdata() > 0.5).astype(np.float32)
            vr = v if X == 512 else zoom(v, (X / 512, X / 512, 1), order=1)
            tmp = f'{sdir}/_tmp_{stem}.nii'
            nib.save(nib.Nifti1Image(vr.astype(np.float32), np.eye(4)), tmp)
            prob = mgr.infer_volume_prob(tmp)          # 原始推理概率（size-X 网格）
            os.remove(tmp)
            # 保存原始推理输出（size-X 概率，uint8 压缩）→ 今后换上采样/阈值/指标
            # 可零成本重评，无需再推理（设计要求：必须保留原始推理）
            os.makedirs(f'{sdir}/infer_prob', exist_ok=True)
            nib.save(nib.Nifti1Image((np.clip(prob, 0, 1) * 255).astype(np.uint8), np.eye(4)),
                     f'{sdir}/infer_prob/{stem}.nii.gz')
            # nnU-Net 严格版上采样：高阶 cubic spline(order=3) 上采样概率回原生 512，再阈值
            pr = prob if X == 512 else np.clip(zoom(prob, (512 / X, 512 / X, 1), order=3), 0, 1)
            # 形状对齐保护（zoom 取整可能差 1px）
            if pr.shape != g.shape:
                pr = pr[:g.shape[0], :g.shape[1], :g.shape[2]]
                if pr.shape != g.shape:
                    padw = [(0, g.shape[i] - pr.shape[i]) for i in range(3)]
                    pr = np.pad(pr, padw, mode='edge')
            m = eval_lite(pr, g)
            score = composite_score(m.get('dice', 0), m.get('cl_dice_3d', 0), m.get('HD95_bound'), TAU)
            rec = {'sample': stem, 'split': split, 'native_um_per_px': umpx,
                   'eff_um_per_px': (umpx * 512 / X if umpx else None),
                   'composite': score, **m}
            per.append(rec)
            if args.save_masks:
                import tifffile
                mk = (pr > 0.5).astype(np.uint8) * 255
                tifffile.imwrite(f'{sdir}/masks/{stem}.tiff',
                                 np.transpose(mk, (2, 0, 1)), photometric='minisblack')
            print(f'[{X}] {stem} dice={m.get("dice",0):.4f} cl3d={m.get("cl_dice_3d",0):.4f} '
                  f'HD95={m.get("HD95_bound",0):.1f} score={score:.4f}', flush=True)

        keys = ['dice', 'iou', 'cl_dice', 'cl_dice_3d', 'HD95_bound', 'sensitivity', 'composite']
        summ = {}
        for k in keys:
            vals = [r[k] for r in per if k in r and r[k] is not None]
            if vals:
                summ[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        json.dump({'size': X, 'tau': TAU, 'n': len(per), 'summary': summ, 'per_sample': per},
                  open(f'{sdir}/eval_summary.json', 'w'), indent=2, ensure_ascii=False)
        print(f'=== size {X} done: composite={summ.get("composite",{}).get("mean",0):.4f} '
              f'dice={summ.get("dice",{}).get("mean",0):.4f} '
              f'cl3d={summ.get("cl_dice_3d",{}).get("mean",0):.4f} ===', flush=True)

    print('=== ALL SIZES DONE ===', flush=True)


if __name__ == '__main__':
    main()
