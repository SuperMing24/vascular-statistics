"""
批量分割（带 XY 重采样）—— cropped_z .mat → 指定尺寸分割掩码

对某成员 cropped_z 目录下所有样本：按 catalog 读 .mat → resize XY 到指定尺寸
→ nnU-Net 2D 分割 → 保存掩码。用于 huaien/xiaoqian cropped_z 的 384 分割
（依据 docs/param_selection_seg_resample_report_20260624.md：384 −2.3% 质量
换 ~44% 体素省，服务下游骨架化提速）。

掩码保存在重采样尺寸（如 384），供骨架化直接在该网格上做（提速目的）。
有效 XY 间距 = 原生间距 × 512/尺寸，写入 run_meta.json 供下游统计。

运行（vesseg 环境，GPU 节点）：
  python scripts/seg_batch_resample.py \\
    --crop-root /share/.../huaien/cropped_z --resize 384 \\
    --out-root  /share/.../experiments/lab_data_seg_384/huaien
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/share/home/sukm/Vascular_Extraction')
import numpy as np
import nibabel as nib
import scipy.io as sio
import tifffile
from scipy.ndimage import zoom

from src.scripts.predict import InferenceManager

EXP = '/share/home/sukm/experiments/ve_phase0_baseline/nnunet_2d_bce_dice_slice3_bs4/seed22'
NATIVE_XY = [1.1142, 1.0652]  # 原生 XY 间距 μm/px（全实验室 2PFM 标定）


def load_stack(mat_path):
    raw = sio.loadmat(mat_path)
    for k, v in raw.items():
        if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
            return v
    raise ValueError(f'无 3D stack: {mat_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop-root', required=True, help='成员 cropped_z 根（含 sample_catalog.json）')
    ap.add_argument('--resize', type=int, default=384)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    catalog = json.load(open(os.path.join(args.crop_root, 'sample_catalog.json'), encoding='utf-8'))
    samples = list(catalog['samples'].items())
    if args.limit > 0:
        samples = samples[:args.limit]
    X = args.resize
    print(f'样本 {len(samples)}，resize {X}', flush=True)

    mgr = InferenceManager(EXP)
    eff_xy = [round(NATIVE_XY[0] * 512 / X, 4), round(NATIVE_XY[1] * 512 / X, 4)]
    ok, failed = 0, 0
    for i, (sk, meta) in enumerate(samples, 1):
        rel = meta['input_rel_path']
        mat = os.path.join(args.crop_root, rel)
        if not os.path.exists(mat):
            print(f'  [FAILED] {sk}: 缺 {mat}'); failed += 1; continue
        try:
            v = load_stack(mat).astype(np.float32)            # 512×512×N
            vr = v if X == 512 else zoom(v, (X / 512, X / 512, 1), order=1)
            out_rel = os.path.splitext(rel)[0]
            outdir = os.path.join(args.out_root, os.path.dirname(out_rel))
            os.makedirs(outdir, exist_ok=True)
            tmp = os.path.join(outdir, '_tmp.nii')
            nib.save(nib.Nifti1Image(vr.astype(np.float32), np.eye(4)), tmp)
            prob = mgr.infer_volume_prob(tmp)                 # X×X×N
            os.remove(tmp)
            mask = (prob > args.threshold).astype(np.uint8) * 255
            tif = os.path.join(args.out_root, out_rel + '_seg.tiff')
            tifffile.imwrite(tif, np.transpose(mask, (2, 0, 1)), photometric='minisblack')
            fg = int((prob > args.threshold).sum())
            print(f'  [{i}/{len(samples)}] {sk}  seg{list(mask.shape)} fg={fg:,}', flush=True)
            ok += 1
        except Exception as e:
            print(f'  [FAILED] {sk}: {e}'); failed += 1

    meta = {
        'resize': X, 'native_xy_um': NATIVE_XY, 'eff_xy_um': eff_xy, 'z_um_note': '见各样本 catalog z 间距',
        'model': EXP, 'threshold': args.threshold, 'crop_root': args.crop_root,
        'n_ok': ok, 'n_failed': failed,
    }
    json.dump(meta, open(os.path.join(args.out_root, 'run_meta.json'), 'w'), indent=2, ensure_ascii=False)
    print(f'\n完成: {ok} 成功, {failed} 失败；有效XY间距={eff_xy} μm/px → {args.out_root}')


if __name__ == '__main__':
    main()
