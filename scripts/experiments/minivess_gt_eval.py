"""
MiniVess GT 验证 —— 预处理变体 × val 样本 → predict.py eval vs 真值

对每个预处理变体（none/bgsub/percentile），对 MiniVess val 图应用预处理后用
nnU-Net 2D seed22 跑 predict.py --eval，对真值算 Dice/clDice/IoU/HD95。

用于预处理 A/B 的 GT 验证（见 docs/preprocessing_ab_gt_report_20260624.md）。

运行（vesseg 环境，GPU 节点）：
  python scripts/experiments/minivess_gt_eval.py
"""
import sys, os, json, subprocess, glob

sys.path.insert(0, '/share/home/sukm/vascular-statistics/python')
import numpy as np
import nibabel as nib
from vascular_statistics.image_preprocess import apply_preprocess

VAL_IMG  = '/share/home/sukm/datasets/MiniVess/val/images'
VAL_MASK = '/share/home/sukm/datasets/MiniVess/val/masks'
EXP      = '/share/home/sukm/experiments/ve_phase0_baseline/nnunet_2d_bce_dice_slice3_bs4/seed22'
PREDICT  = '/share/home/sukm/Vascular_Extraction/src/scripts/predict.py'
AB       = '/share/home/sukm/experiments/vs_preproc_gt'


def main():
    imgs = sorted(glob.glob(VAL_IMG + '/*.nii'))
    print(f'val images: {len(imgs)}')
    for variant in ['none', 'bgsub', 'percentile']:
        vdir = f'{AB}/{variant}/images'
        os.makedirs(vdir, exist_ok=True)
        for f in imgs:
            v = nib.load(f).get_fdata().astype(np.float32)
            out = apply_preprocess(v, variant).astype(np.float32)
            nib.save(nib.Nifti1Image(out, np.eye(4)), f'{vdir}/{os.path.basename(f)}')
        outjson = f'{AB}/{variant}/eval.json'
        print(f'=== 评测 {variant} ===', flush=True)
        subprocess.run([sys.executable, PREDICT, '--exp_dir', EXP, '--input', vdir,
                        '--eval', 'true', '--mask_dir', VAL_MASK, '--out_json', outjson],
                       check=True)
    print('=== ALL DONE ===')


if __name__ == '__main__':
    main()
