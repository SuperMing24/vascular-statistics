"""
从 raw 重切 cropped_z —— 裁切与预处理解耦

依据 cropped_z 的 sample_catalog.json（含 z_start/z_end），定位对应 raw
angiogram.mat，按 frameRange 无损 z 切，再应用可选预处理，写盘镜像 cropped_z 结构。

用法：
  # 全量生成 cropped_z_raw（纯 z 切，无强度预处理）
  python scripts/recrop_from_raw.py \\
    --crop-catalog /share/.../huaien/cropped_z/sample_catalog.json \\
    --raw-root     /share/.../huaien/raw \\
    --out-root     /share/.../huaien/cropped_z_raw \\
    --preprocess none

  # A/B：仅对若干样本产出某预处理变体
  python scripts/recrop_from_raw.py ... --preprocess percentile \\
    --out-root /share/.../_ab/huaien_percentile --limit 5

raw 路径推导（member 无关）：
  cropped input_rel_path = {dir}/{stem}_crop_{zs}_{ze}.mat
  raw_rel = {dir}/{stem 去 _crop_zs_ze}.mat
  （huaien: .../angiogram/angiogram.mat；xiaoqian: .../{date}_angiogram.mat）
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from vascular_statistics.image_preprocess import (  # noqa: E402
    apply_preprocess,
    volume_intensity_summary,
    zcrop,
)


def derive_raw_candidates(crop_rel: str) -> list:
    """从 cropped 相对路径推导对应 raw 相对路径候选（按优先级）。

    huaien cropped_z 结构不一致（部分在 .../angiogram/ 下、部分直接在 date_dir 下），
    而 raw 统一在 .../angiogram/angiogram.mat。故返回多候选，由调用方取存在者：
      1. 同目录 base.mat（xiaoqian 扁平 + huaien 含 angiogram/ 子目录）
      2. 同目录插入 angiogram/ 子目录（huaien 不含 angiogram/ 子目录的样本）
    """
    p = Path(crop_rel)
    base = p.stem.split("_crop_")[0]
    return [
        str(p.with_name(base + ".mat")).replace("\\", "/"),
        str(p.parent / "angiogram" / (base + ".mat")).replace("\\", "/"),
    ]


def load_stack(mat_path: str) -> np.ndarray:
    raw = sio.loadmat(mat_path)
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v
    raise ValueError(f"未找到 3D stack: {mat_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-catalog", required=True, help="cropped_z 的 sample_catalog.json")
    ap.add_argument("--raw-root", required=True, help="该成员 raw/ 根目录")
    ap.add_argument("--out-root", required=True, help="输出根目录（镜像 cropped_z 结构）")
    ap.add_argument("--preprocess", default="none", choices=["none", "bgsub", "percentile"])
    ap.add_argument("--limit", type=int, default=0, help=">0 时仅处理前 N 个样本（A/B 用）")
    ap.add_argument("--sample-keys", nargs="*", default=None, help="仅处理指定 sample_key（A/B 用）")
    args = ap.parse_args()

    catalog = json.load(open(args.crop_catalog, encoding="utf-8"))
    samples = catalog["samples"]
    keys = list(samples.keys())
    if args.sample_keys:
        keys = [k for k in keys if k in set(args.sample_keys)]
    if args.limit > 0:
        keys = keys[: args.limit]

    print(f"待处理 {len(keys)} 样本，preprocess={args.preprocess}\n")

    ok, failed = 0, 0
    for i, sk in enumerate(keys, 1):
        meta = samples[sk]
        crop_rel = meta["input_rel_path"]
        zs, ze = meta.get("z_start"), meta.get("z_end")
        if zs is None or ze is None:
            print(f"  [SKIP] {sk}: catalog 缺 z_start/z_end")
            failed += 1
            continue
        cands = derive_raw_candidates(crop_rel)
        raw_path = next((os.path.join(args.raw_root, r) for r in cands
                         if os.path.exists(os.path.join(args.raw_root, r))), None)
        if raw_path is None:
            print(f"  [FAILED] {sk}: raw 不存在（候选 {cands}）")
            failed += 1
            continue
        try:
            stack = load_stack(raw_path)
            cropped = zcrop(stack, zs, ze)              # 无损 z 切（保留 raw dtype）
            processed = apply_preprocess(cropped, args.preprocess)
            # none 保留 raw dtype；bgsub/percentile 为 float32
            out_rel = crop_rel
            out_path = os.path.join(args.out_root, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sio.savemat(out_path, {
                "stack": processed,
                "frameRange": np.array([[zs, ze]], dtype=np.uint16),
            }, do_compression=True)
            s = volume_intensity_summary(processed)
            print(f"  [{i}/{len(keys)}] {sk}  shape={s['shape']} "
                  f"min={s['min']:.1f} max={s['max']:.1f} zero%={s['zero_frac']*100:.1f}")
            ok += 1
        except Exception as e:
            print(f"  [FAILED] {sk}: {e}")
            failed += 1

    print(f"\n完成: {ok} 成功, {failed} 失败 → {args.out_root}")


if __name__ == "__main__":
    main()
