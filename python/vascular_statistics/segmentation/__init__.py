"""
血管分割桥接模块 —— 为骨架化管线提供分割预处理。

管线位置：
    .mat 原图 → [本模块] → 二值掩码 .tif → VascGraph Skeletonize → ...

核心用法：
    >>> from vascular_statistics.segmentation import segment_mat
    >>> mask_path = segment_mat("sample.mat", output_dir="./output")

模块结构：
    _pipeline.py    — 一站式 segment_mat() 编排
    convert.py      — .mat → .nii 格式转换
    preprocess.py   — 强度归一化策略（可插拔）
    inference.py    — subprocess 调用 Vascular_Extraction predict.py
    postprocess.py  — 掩码加载 / 小连通域去除 / 质量报告

模型选型（2026-06-07）：
    nnU-Net 2D (seed22, epoch 41) — MiniVess 验证集 6 项指标全第一
    详见 docs/model_comparison_minivess_20260607.md

依赖说明：
    子模块依赖 nibabel / tifffile / scipy，仅在服务器端（vesseg 环境）可用。
    本地 vascstats 环境不安装这些包。通过 __getattr__ 延迟加载，
    保证 `import vascular_statistics.segmentation` 本身不失败。
"""

# ── 延迟加载映射 ──────────────────────────────────────────
_module_imports = {
    "mat_to_nii":        ("vascular_statistics.segmentation.convert", "mat_to_nii"),
    "mat_info":          ("vascular_statistics.segmentation.convert", "mat_info"),
    "normalize_volume":  ("vascular_statistics.segmentation.preprocess", "normalize_volume"),
    "volume_stats":      ("vascular_statistics.segmentation.preprocess", "volume_stats"),
    "NormalizeStrategy": ("vascular_statistics.segmentation.preprocess", "NormalizeStrategy"),
    "run_segmentation":  ("vascular_statistics.segmentation.inference", "run_segmentation"),
    "load_mask":         ("vascular_statistics.segmentation.postprocess", "load_mask"),
    "crop_mask":         ("vascular_statistics.segmentation.postprocess", "crop_mask"),
    "mask_stats":        ("vascular_statistics.segmentation.postprocess", "mask_stats"),
    "remove_small_components": ("vascular_statistics.segmentation.postprocess", "remove_small_components"),
    "segment_mat":       ("vascular_statistics.segmentation._pipeline", "segment_mat"),
}


def __getattr__(name):
    if name in _module_imports:
        mod_name, attr_name = _module_imports[name]
        import importlib
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_imports.keys())
