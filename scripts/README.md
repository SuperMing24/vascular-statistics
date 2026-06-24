# scripts/ 目录组织

> 脚本定位规范。**所有脚本只能经 git 同步到集群，严禁 scp/heredoc 直写**（CLAUDE.md §0）。
> `experiments/`（实验结果）与 `datasets/`（数据集）目录**永不放脚本**——脚本一律在此 repo。

---

## 顶层：核心入口（reusable，被 docs / 彼此引用，保持稳定）

| 脚本 | 职责 |
|------|------|
| `pipeline.slurm` | 端到端管线 Slurm 模板（骨架化 → 转换 → 统计） |
| `batch_launch.sh` | 批量提交包装器（manifest + 断点续跑 + 过滤） |
| `run_pipeline.py` | 端到端管线 Python 入口 |
| `seg_batch_resample.py` | 批量分割（带 XY 重采样）：cropped_z .mat → 指定尺寸掩码 |

## `data/`：数据准备 + 元数据工具（reusable）

| 脚本 | 职责 |
|------|------|
| `extract_raw_metadata.py` | raw/cropped 元数据提取 → sample_catalog.json |
| `recrop_from_raw.py` | 从 raw 无损 Z 重切 + 可选强度预处理 |
| `resize_tiff.py` | TIFF 尺寸缩放 |
| `fix_2pfm_metadata.py` | 2PFM 金标准元数据修正 |
| `restats_anisotropic.py` | 各向异性口径重统计 |

## `eval/`：评估工具（reusable）

| 脚本 | 职责 |
|------|------|
| `skeleton_match_3d.py` | 3D 骨架健康度 + 匹配度（.pajek vs 金标准，物理 μm） |

## `experiments/`：实验 runner（一次性/特定实验，非通用管线）

| 脚本 | 实验 |
|------|------|
| `seg_resample_eval.py` + `run_seg_resample.slurm` | 分割采样尺寸选择（MiniVess） |
| `minivess_gt_eval.py` | 预处理 A/B 的 MiniVess GT 验证 |
| `ab_preprocess_segment.sh` | cropped_z 预处理 A/B 分割 |
| `ab_skel_compare.slurm` | 骨架化 A/B（W2R_ws1，历史） |
| `csam_infer_382.slurm` | CSAM 推理 382（W2R_ws1，历史） |
| `w2r_ws1_382.slurm` | W2R 骨架化 382（历史） |

## `tools/`：杂项工具

| 脚本 | 职责 |
|------|------|
| `survey_nodes.sh` | 集群全节点 CPU 检测 |

---

## 分类准则

- **reusable（顶层 / data/ / eval/ / tools/）**：可跨数据集/样本复用的管线与工具。
- **experiments/**：服务于某次特定实验、与具体数据/参数耦合的 runner。一次性产出实验结果后归档保留。
- 判断：「换个数据集还成立、还会再用」→ reusable；「只为这次实验跑一遍」→ experiments/。
