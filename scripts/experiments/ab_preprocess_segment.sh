#!/bin/bash
# 预处理 A/B 分割 runner —— 对各预处理变体的样本跑 cli segment，打印掩码统计
# 用于 cropped_z 预处理 A/B（见 docs/preprocessing_ab_gt_report_20260624.md）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vascstats
AB=/share/home/sukm/experiments/ab_preprocess
for V in huaien_none huaien_bgsub huaien_percentile xiaoqian_none xiaoqian_bgsub xiaoqian_percentile; do
  for mat in $(find $AB/$V -name '*.mat' | sort); do
    stem=$(basename $mat .mat)
    echo "===ABSEG=== $V / $stem"
    python -m vascular_statistics.cli segment $mat -o $AB/seg/$V -n passthrough --stats --device cuda 2>&1 \
      | grep -E '前景|连通域|最大|平均|形状|失败|Error|Traceback'
  done
done
echo '===ABSEG=== DONE'
