"""
补全 2PFM_SkelGT 7 例 sample_metadata.json 的顶层字段，
使 aggregate 报告能正确显示 group/batch_id/daypoint。

用法（在集群上执行）:
    python scripts/fix_2pfm_metadata.py

从样本命名 W{N}_{YYYYMMDD}_{scan_type} 提取:
  - batch_id: W1N / W2N / W2R
  - daypoint: 日期（2PFM 以扫描日期为标识，非 BCAS D0/D14...）
  - tissue_type: "reference" | "watershed"
"""
import json
import os
import sys

OUTPUT_ROOT = "/share/home/sukm/experiments/vs_2pfm_skelgt"

SCAN_TYPE_MAP = {
    "ref": "reference",
    "ws1": "watershed",
    "ws2": "watershed",
}

SAMPLE_KEYS = [
    "W1N_20190920_ref", "W1N_20190920_ws1",
    "W2N_20190911_ref", "W2N_20190911_ws1", "W2N_20190911_ws2",
    "W2R_20190903_ref", "W2R_20190903_ws1",
]

fixed = 0
for sk in SAMPLE_KEYS:
    meta_path = os.path.join(OUTPUT_ROOT, sk, "sample_metadata.json")
    if not os.path.exists(meta_path):
        print(f"[SKIP] {sk} — 无 sample_metadata.json")
        continue

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pp = meta.get("path_parsed", {})
    subject = pp.get("subject", sk.split("_")[0])
    date = pp.get("date", sk.split("_")[1] if len(sk.split("_")) >= 2 else "?")
    scan_type_raw = pp.get("scan_type", sk.rsplit("_", 1)[-1])
    tissue_type = SCAN_TYPE_MAP.get(scan_type_raw, scan_type_raw)

    meta["batch_id"] = subject
    meta["daypoint"] = date
    meta["tissue_type"] = tissue_type
    # group 来自 path_parsed（已存在），确保顶层也有
    if "group" not in meta:
        meta["group"] = pp.get("group", "?")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[FIXED] {sk}: batch={subject}, daypoint={date}, "
          f"tissue_type={tissue_type}, group={meta['group']}")
    fixed += 1

print(f"\n完成: {fixed}/{len(SAMPLE_KEYS)} 样本")
