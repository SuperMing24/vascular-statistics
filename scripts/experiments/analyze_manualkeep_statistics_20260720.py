#!/usr/bin/env python3
"""Generate statistical artifacts for the 145-sample manual-keep experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from vascular_statistics.manualkeep_analysis import run_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--xiaoqian-source", type=Path, required=True)
    parser.add_argument("--huaien-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = run_analysis(
        args.final_root,
        args.output_root,
        {"xiaoqian": args.xiaoqian_source, "huaien": args.huaien_source},
        seed=args.seed,
        n_boot=args.bootstrap,
        overwrite=args.overwrite,
    )
    audit = payload["audit"]
    print(f"paired_samples={audit['paired_samples']}")
    print(f"metric_rows={audit['metric_rows']}")
    print(f"input_sha256={audit['input_sha256']}")
    print(f"output_root={args.output_root.resolve()}")


if __name__ == "__main__":
    main()
