#!/usr/bin/env python3
"""Render per-sample coordinate audit PNGs for a manual-cut experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from vascular_statistics.manual_cut_review import (  # noqa: E402
    discover_manual_cut_metadata,
    render_sample_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--sample", action="append", default=[], metavar="UNIFIED_SAMPLE_KEY",
        help="render only this exact destination sample key; may be repeated",
    )
    parser.add_argument("--output-name", default="manual_cut_review.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if os.path.basename(args.output_name) != args.output_name:
        parser.error("--output-name must be a filename, not a path")
    if not args.output_name.casefold().endswith(".png"):
        parser.error("--output-name must end with .png")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    requested = set(args.sample)
    selected: list[tuple[str, str]] = []
    found: set[str] = set()
    for meta_path in discover_manual_cut_metadata(args.experiment_root):
        with open(meta_path, "r", encoding="utf-8") as handle:
            sample_key = str(json.load(handle).get("destination_sample_key", ""))
        if not requested or sample_key in requested:
            selected.append((sample_key, meta_path))
            found.add(sample_key)
    missing = sorted(requested - found)
    if missing:
        parser.error(f"sample key(s) not found: {missing}")

    processed = 0
    skipped = 0
    failed = 0
    mismatched = 0
    for sample_key, meta_path in selected:
        output_path = os.path.join(os.path.dirname(meta_path), args.output_name)
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            print(f"[skip] {sample_key}: {output_path}")
            continue
        try:
            summary = render_sample_review(
                meta_path, output_path=output_path, dpi=args.dpi,
                overwrite=args.overwrite,
            )
            processed += 1
            if not summary.all_run_positions_match:
                mismatched += 1
            state = "match" if summary.all_run_positions_match else "MISMATCH"
            print(
                f"[rendered] {sample_key}: runs={summary.run_count} "
                f"panels={summary.panel_count} positions={state} "
                f"layout=nonoverlap {summary.output_path}"
            )
        except Exception as exc:
            failed += 1
            print(f"[failed] {sample_key}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    print(
        f"[summary] selected={len(selected)} processed={processed} skipped={skipped} "
        f"failed={failed} position_mismatches={mismatched}"
    )
    if failed or mismatched:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
