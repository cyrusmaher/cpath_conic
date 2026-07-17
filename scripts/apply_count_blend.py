#!/usr/bin/env python3
"""Apply a previously locked per-class count-blend recipe without label access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cpath_conic.constants import CLASS_NAMES
from scripts.sweep_count_ensemble import blended_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.selection_report.read_text())
    weights = np.asarray([report["selected_alpha_a"][name] for name in CLASS_NAMES], dtype=np.float64)
    counts_a = np.load(args.counts_a)
    counts_b = np.load(args.counts_b)
    if counts_a.shape != counts_b.shape or counts_a.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"incompatible count arrays: {counts_a.shape} and {counts_b.shape}")
    output = blended_counts(counts_a, counts_b, weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, output)
    args.out.with_suffix(".json").write_text(json.dumps({
        "selection_report": str(args.selection_report),
        "weights_a": dict(zip(CLASS_NAMES, weights.tolist())),
        "counts_a": str(args.counts_a),
        "counts_b": str(args.counts_b),
        "labels_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
