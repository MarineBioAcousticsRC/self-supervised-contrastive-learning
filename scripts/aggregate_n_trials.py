#!/usr/bin/env python3
"""Aggregate per-trial elbow metrics into mean ± sample std tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

METRIC_KEYS = [
    "selected_k",
    "ari",
    "ami",
    "nmi",
    "ri",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "inertia",
    "n_points",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mean±std over N-trial elbow metrics")
    p.add_argument("--out-root", required=True, help="Root written by run_n_trials.py")
    p.add_argument(
        "--aggregate-dir",
        default=None,
        help="Where to write tables (default: OUT_ROOT/aggregate)",
    )
    return p.parse_args()


def _stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
        "values": [float(x) for x in arr],
    }


def load_trials(out_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(out_root.glob("*/trial_*/trial_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["trial_dir"] = str(path.parent)
        rows.append(data)
    if not rows:
        raise SystemExit(f"No trial_metrics.json files under {out_root}")
    return pd.DataFrame(rows)


def format_summary(per_model: Dict[str, Dict[str, Any]], out_root: Path) -> str:
    lines = [
        "N-TRIAL ELBOW PIPELINE — AGGREGATE SUMMARY",
        f"Out root: {out_root}",
        "",
        "Metrics are MiniBatch k-means scores at the per-trial inertia elbow.",
        "std is the sample standard deviation (ddof=1).",
        "",
    ]
    for model, agg in per_model.items():
        lines.append("=" * 72)
        lines.append(model)
        lines.append("=" * 72)
        n_trials = None
        for key in METRIC_KEYS:
            if key not in agg:
                continue
            s = agg[key]
            n_trials = s["n"]
            lines.append(
                f"  {key:22s}  mean={s['mean']:.6f}  std={s['std']:.6f}  "
                f"min={s['min']:.6f}  max={s['max']:.6f}  n={s['n']}"
            )
        lines.append(f"  {'n_trials':22s}  {n_trials}")
        lines.append("")
    return "\n".join(lines)


def summary_table(per_model: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, agg in per_model.items():
        row: Dict[str, Any] = {"model": model}
        for key in METRIC_KEYS:
            if key not in agg:
                continue
            s = agg[key]
            row[f"{key}_mean"] = s["mean"]
            row[f"{key}_std"] = s["std"]
            row[f"{key}_min"] = s["min"]
            row[f"{key}_max"] = s["max"]
            row["n"] = s["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    agg_dir = Path(args.aggregate_dir).expanduser() if args.aggregate_dir else out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    df = load_trials(out_root)
    per_trial_path = agg_dir / "per_trial.csv"
    keep_cols = [
        c
        for c in [
            "model",
            "trial_idx",
            "seed",
            "selector",
            *METRIC_KEYS,
            "n_gt_classes",
            "emb_dir",
            "checkpoint",
            "trial_dir",
        ]
        if c in df.columns
    ]
    df[keep_cols].sort_values(["model", "trial_idx"]).to_csv(per_trial_path, index=False)

    per_model: Dict[str, Dict[str, Any]] = {}
    for model, g in df.groupby("model", sort=True):
        agg: Dict[str, Any] = {}
        for key in METRIC_KEYS:
            if key not in g.columns:
                continue
            vals = [float(v) for v in g[key].tolist() if v is not None and pd.notna(v)]
            if vals:
                agg[key] = _stats(vals)
        per_model[str(model)] = agg

    summary_df = summary_table(per_model)
    summary_csv = agg_dir / "summary_mean_std.csv"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "out_root": str(out_root),
        "n_rows": int(len(df)),
        "models": per_model,
    }
    summary_json = agg_dir / "aggregate.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    text = format_summary(per_model, out_root)
    summary_txt = agg_dir / "summary.txt"
    summary_txt.write_text(text, encoding="utf-8")

    print(text)
    print(f"Wrote {per_trial_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_txt}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
