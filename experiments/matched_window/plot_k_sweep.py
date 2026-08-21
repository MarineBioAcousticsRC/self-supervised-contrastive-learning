#!/usr/bin/env python3
"""Line plots of matched-window KMeans metrics vs k.

Three figures (1s / 3s / 5s). Each shows NMI, ARI, silhouette, and purity vs k.
3s overlays BirdNET; 5s overlays Perch. Titles include n_points clustered.

Example:
  python -m experiments.matched_window.plot_k_sweep \\
    --metrics /path/to/clustering_metrics.csv \\
    --out-dir ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Matched-window evaluate.py metrics (no AMI/RI in this run).
METRICS: List[Tuple[str, str]] = [
    ("nmi", "NMI"),
    ("ari", "ARI"),
    ("silhouette", "Silhouette"),
    ("purity", "Purity"),
]

PRIMARY_LABEL = "self-supervised contrastive learning"

# (title_window, primary_repr, optional_baseline_repr, baseline_label)
PANELS: List[Tuple[str, str, Optional[str], Optional[str]]] = [
    ("1s (self-supervised contrastive learning)", "a2v_1s", None, None),
    ("3s matched window", "a2v_3s", "birdnet_3s", "BirdNET"),
    ("5s matched window", "a2v_5s", "perch_5s", "Perch"),
]

METRIC_COLORS = {
    "nmi": "#1b9e77",
    "ari": "#d95f02",
    "silhouette": "#7570b3",
    "purity": "#e7298a",
}


def _load(metrics_path: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    need = {"representation", "subset", "k", "n_points"} | {m for m, _ in METRICS}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {metrics_path}: {sorted(missing)}")
    out = df[df["subset"].astype(str) == subset].copy()
    if out.empty:
        raise SystemExit(f"No rows with subset={subset!r} in {metrics_path}")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    for col, _ in METRICS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["n_points"] = pd.to_numeric(out["n_points"], errors="coerce")
    return out.sort_values(["representation", "k"])


def _n_points(df: pd.DataFrame, repr_name: str) -> int:
    sub = df[df["representation"] == repr_name]
    if sub.empty:
        return 0
    return int(sub["n_points"].iloc[0])


def _plot_panel(
    df: pd.DataFrame,
    *,
    window_label: str,
    primary: str,
    baseline: Optional[str],
    baseline_label: Optional[str],
    out_path: Path,
) -> None:
    prim = df[df["representation"] == primary]
    if prim.empty:
        raise SystemExit(f"No rows for representation={primary!r}")

    n_prim = _n_points(df, primary)
    title_bits = [window_label, f"n={n_prim} ({PRIMARY_LABEL})"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for col, nice in METRICS:
        ax.plot(
            prim["k"],
            prim[col],
            color=METRIC_COLORS[col],
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{PRIMARY_LABEL} · {nice}",
        )

    if baseline:
        base = df[df["representation"] == baseline]
        if base.empty:
            raise SystemExit(f"No rows for representation={baseline!r}")
        n_base = _n_points(df, baseline)
        bl = baseline_label or baseline
        title_bits.append(f"n={n_base} ({bl})")
        for col, nice in METRICS:
            ax.plot(
                base["k"],
                base[col],
                color=METRIC_COLORS[col],
                linestyle="--",
                linewidth=1.6,
                marker="s",
                markersize=3.5,
                alpha=0.85,
                label=f"{bl} · {nice}",
            )

    ax.set_xlabel("k (MiniBatchKMeans)")
    ax.set_ylabel("Score")
    ax.set_title(" · ".join(title_bits))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_all(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    formats: Sequence[str] = ("png",),
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    name_by_primary = {
        "a2v_1s": "k_sweep_a2v_1s",
        "a2v_3s": "k_sweep_matched_3s",
        "a2v_5s": "k_sweep_matched_5s",
    }
    for window_label, primary, baseline, baseline_label in PANELS:
        stem = name_by_primary[primary]
        for fmt in formats:
            path = out_dir / f"{stem}.{fmt}"
            _plot_panel(
                df,
                window_label=window_label,
                primary=primary,
                baseline=baseline,
                baseline_label=baseline_label,
                out_path=path,
            )
            written[f"{stem}.{fmt}"] = path
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to clustering_metrics.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for PNG/PDF figures",
    )
    p.add_argument("--subset", default="single", help="Metrics subset (default: single)")
    p.add_argument(
        "--formats",
        default="png",
        help="Comma-separated formats (default: png)",
    )
    args = p.parse_args()
    formats = [f.strip().lstrip(".") for f in str(args.formats).split(",") if f.strip()]
    df = _load(Path(args.metrics), subset=str(args.subset))
    plot_all(df, Path(args.out_dir), formats=formats)


if __name__ == "__main__":
    main()
