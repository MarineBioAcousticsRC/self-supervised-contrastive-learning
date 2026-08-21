#!/usr/bin/env python3
"""Line plots for matched-window *voting* k-sweep metrics.

Three figures (1s / 3s / 5s). Metrics: NMI, ARI, AMI, purity vs k.
3s overlays BirdNET; 5s overlays Perch. Legend linestyles match series.

  python -m experiments.matched_window.plot_k_sweep_vote \\
    --metrics /path/to/results/vote/clustering_metrics.csv \\
    --out-dir ./figures/vote
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiments.matched_window.config import KMEANS_BATCH_SIZE, UMAP_N_COMPONENTS

PRIMARY_LABEL = "self-supervised contrastive learning (vote)"

METRICS: List[Tuple[str, str]] = [
    ("nmi", "NMI"),
    ("ari", "ARI"),
    ("ami", "AMI"),
    ("purity", "Purity"),
]

# (title_window, primary_repr, optional_baseline_repr, baseline_label)
PANELS: List[Tuple[str, str, Optional[str], Optional[str]]] = [
    ("1s (self-supervised contrastive, vote)", "a2v_1s_vote", None, None),
    ("3s matched window (vote)", "a2v_3s_vote", "birdnet_3s", "BirdNET"),
    ("5s matched window (vote)", "a2v_5s_vote", "perch_5s", "Perch"),
]

METRIC_COLORS = {
    "nmi": "#1b9e77",
    "ari": "#d95f02",
    "ami": "#7570b3",
    "purity": "#e7298a",
}


def _load(metrics_path: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    need = {"representation", "subset", "k"} | {m for m, _ in METRICS}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {metrics_path}: {sorted(missing)}")
    out = df[df["subset"].astype(str) == subset].copy()
    if out.empty:
        raise SystemExit(f"No rows with subset={subset!r} in {metrics_path}")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    for col, _ in METRICS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "n_windows" in out.columns:
        out["n_windows"] = pd.to_numeric(out["n_windows"], errors="coerce")
    elif "n_points" in out.columns:
        out["n_windows"] = pd.to_numeric(out["n_points"], errors="coerce")
    return out.sort_values(["representation", "k"])


def _n_windows(df: pd.DataFrame, repr_name: str) -> int:
    sub = df[df["representation"] == repr_name]
    if sub.empty:
        return 0
    if "n_windows" in sub.columns and pd.notna(sub["n_windows"].iloc[0]):
        return int(sub["n_windows"].iloc[0])
    return int(sub["n_points"].iloc[0]) if "n_points" in sub.columns else 0


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

    n_prim = _n_windows(df, primary)
    title_bits = [window_label, f"n={n_prim} windows"]
    space = "umap"
    if "cluster_space" in prim.columns and pd.notna(prim["cluster_space"].iloc[0]):
        space = str(prim["cluster_space"].iloc[0])
    if space == "native":
        cdim = ""
        if "cluster_dim" in prim.columns and pd.notna(prim["cluster_dim"].iloc[0]):
            cdim = f" dim={int(prim['cluster_dim'].iloc[0])}"
        space_txt = f"native{cdim} (no UMAP)"
    else:
        udim = UMAP_N_COMPONENTS
        if "umap_dim" in prim.columns and pd.notna(prim["umap_dim"].iloc[0]):
            udim = int(prim["umap_dim"].iloc[0])
        space_txt = f"UMAP dim={udim}"
    subtitle = (
        f"Voting scheme (frame cluster → majority label per window) · "
        f"{space_txt} · MiniBatchKMeans batch_size={KMEANS_BATCH_SIZE}"
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

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
        n_base = _n_windows(df, baseline)
        bl = baseline_label or baseline
        title_bits.append(f"n={n_base} ({bl})")
        for col, nice in METRICS:
            ax.plot(
                base["k"],
                base[col],
                color=METRIC_COLORS[col],
                linestyle="--",
                linewidth=1.8,
                marker="s",
                markersize=3.5,
                alpha=0.9,
                label=f"{bl} · {nice}",
            )

    ax.set_xlabel("k (MiniBatchKMeans)")
    ax.set_ylabel("Extrinsic score")
    ax.set_title(" · ".join(title_bits) + f"\n{subtitle}", fontsize=10)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    # Keep dashed/solid styles visible in the legend.
    leg = ax.legend(loc="best", fontsize=7.5, ncol=2, frameon=True)
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", [])
    for handle in handles:
        ls = handle.get_linestyle()
        # Older matplotlib may normalize "--" oddly; force readable dash
        if ls not in ("-", "solid", None):
            handle.set_linestyle("--")
            handle.set_dashes((4, 2))
        else:
            handle.set_linestyle("-")
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
        "a2v_1s_vote": "k_sweep_vote_a2v_1s",
        "a2v_3s_vote": "k_sweep_vote_matched_3s",
        "a2v_5s_vote": "k_sweep_vote_matched_5s",
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
        help="Path to results/vote/clustering_metrics.csv",
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
