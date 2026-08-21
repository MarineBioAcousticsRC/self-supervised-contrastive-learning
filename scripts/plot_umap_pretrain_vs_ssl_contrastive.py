#!/usr/bin/env python3
"""
UMAP pairs: self-supervised pretraining vs self-supervised contrastive learning.

Example:

  python scripts/plot_umap_pretrain_vs_ssl_contrastive.py \\
    --pretrain-dir /path/to/embeddings/self_supervised_pretrain \\
    --contrastive-dir /path/to/embeddings/self_supervised_contrastive \\
    --wav-csv-dir /path/to/nips4bplus \\
    --out-dir /path/to/umap_out \\
    --max-points 8000 --ignore-test-files
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from frame_level_clustering import (  # noqa: E402
    filter_by_min_label_count,
    load_h5_embeddings_for_clustering,
    maybe_add_labels_from_csv,
    subsample_max_per_label,
    wav_stem_from_h5_name,
)

MARKERS: Sequence[str] = (
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "*",
    "h",
    "<",
    ">",
    "p",
    "8",
    "H",
    "d",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pretrain-dir",
        required=True,
        help="Frame-level .h5 embeddings for pretrained A2V",
    )
    p.add_argument(
        "--contrastive-dir",
        required=True,
        help="Frame-level .h5 embeddings for pretrained + vocal contrastive",
    )
    p.add_argument(
        "--wav-csv-dir",
        required=True,
        help="NIPS4Bplus WAV/CSV annotation directory",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory for PNG/HTML figures + sidecar metadata",
    )
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--max-per-label", type=int, default=400)
    p.add_argument("--min-label-count", type=int, default=20)
    p.add_argument("--ignore-test-files", action="store_true", default=True)
    p.add_argument("--include-test-files", action="store_true", default=False)
    p.add_argument("--only-call", action="store_true", default=False)
    p.add_argument("--segment-h5-use-recording-csv", action="store_true", default=True)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--umap-metric", default="cosine")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--point-size", type=float, default=12.0)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--no-plotly",
        action="store_true",
        help="Skip interactive 3D HTML (matplotlib PNG only)",
    )
    return p.parse_args()


def _frame_key(df: pd.DataFrame) -> pd.Series:
    stems = df["h5_name"].map(wav_stem_from_h5_name).astype(str)
    # Round time to 1 ms so tiny float drift does not break alignment.
    t = np.round(df["time_s"].to_numpy(dtype=np.float64), 3)
    return stems + "|" + pd.Series(t.astype(str), index=df.index)


def _prepare_labeled(
    emb_dir: str,
    *,
    wav_csv_dir: str,
    ignore_test: bool,
    only_call: bool,
    segment_h5_use_recording_csv: bool,
) -> pd.DataFrame:
    print(f"Loading embeddings: {emb_dir}")
    df = load_h5_embeddings_for_clustering(emb_dir)
    print(f"  raw frames: {len(df)}")
    df = maybe_add_labels_from_csv(
        df,
        wav_csv_dir=wav_csv_dir,
        segment_h5_use_recording_csv=segment_h5_use_recording_csv,
        ignore_test_files=ignore_test,
        only_call=only_call,
        max_segment_duration=None,
    )
    df = df[df["class_name"].astype(str).str.lower() != "unknown"].copy()
    print(f"  labeled (non-Unknown): {len(df)}")
    df["align_key"] = _frame_key(df)
    return df


def _align_and_subsample(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    max_points: int,
    max_per_label: int,
    min_label_count: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # One row per key (first occurrence) then intersect.
    a = df_a.drop_duplicates("align_key", keep="first").set_index("align_key", drop=False)
    b = df_b.drop_duplicates("align_key", keep="first").set_index("align_key", drop=False)
    common = a.index.intersection(b.index)
    print(f"Aligned common frames: {len(common)}")
    if len(common) < 50:
        raise RuntimeError(
            f"Too few aligned frames ({len(common)}). "
            "Check that both dirs cover the same NIPS files."
        )

    a = a.loc[common].copy()
    b = b.loc[common].copy()
    # Prefer pretrain labels; they should match if CSVs are shared.
    labels = a["class_name"].astype(str)
    b["class_name"] = labels.to_numpy()

    joined = a[["align_key", "class_name"]].copy()
    joined = filter_by_min_label_count(joined, min_label_count)
    keep_keys = set(joined["align_key"].tolist())
    a = a[a["align_key"].isin(keep_keys)].copy()
    b = b[b["align_key"].isin(keep_keys)].copy()
    print(f"After min_label_count={min_label_count}: {len(a)} frames, "
          f"{a['class_name'].nunique()} species")

    # Subsample on the shared label table, then apply the same keys to both.
    meta = a[["align_key", "class_name"]].reset_index(drop=True)
    meta = subsample_max_per_label(meta, max_per_label, random_state)
    if len(meta) > int(max_points):
        rng = np.random.default_rng(int(random_state))
        take = rng.choice(len(meta), size=int(max_points), replace=False)
        meta = meta.iloc[np.sort(take)].copy()
    keys = meta["align_key"].tolist()
    a = a.set_index("align_key").loc[keys].reset_index()
    b = b.set_index("align_key").loc[keys].reset_index()
    a["class_name"] = meta["class_name"].to_numpy()
    b["class_name"] = meta["class_name"].to_numpy()
    print(f"Final subsample: {len(a)} frames, {a['class_name'].nunique()} species")
    return a.reset_index(drop=True), b.reset_index(drop=True)


def _stack_X(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in df["embedding_vec"].to_numpy()])


def _fit_umap(X: np.ndarray, n_components: int, args: argparse.Namespace) -> np.ndarray:
    import umap

    n = int(X.shape[0])
    nn = int(min(args.umap_n_neighbors, max(2, n - 1)))
    n_comp = int(min(n_components, max(2, n - 2)))
    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=nn,
        min_dist=float(args.umap_min_dist),
        metric=str(args.umap_metric),
        random_state=int(args.random_state),
        init="random",
    )
    Z = reducer.fit_transform(X)
    return np.asarray(Z, dtype=np.float32)


def _species_style(species: Sequence[str]) -> Dict[str, Tuple[Tuple[float, ...], str]]:
    uniq = sorted({str(s) for s in species if str(s)})
    cmap = plt.get_cmap("tab20")
    styles: Dict[str, Tuple[Tuple[float, ...], str]] = {}
    for i, s in enumerate(uniq):
        color = cmap(i % 20)
        marker = MARKERS[i % len(MARKERS)]
        styles[s] = (color, marker)
    return styles


def _scatter_2d(ax, Z: np.ndarray, labels: np.ndarray, styles, *, s: float, alpha: float) -> None:
    for sp, (color, marker) in styles.items():
        m = labels == sp
        if not np.any(m):
            continue
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[color],
            marker=marker,
            s=s,
            alpha=alpha,
            linewidths=0.2,
            edgecolors="k",
            label=sp,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)


def _scatter_3d(ax, Z: np.ndarray, labels: np.ndarray, styles, *, s: float, alpha: float) -> None:
    for sp, (color, marker) in styles.items():
        m = labels == sp
        if not np.any(m):
            continue
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            Z[m, 2],
            c=[color],
            marker=marker,
            s=s,
            alpha=alpha,
            linewidths=0.2,
            edgecolors="k",
            depthshade=True,
            label=sp,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")


def _legend_handles(styles: Dict[str, Tuple[Tuple[float, ...], str]]) -> List[Line2D]:
    handles = []
    for sp, (color, marker) in styles.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor="k",
                markeredgewidth=0.3,
                markersize=7,
                label=sp,
                linestyle="None",
            )
        )
    return handles


def _save_legend(styles, out_path: Path, dpi: int) -> None:
    handles = _legend_handles(styles)
    n = len(handles)
    ncols = 3 if n > 24 else (2 if n > 12 else 1)
    fig_h = max(2.5, 0.28 * ((n + ncols - 1) // ncols))
    fig = plt.figure(figsize=(10, fig_h))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=ncols,
        frameon=True,
        fontsize=7,
        title="Species (color + marker)",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_2d_pair(
    Z_pre: np.ndarray,
    Z_con: np.ndarray,
    labels: np.ndarray,
    styles,
    out_path: Path,
    *,
    s: float,
    alpha: float,
    dpi: int,
    n: int,
    n_sp: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    _scatter_2d(axes[0], Z_pre, labels, styles, s=s, alpha=alpha)
    axes[0].set_title("Pretrained")
    _scatter_2d(axes[1], Z_con, labels, styles, s=s, alpha=alpha)
    axes[1].set_title("Pretrained + vocal contrastive SSL")
    fig.suptitle(
        f"Frame-level UMAP (2D) — species-coded  |  n={n}, species={n_sp}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_3d_pair(
    Z_pre: np.ndarray,
    Z_con: np.ndarray,
    labels: np.ndarray,
    styles,
    out_path: Path,
    *,
    s: float,
    alpha: float,
    dpi: int,
    n: int,
    n_sp: int,
) -> None:
    fig = plt.figure(figsize=(14, 6.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    _scatter_3d(ax0, Z_pre, labels, styles, s=s, alpha=alpha)
    ax0.set_title("Pretrained")
    _scatter_3d(ax1, Z_con, labels, styles, s=s, alpha=alpha)
    ax1.set_title("Pretrained + vocal contrastive SSL")
    fig.suptitle(
        f"Frame-level UMAP (3D) — species-coded  |  n={n}, species={n_sp}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plotly_3d_pair(
    Z_pre: np.ndarray,
    Z_con: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skip interactive HTML")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("Pretrained", "Pretrained + vocal contrastive SSL"),
    )
    uniq = sorted(set(labels.tolist()))
    for sp in uniq:
        m = labels == sp
        for col, Z in ((1, Z_pre), (2, Z_con)):
            fig.add_trace(
                go.Scatter3d(
                    x=Z[m, 0],
                    y=Z[m, 1],
                    z=Z[m, 2],
                    mode="markers",
                    name=sp,
                    legendgroup=sp,
                    showlegend=(col == 1),
                    marker=dict(size=3, opacity=0.75),
                    hovertemplate=f"{sp}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        title="Frame-level UMAP (3D) — species-coded",
        height=700,
        legend=dict(font=dict(size=9)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main() -> None:
    args = _parse_args()
    ignore_test = bool(args.ignore_test_files) and not bool(args.include_test_files)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pre = _prepare_labeled(
        args.pretrain_dir,
        wav_csv_dir=args.wav_csv_dir,
        ignore_test=ignore_test,
        only_call=bool(args.only_call),
        segment_h5_use_recording_csv=bool(args.segment_h5_use_recording_csv),
    )
    df_con = _prepare_labeled(
        args.contrastive_dir,
        wav_csv_dir=args.wav_csv_dir,
        ignore_test=ignore_test,
        only_call=bool(args.only_call),
        segment_h5_use_recording_csv=bool(args.segment_h5_use_recording_csv),
    )
    df_pre, df_con = _align_and_subsample(
        df_pre,
        df_con,
        max_points=int(args.max_points),
        max_per_label=int(args.max_per_label),
        min_label_count=int(args.min_label_count),
        random_state=int(args.random_state),
    )

    labels = df_pre["class_name"].astype(str).to_numpy()
    styles = _species_style(labels)
    X_pre = _stack_X(df_pre)
    X_con = _stack_X(df_con)
    print(f"X_pre {X_pre.shape}, X_con {X_con.shape}")

    print("Fitting UMAP 2D (pretrained)…")
    Z2_pre = _fit_umap(X_pre, 2, args)
    print("Fitting UMAP 2D (contrastive)…")
    Z2_con = _fit_umap(X_con, 2, args)
    print("Fitting UMAP 3D (pretrained)…")
    Z3_pre = _fit_umap(X_pre, 3, args)
    print("Fitting UMAP 3D (contrastive)…")
    Z3_con = _fit_umap(X_con, 3, args)

    n = len(labels)
    n_sp = int(len(styles))
    _plot_2d_pair(
        Z2_pre,
        Z2_con,
        labels,
        styles,
        out_dir / "umap_2d_pretrain_vs_contrastive.png",
        s=float(args.point_size),
        alpha=float(args.alpha),
        dpi=int(args.dpi),
        n=n,
        n_sp=n_sp,
    )
    _plot_3d_pair(
        Z3_pre,
        Z3_con,
        labels,
        styles,
        out_dir / "umap_3d_pretrain_vs_contrastive.png",
        s=float(args.point_size),
        alpha=float(args.alpha),
        dpi=int(args.dpi),
        n=n,
        n_sp=n_sp,
    )
    _save_legend(styles, out_dir / "umap_species_legend.png", dpi=int(args.dpi))

    np.savez_compressed(
        out_dir / "umap_coords.npz",
        Z2_pre=Z2_pre,
        Z2_con=Z2_con,
        Z3_pre=Z3_pre,
        Z3_con=Z3_con,
        labels=labels,
        align_key=df_pre["align_key"].astype(str).to_numpy(),
    )
    meta = pd.DataFrame(
        {
            "align_key": df_pre["align_key"].astype(str),
            "class_name": labels,
            "wav": df_pre["wav"].astype(str),
            "time_s": df_pre["time_s"].astype(float),
        }
    )
    meta.to_csv(out_dir / "umap_sample_meta.csv", index=False)

    if not args.no_plotly:
        _plotly_3d_pair(
            Z3_pre,
            Z3_con,
            labels,
            out_dir / "umap_3d_pretrain_vs_contrastive.html",
        )

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
