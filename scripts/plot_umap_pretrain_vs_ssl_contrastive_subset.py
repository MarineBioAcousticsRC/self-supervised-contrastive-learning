#!/usr/bin/env python3
"""
Subset UMAP pair: self-supervised pretraining vs self-supervised contrastive learning.

Same loading/alignment as plot_umap_pretrain_vs_ssl_contrastive.py.

Example:

  python scripts/plot_umap_pretrain_vs_ssl_contrastive_subset.py \\
    --pretrain-dir /path/to/embeddings/self_supervised_pretrain \\
    --contrastive-dir /path/to/embeddings/self_supervised_contrastive \\
    --wav-csv-dir /path/to/nips4bplus \\
    --out-dir /path/to/umap_out \\
    --n-classes 8 --ignore-test-files
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

from frame_level_clustering import subsample_max_per_label  # noqa: E402
from plot_umap_pretrain_vs_ssl_contrastive import (  # noqa: E402
    MARKERS,
    _fit_umap,
    _prepare_labeled,
    _stack_X,
)

_TYPE_SUFFIXES = ("_song", "_call", "_drum")
TITLE_PRE = "Self-supervised pretraining"
TITLE_CON = "Self-supervised contrastive learning"
PAPER_FIGSIZE_2D = (7.16, 2.95)
PAPER_FIGSIZE_3D = (7.16, 3.20)


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
        }
    )


def _panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
    kwargs = dict(
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="bottom",
        ha="left",
    )
    text = getattr(ax, "text2D", ax.text)
    text(x, y, f"({letter})", **kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-dir", required=True)
    p.add_argument("--contrastive-dir", required=True)
    p.add_argument("--wav-csv-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--n-classes",
        type=int,
        default=8,
        help="Number of classes to keep (0 = all). Default 8.",
    )
    p.add_argument(
        "--class-selection",
        choices=["frequency", "random"],
        default="frequency",
        help="frequency = most aligned frames; random = uniform draw among eligible classes.",
    )
    p.add_argument(
        "--vocal-type",
        choices=["all", "call", "song"],
        default="all",
        help="Restrict to NIPS class names ending in _call or _song.",
    )
    p.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated class names. Overrides --n-classes.",
    )
    p.add_argument(
        "--exclude-classes",
        default="Human",
        help="Comma-separated class names to drop (default: Human).",
    )
    p.add_argument(
        "--one-type-per-species",
        action="store_true",
        default=True,
        help="If both song and call of a species rank high, keep the more frequent type.",
    )
    p.add_argument(
        "--allow-multiple-types-per-species",
        action="store_true",
        default=False,
        help="Disable one-type-per-species (allow Parmaj_song and Parmaj_call together).",
    )
    p.add_argument("--max-per-label", type=int, default=300)
    p.add_argument("--min-label-count", type=int, default=50)
    p.add_argument("--ignore-test-files", action="store_true", default=True)
    p.add_argument("--include-test-files", action="store_true", default=False)
    p.add_argument("--only-call", action="store_true", default=False)
    p.add_argument("--segment-h5-use-recording-csv", action="store_true", default=True)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--umap-metric", default="cosine")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--point-size", type=float, default=12.0)
    p.add_argument("--alpha", type=float, default=0.82)
    p.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster DPI for PNG and for rasterized points in PDF (paper requires ≥600).",
    )
    p.add_argument(
        "--species-list",
        default="",
        help="NIPS espece CSV with columns 'class name' and 'Scientific_name'.",
    )
    p.add_argument(
        "--legend-names",
        choices=["pretty", "scientific"],
        default="pretty",
        help="Legend: NIPS code (pretty) or italic scientific name.",
    )
    p.add_argument(
        "--n-random-draws",
        type=int,
        default=1,
        help="With --class-selection random, write this many independent random subsets.",
    )
    p.add_argument("--no-plotly", action="store_true")
    return p.parse_args()


def _species_code(name: str) -> str:
    s = str(name)
    for suf in _TYPE_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _pretty_label(name: str) -> str:
    s = str(name)
    for suf in _TYPE_SUFFIXES:
        if s.endswith(suf):
            return f"{s[: -len(suf)]} ({suf[1:]})"
    return s


def _vocal_type_of(name: str) -> str:
    s = str(name)
    for suf in _TYPE_SUFFIXES:
        if s.endswith(suf):
            return suf[1:]
    return ""


def _normalize_scientific(name: str) -> str:
    s = " ".join(str(name).strip().split())
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def _italic_scientific(name: str) -> str:
    parts = [p.replace("_", r"\_") for p in _normalize_scientific(name).split() if p]
    joined = r"\ ".join(parts)
    return rf"$\mathit{{{joined}}}$"


def _resolve_species_list(explicit: str) -> Path:
    candidates = []
    if str(explicit).strip():
        candidates.append(Path(str(explicit).strip()).expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "data" / "nips4b_birdchallenge_espece_list.csv",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "NIPS species list CSV not found. Pass --species-list to "
        "nips4b_birdchallenge_espece_list.csv"
    )


def _load_scientific_names(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    df = pd.read_csv(path)
    cols = {str(c).strip(): c for c in df.columns}
    class_col = cols.get("class name") or cols.get("class_name")
    sci_col = cols.get("Scientific_name") or cols.get("scientific_name")
    eng_col = cols.get("English_name") or cols.get("english_name")
    if class_col is None or sci_col is None:
        raise RuntimeError(
            f"{path} needs 'class name' and 'Scientific_name' columns; got {list(df.columns)}"
        )
    mapping: Dict[str, str] = {}
    english: Dict[str, str] = {}
    skip = {"", "nan", "noise sample", "empty"}
    for _, row in df.iterrows():
        cls = str(row[class_col]).strip()
        sci = _normalize_scientific(row[sci_col])
        if not cls or sci.lower() in skip:
            continue
        mapping[cls] = sci
        if eng_col is not None:
            eng = str(row[eng_col]).strip()
            if eng and eng.lower() not in skip:
                english[cls] = eng
    print(f"Loaded {len(mapping)} scientific names from {path}")
    return mapping, english


def _scientific_label(class_name: str, sci_map: Dict[str, str]) -> str:
    sci = sci_map.get(str(class_name))
    if not sci:
        return _pretty_label(class_name)
    vtype = _vocal_type_of(class_name)
    core = _italic_scientific(sci)
    return f"{core} ({vtype})" if vtype else core


def _align(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    labels = a["class_name"].astype(str)
    b["class_name"] = labels.to_numpy()
    return a.reset_index(drop=True), b.reset_index(drop=True)


def _select_classes(
    labels: pd.Series,
    *,
    n_classes: int,
    explicit: Sequence[str],
    exclude: Sequence[str],
    one_type_per_species: bool,
    min_label_count: int,
    class_selection: str,
    vocal_type: str,
    random_state: int,
) -> List[str]:
    counts = labels.astype(str).value_counts()
    exclude_set = {str(x).strip() for x in exclude if str(x).strip()}
    counts = counts[~counts.index.isin(exclude_set)]
    if vocal_type in ("call", "song"):
        suf = f"_{vocal_type}"
        counts = counts[counts.index.astype(str).str.endswith(suf)]
    counts = counts[counts >= int(min_label_count)]
    if counts.empty:
        raise RuntimeError(
            f"No classes left after vocal_type={vocal_type}, "
            f"min_label_count={min_label_count}, exclude={sorted(exclude_set)}"
        )

    if explicit:
        wanted = [str(x).strip() for x in explicit if str(x).strip()]
        missing = [c for c in wanted if c not in counts.index]
        if missing:
            raise RuntimeError(f"Requested classes not in aligned data: {missing}")
        print("Using explicit classes:", wanted)
        return wanted

    pool = list(counts.index)
    print(
        f"Eligible classes ({len(pool)}) after vocal_type={vocal_type}, "
        f"min_label_count={min_label_count}:"
    )
    print(counts.to_string())

    if class_selection == "random":
        n = int(n_classes) if n_classes > 0 else len(pool)
        n = min(n, len(pool))
        rng = np.random.default_rng(int(random_state))
        pool_sorted = sorted(pool)
        take = rng.choice(len(pool_sorted), size=n, replace=False)
        selected = [pool_sorted[int(i)] for i in take]
        print(f"Randomly selected {len(selected)} classes (seed={random_state}):", selected)
        return selected

    ranked = list(counts.index)
    if one_type_per_species:
        chosen: List[str] = []
        seen_sp = set()
        for cls in ranked:
            sp = _species_code(cls)
            if sp in seen_sp:
                continue
            seen_sp.add(sp)
            chosen.append(cls)
            if n_classes > 0 and len(chosen) >= int(n_classes):
                break
        selected = chosen
    else:
        selected = ranked[: int(n_classes)] if n_classes > 0 else ranked

    print(f"Selected {len(selected)} classes:", selected)
    return selected


def _apply_subset(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    classes: Sequence[str],
    *,
    max_per_label: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep = set(classes)
    a = df_a[df_a["class_name"].astype(str).isin(keep)].copy()
    b = df_b[df_b["class_name"].astype(str).isin(keep)].copy()
    meta = a[["align_key", "class_name"]].reset_index(drop=True)
    meta = subsample_max_per_label(meta, max_per_label, random_state)
    keys = meta["align_key"].tolist()
    a = a.set_index("align_key").loc[keys].reset_index()
    b = b.set_index("align_key").loc[keys].reset_index()
    a["class_name"] = meta["class_name"].to_numpy()
    b["class_name"] = meta["class_name"].to_numpy()
    print(
        f"Subset subsample: {len(a)} frames, {a['class_name'].nunique()} classes, "
        f"max_per_label={max_per_label}"
    )
    print(a["class_name"].value_counts().to_string())
    return a.reset_index(drop=True), b.reset_index(drop=True)


def _subset_styles(species: Sequence[str]) -> Dict[str, Tuple[Tuple[float, ...], str]]:
    uniq = sorted({str(s) for s in species if str(s)})
    cmap = plt.get_cmap("tab10")
    styles: Dict[str, Tuple[Tuple[float, ...], str]] = {}
    for i, s in enumerate(uniq):
        styles[s] = (cmap(i % 10), MARKERS[i % len(MARKERS)])
    return styles


def _legend_handles(
    styles: Dict[str, Tuple[Tuple[float, ...], str]],
    label_fn,
) -> List[Line2D]:
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
                markeredgewidth=0.25,
                markersize=6.5,
                label=label_fn(sp),
                linestyle="None",
            )
        )
    return handles


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _scatter_2d_paper(
    ax, Z: np.ndarray, labels: np.ndarray, styles, *, s: float, alpha: float
) -> None:
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
            linewidths=0.15,
            edgecolors="k",
            rasterized=True,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def _scatter_3d_paper(
    ax, Z: np.ndarray, labels: np.ndarray, styles, *, s: float, alpha: float
) -> None:
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
            linewidths=0.15,
            edgecolors="k",
            depthshade=False,
            rasterized=True,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.tick_params(labelsize=7)
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.zaxis.labelpad = 0


def _verify_png_dpi(path: Path, min_dpi: int, figsize_in: Tuple[float, float]) -> None:
    try:
        from PIL import Image
    except ImportError:
        print(f"  saved {path.name} (install Pillow to print PNG DPI metadata)")
        return
    with Image.open(path) as im:
        w, h = im.size
        raw = im.info.get("dpi", (min_dpi, min_dpi))
        if isinstance(raw, tuple):
            xdpi = float(raw[0])
        else:
            xdpi = float(raw)
    print(
        f"  {path.name}: {w}x{h} px, metadata DPI={xdpi:.1f} "
        f"(target {min_dpi}; ~{figsize_in[0]:.2f} in wide)"
    )
    if xdpi + 0.5 < float(min_dpi):
        raise RuntimeError(f"{path} DPI {xdpi} is below required {min_dpi}")


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int, extra=None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep dpi on PDF too: scatter points are rasterized, so this is the image DPI.
    kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    if extra:
        kw["bbox_extra_artists"] = extra
    fig.savefig(out_path, **kw)
    fig.savefig(out_path.with_suffix(".pdf"), **kw)
    _verify_png_dpi(out_path, dpi, tuple(fig.get_size_inches()))


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
    label_fn,
    legend_ncol: int,
) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=PAPER_FIGSIZE_2D)
    _scatter_2d_paper(axes[0], Z_pre, labels, styles, s=s, alpha=alpha)
    _scatter_2d_paper(axes[1], Z_con, labels, styles, s=s, alpha=alpha)
    axes[0].set_title(f"(a)  {TITLE_PRE}", loc="left", pad=4)
    axes[1].set_title(f"(b)  {TITLE_CON}", loc="left", pad=4)
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout(w_pad=1.6, pad=0.25)
    fontsize = 6.5 if legend_ncol <= 2 else 7.5
    leg = fig.legend(
        handles=_legend_handles(styles, label_fn),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=int(legend_ncol),
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.15,
        labelspacing=0.35,
    )
    _save_fig(fig, out_path, dpi, extra=[leg])
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
    label_fn,
    legend_ncol: int,
) -> None:
    _paper_style()
    fig = plt.figure(figsize=PAPER_FIGSIZE_3D)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    _scatter_3d_paper(ax0, Z_pre, labels, styles, s=s, alpha=alpha)
    _scatter_3d_paper(ax1, Z_con, labels, styles, s=s, alpha=alpha)
    ax0.set_title(f"(a)  {TITLE_PRE}", loc="left", pad=6)
    ax1.set_title(f"(b)  {TITLE_CON}", loc="left", pad=6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.02, wspace=-0.05)
    fontsize = 6.5 if legend_ncol <= 2 else 7.5
    leg = fig.legend(
        handles=_legend_handles(styles, label_fn),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=int(legend_ncol),
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.15,
        labelspacing=0.35,
    )
    _save_fig(fig, out_path, dpi, extra=[leg])
    plt.close(fig)


def _plotly_3d_pair(
    Z_pre: np.ndarray,
    Z_con: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    label_fn,
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
        subplot_titles=(TITLE_PRE, TITLE_CON),
    )
    uniq = sorted(set(labels.tolist()))
    for sp in uniq:
        m = labels == sp
        pretty = str(label_fn(sp)).replace("$", "").replace("\\mathit{", "").replace("}", "").replace("\\ ", " ")
        for col, Z in ((1, Z_pre), (2, Z_con)):
            fig.add_trace(
                go.Scatter3d(
                    x=Z[m, 0],
                    y=Z[m, 1],
                    z=Z[m, 2],
                    mode="markers",
                    name=pretty,
                    legendgroup=pretty,
                    showlegend=(col == 1),
                    marker=dict(size=4, opacity=0.8),
                    hovertemplate=f"{pretty}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        title="Frame-level UMAP (3D) — subset",
        height=700,
        legend=dict(font=dict(size=11)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main() -> None:
    args = _parse_args()
    if int(args.dpi) < 600:
        raise ValueError(f"--dpi must be >= 600 (got {args.dpi})")
    n_draws = max(1, int(args.n_random_draws))
    if n_draws > 1 and str(args.class_selection) != "random":
        raise ValueError("--n-random-draws > 1 requires --class-selection random")

    ignore_test = bool(args.ignore_test_files) and not bool(args.include_test_files)
    one_type = bool(args.one_type_per_species) and not bool(
        args.allow_multiple_types_per_species
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sci_map: Dict[str, str] = {}
    eng_map: Dict[str, str] = {}
    if str(args.legend_names) == "scientific":
        species_path = _resolve_species_list(str(args.species_list))
        sci_map, eng_map = _load_scientific_names(species_path)

    def label_fn(class_name: str) -> str:
        if str(args.legend_names) == "scientific":
            return _scientific_label(class_name, sci_map)
        return _pretty_label(class_name)

    legend_ncol = 2 if str(args.legend_names) == "scientific" else 4

    df_pre_all = _prepare_labeled(
        args.pretrain_dir,
        wav_csv_dir=args.wav_csv_dir,
        ignore_test=ignore_test,
        only_call=bool(args.only_call),
        segment_h5_use_recording_csv=bool(args.segment_h5_use_recording_csv),
    )
    df_con_all = _prepare_labeled(
        args.contrastive_dir,
        wav_csv_dir=args.wav_csv_dir,
        ignore_test=ignore_test,
        only_call=bool(args.only_call),
        segment_h5_use_recording_csv=bool(args.segment_h5_use_recording_csv),
    )
    df_pre_all, df_con_all = _align(df_pre_all, df_con_all)

    explicit = [x.strip() for x in str(args.classes).split(",") if x.strip()]
    exclude = [x.strip() for x in str(args.exclude_classes).split(",") if x.strip()]
    counts = df_pre_all["class_name"].astype(str).value_counts()
    summary_rows: List[dict] = []
    base_seed = int(args.random_state)

    for draw_i in range(1, n_draws + 1):
        seed = base_seed + draw_i - 1
        draw_dir = out_dir if n_draws == 1 else out_dir / f"draw_{draw_i:02d}_seed{seed}"
        draw_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== draw {draw_i}/{n_draws}  seed={seed}  out={draw_dir} =====")

        selected = _select_classes(
            df_pre_all["class_name"],
            n_classes=int(args.n_classes),
            explicit=explicit,
            exclude=exclude,
            one_type_per_species=one_type,
            min_label_count=int(args.min_label_count),
            class_selection=str(args.class_selection),
            vocal_type=str(args.vocal_type),
            random_state=seed,
        )
        sel_df = pd.DataFrame(
            {
                "class_name": selected,
                "aligned_frames": [int(counts.get(c, 0)) for c in selected],
                "pretty": [_pretty_label(c) for c in selected],
                "scientific_name": [sci_map.get(c, "") for c in selected],
                "english_name": [eng_map.get(c, "") for c in selected],
                "species": [_species_code(c) for c in selected],
                "class_selection": str(args.class_selection),
                "vocal_type": str(args.vocal_type),
                "random_state": seed,
                "draw": draw_i,
            }
        )
        sel_df.to_csv(draw_dir / "selected_classes.csv", index=False)
        for _, row in sel_df.iterrows():
            summary_rows.append(row.to_dict())

        df_pre, df_con = _apply_subset(
            df_pre_all,
            df_con_all,
            selected,
            max_per_label=int(args.max_per_label),
            random_state=seed,
        )

        labels = df_pre["class_name"].astype(str).to_numpy()
        styles = _subset_styles(labels)
        X_pre = _stack_X(df_pre)
        X_con = _stack_X(df_con)
        print(f"X_pre {X_pre.shape}, X_con {X_con.shape}")

        draw_args = argparse.Namespace(**vars(args))
        draw_args.random_state = seed

        print("Fitting UMAP 2D (pretrained)…")
        Z2_pre = _fit_umap(X_pre, 2, draw_args)
        print("Fitting UMAP 2D (contrastive)…")
        Z2_con = _fit_umap(X_con, 2, draw_args)
        print("Fitting UMAP 3D (pretrained)…")
        Z3_pre = _fit_umap(X_pre, 3, draw_args)
        print("Fitting UMAP 3D (contrastive)…")
        Z3_con = _fit_umap(X_con, 3, draw_args)

        _plot_2d_pair(
            Z2_pre,
            Z2_con,
            labels,
            styles,
            draw_dir / "umap_2d_pretrain_vs_ssl_contrastive_subset.png",
            s=float(args.point_size),
            alpha=float(args.alpha),
            dpi=int(args.dpi),
            label_fn=label_fn,
            legend_ncol=legend_ncol,
        )
        _plot_3d_pair(
            Z3_pre,
            Z3_con,
            labels,
            styles,
            draw_dir / "umap_3d_pretrain_vs_ssl_contrastive_subset.png",
            s=float(args.point_size),
            alpha=float(args.alpha),
            dpi=int(args.dpi),
            label_fn=label_fn,
            legend_ncol=legend_ncol,
        )

        np.savez_compressed(
            draw_dir / "umap_coords.npz",
            Z2_pre=Z2_pre,
            Z2_con=Z2_con,
            Z3_pre=Z3_pre,
            Z3_con=Z3_con,
            labels=labels,
            align_key=df_pre["align_key"].astype(str).to_numpy(),
        )
        pd.DataFrame(
            {
                "align_key": df_pre["align_key"].astype(str),
                "class_name": labels,
                "scientific_name": [sci_map.get(c, "") for c in labels],
                "wav": df_pre["wav"].astype(str),
                "time_s": df_pre["time_s"].astype(float),
            }
        ).to_csv(draw_dir / "umap_sample_meta.csv", index=False)

        if not args.no_plotly:
            _plotly_3d_pair(
                Z3_pre,
                Z3_con,
                labels,
                draw_dir / "umap_3d_pretrain_vs_ssl_contrastive_subset.html",
                label_fn=label_fn,
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "draws_summary.csv", index=False)
    print(f"Wrote figures under {out_dir} ({n_draws} draw(s), dpi={int(args.dpi)})")


if __name__ == "__main__":
    main()
