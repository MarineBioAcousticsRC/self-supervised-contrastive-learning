#!/usr/bin/env python3
"""
N=31 mean ARI / AMI versus k, with a sample-s.d. band and mean elbow k.

Uses the per-trial MiniBatch k-sweeps already written by run_n_trials.py
(k=10..200 step 10). Class-aware is the 50-50-coin rerun
(multi_trial_20260903_class_aware_noise); the other three arms are
multi_trial_20260827.

Style matches plot_matched_window_baselines_paper.py (serif, two-panel
layout, Okabe–Ito colors).

  python self-supervised-contrastive-learning/scripts/plot_ari_ami_vs_k_n31.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "kmeans_sweep" / "n31_vs_k"
RAW_DUMP = DATA_DIR / "raw_dump.txt"
SWEEP_CSV = DATA_DIR / "sweeps_long.csv"
ELBOW_CSV = DATA_DIR / "elbows.csv"
SUMMARY_CSV = DATA_DIR / "summary_mean_std_vs_k.csv"
OUT_DIR = ROOT / "paper" / "figures"
JASA_OUT = ROOT.parent / "paper" / "figures"

PAPER_ORDER = [
    ("ssl_pretrain", "Self-supervised pretraining"),
    ("supervised_finetune", "Supervised finetuning"),
    ("ssl_contrastive", "Self-supervised contrastive learning"),
    ("supervised_contrastive", "Supervised contrastive learning"),
]

# Okabe–Ito; SSL solid blue / class-aware dashed orange match the
# BirdNET matched-window figures (Ours vs baseline).
STYLE = {
    "ssl_pretrain": dict(color="#000000", ls=":", marker="^", ms=3.2),
    "supervised_finetune": dict(color="#009E73", ls="-.", marker="D", ms=3.0),
    "ssl_contrastive": dict(color="#0072B2", ls="-", marker="o", ms=3.4),
    "supervised_contrastive": dict(color="#D55E00", ls="--", marker="s", ms=3.2),
}

FIGURES = [
    ("ARI", "ARI", "fig_ari_vs_k_n31", (0.0, 0.60)),
    ("AMI", "AMI", "fig_ami_vs_k_n31", (0.0, 0.85)),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
        }
    )


def _spine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        f"({letter})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="bottom",
        ha="left",
    )


def parse_raw_dump(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = path.read_text(encoding="utf-8")
    if "===SWEEPS===" not in text or "===ELBOWS===" not in text:
        raise SystemExit(f"Missing section markers in {path}")
    body = text.split("===SWEEPS===", 1)[1]
    sweep_txt, rest = body.split("===ELBOWS===", 1)
    elbow_txt = rest.strip()
    sweeps = pd.read_csv(pd.io.common.StringIO(sweep_txt.strip()))
    elbows = pd.read_csv(pd.io.common.StringIO(elbow_txt.strip()))
    for col in ("trial", "seed", "k", "ARI", "AMI", "NMI", "RI", "Silhouette", "Inertia"):
        if col in sweeps.columns:
            sweeps[col] = pd.to_numeric(sweeps[col], errors="coerce")
    for col in ("trial", "seed", "selected_k", "ari", "ami", "nmi"):
        if col in elbows.columns:
            elbows[col] = pd.to_numeric(elbows[col], errors="coerce")
    return sweeps, elbows


def aggregate(sweeps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in sweeps.groupby("model", sort=False):
        for k, gk in g.groupby("k", sort=True):
            n = int(len(gk))
            if n != 31:
                raise SystemExit(f"{model} k={k} has n={n}, expected 31")
            row = {"model": model, "k": int(k), "n": n}
            for col in ("ARI", "AMI", "NMI", "RI", "Silhouette"):
                vals = gk[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def elbow_means(elbows: pd.DataFrame) -> dict[str, float]:
    out = {}
    for model, g in elbows.groupby("model", sort=False):
        if len(g) != 31:
            raise SystemExit(f"{model} elbows n={len(g)}, expected 31")
        out[str(model)] = float(g["selected_k"].astype(float).mean())
    return out


def _draw_models(ax, summary: pd.DataFrame, elbows: dict[str, float], col: str) -> None:
    mean_c = f"{col}_mean"
    std_c = f"{col}_std"
    for model, label in PAPER_ORDER:
        g = summary.loc[summary["model"] == model].sort_values("k")
        if g.empty:
            raise SystemExit(f"No summary rows for {model}")
        st = STYLE[model]
        k = g["k"].to_numpy(dtype=float)
        y = g[mean_c].to_numpy(dtype=float)
        s = g[std_c].to_numpy(dtype=float)
        ax.fill_between(k, y - s, y + s, color=st["color"], alpha=0.18, linewidth=0, zorder=1)
        ax.plot(
            k,
            y,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=st["ms"],
            label=label,
            zorder=3,
        )
        k_e = elbows[model]
        y_e = float(np.interp(k_e, k, y))
        ax.axvline(k_e, color=st["color"], linestyle=":", linewidth=0.9, alpha=0.85, zorder=2)
        ax.plot(k_e, y_e, marker="o", ms=6.4, color=st["color"], mfc="white", mew=1.35, zorder=4)


def _legend_entries(ax):
    return ax.get_legend_handles_labels()


def _combined_legend_entries():
    """Combined two-panel legend: four models only, no mean-elbow-k, no circles."""
    handles = []
    labels = []
    for model, label in PAPER_ORDER:
        st = STYLE[model]
        kwargs = dict(color=st["color"], linestyle=st["ls"], linewidth=1.4)
        if st["marker"] != "o":
            kwargs.update(marker=st["marker"], markersize=st["ms"])
        handles.append(Line2D([0], [0], **kwargs))
        labels.append(label)
    return handles, labels


def _place_fig_legend(fig, handles, labels, ncol: int = 3) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.3,
        handletextpad=0.5,
        borderaxespad=0.2,
    )


def plot_single(
    summary: pd.DataFrame,
    elbows: dict[str, float],
    col: str,
    ylabel: str,
    stem: str,
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.45))
    _draw_models(ax, summary, elbows, col)
    ax.set_xlabel(r"Number of clusters $k$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 205)
    ax.set_ylim(*ylim)
    _spine(ax)
    fig.tight_layout()
    _place_fig_legend(fig, *_legend_entries(ax), ncol=2)
    _save(fig, stem)


def plot_combined(summary: pd.DataFrame, elbows: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharex=True)
    for ax, (col, ylabel, _, ylim), letter in zip(
        axes,
        FIGURES,
        ("a", "b"),
    ):
        _draw_models(ax, summary, elbows, col)
        ax.set_xlabel(r"Number of clusters $k$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 205)
        ax.set_ylim(*ylim)
        _spine(ax)
        _panel_letter(ax, letter)
    fig.tight_layout(w_pad=1.4)
    _place_fig_legend(fig, *_combined_legend_entries(), ncol=2)
    _save(fig, "fig_ari_ami_vs_k_n31")


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JASA_OUT.mkdir(parents=True, exist_ok=True)
    for folder in (OUT_DIR, JASA_OUT):
        for ext in ("png", "svg", "pdf"):
            path = folder / f"{stem}.{ext}"
            fig.savefig(path)
            print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    _style()
    if not RAW_DUMP.is_file():
        raise SystemExit(f"Missing {RAW_DUMP}")
    sweeps, elbows_df = parse_raw_dump(RAW_DUMP)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sweeps.to_csv(SWEEP_CSV, index=False)
    elbows_df.to_csv(ELBOW_CSV, index=False)
    summary = aggregate(sweeps)
    summary.to_csv(SUMMARY_CSV, index=False)
    elbows = elbow_means(elbows_df)
    print("Mean elbow k (N=31):")
    for model, label in PAPER_ORDER:
        print(f"  {label:28s} {elbows[model]:.3f}")
    plot_combined(summary, elbows)
    for col, ylabel, stem, ylim in FIGURES:
        plot_single(summary, elbows, col, ylabel, stem, ylim)


if __name__ == "__main__":
    main()
