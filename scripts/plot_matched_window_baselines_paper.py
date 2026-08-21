#!/usr/bin/env python3
"""
Paper figures: one metric per figure, two side-by-side panels.

(a) 3 s windows: Animal2Vec vs BirdNET
(b) 5 s windows: Animal2Vec vs Perch

Existing native-dim vote-eval CSV only (no re-run). Metrics: NMI, AMI, ARI.

  python scripts/plot_matched_window_baselines_paper.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "paper" / "figures" / "matched_window_vote_metrics.csv"
OUT_DIR = ROOT / "paper" / "figures"

A2V_LABEL = "Self-supervised contrastive learning"
C_A2V = "#0072B2"
C_BASE = "#D55E00"

PANELS = [
    {
        "letter": "a",
        "title": r"$3\,\mathrm{s}$ windows",
        "a2v": "a2v_3s_vote",
        "baseline": "birdnet_3s",
        "baseline_label": "BirdNET",
        "n": 310,
    },
    {
        "letter": "b",
        "title": r"$5\,\mathrm{s}$ windows",
        "a2v": "a2v_5s_vote",
        "baseline": "perch_5s",
        "baseline_label": "Perch",
        "n": 141,
    },
]

FIGURES = [
    ("nmi", "NMI", "fig_nmi_matched_windows", (0.0, 1.0)),
    ("ami", "AMI", "fig_ami_matched_windows", (0.0, 0.70)),
    ("ari", "ARI", "fig_ari_matched_windows", (0.0, 0.60)),
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
            "legend.fontsize": 7.0,
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


def _series(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = df.loc[df["representation"] == name].sort_values("k")
    if g.empty:
        raise SystemExit(f"No rows for {name}")
    return g


def plot_metric(df: pd.DataFrame, col: str, ylabel: str, stem: str, ylim: tuple) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for ax, spec in zip(axes, PANELS):
        a2v = _series(df, spec["a2v"])
        base = _series(df, spec["baseline"])
        ax.plot(
            a2v["k"],
            a2v[col],
            color=C_A2V,
            linestyle="-",
            marker="o",
            markersize=3.2,
            label=A2V_LABEL,
        )
        ax.plot(
            base["k"],
            base[col],
            color=C_BASE,
            linestyle="--",
            marker="s",
            markersize=3.0,
            label=spec["baseline_label"],
        )
        ax.set_title(spec["title"] + rf" ($n={spec['n']}$)")
        ax.set_xlabel(r"Number of clusters $k$")
        ax.set_xlim(0, 205)
        ax.set_ylim(*ylim)
        _spine(ax)
        _panel_letter(ax, spec["letter"])
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(w_pad=1.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    svg = OUT_DIR / f"{stem}.svg"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    print(f"Wrote {pdf}")


def main() -> None:
    _style()
    df = pd.read_csv(CSV)
    for col, ylabel, stem, ylim in FIGURES:
        plot_metric(df, col, ylabel, stem, ylim)


if __name__ == "__main__":
    main()
