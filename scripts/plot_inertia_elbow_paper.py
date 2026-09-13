#!/usr/bin/env python3
"""
Publication figures for unsupervised k selection by the inertia elbow.

The elbow is the k that maximizes perpendicular distance from the
min–max-normalized (k, WCSS) curve to the chord joining the first and
last sweep points — the same rule as analyze_k_selection_metrics.py.

Outputs (PDF + 600 dpi PNG) under paper/figures/:
  fig_elbow_method            how the inertia elbow selects k
  fig_elbow_four_models       inertia vs k for all four representations (elbow only)
  fig_elbow_operating_points  new elbow k vs old silhouette k, plus scores at elbow k
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "results" / "kmeans_sweep" / "four_representations" / "kmeans_sweep.csv"
OUT_DIR = ROOT / "paper" / "figures"

# Must match analyze_k_selection_metrics.py / reported k values.
EXPECTED_ELBOW = {
    "ssl_pretrain": 120,
    "supervised_finetune": 40,
    "supervised_contrastive": 70,
    "ssl_contrastive": 60,
}

MODEL_LABELS = {
    "ssl_pretrain": "Self-supervised pretraining",
    "supervised_finetune": "Supervised finetuning",
    "ssl_contrastive": "Self-supervised contrastive learning",
    "supervised_contrastive": "Supervised contrastive learning",
}
# Same order as Sec. III A (Compared representations).
MODEL_ORDER = ["ssl_pretrain", "supervised_finetune", "ssl_contrastive", "supervised_contrastive"]

# Print-safe Okabe–Ito-ish palette.
C_CURVE = "#000000"
C_CHORD = "#777777"
C_ELBOW = "#D55E00"
C_ORACLE = "#009E73"
C_DIST = "#0072B2"
C_MUTED = "#B0B0B0"

EXPECTED_ARI_ORACLE = {
    "ssl_pretrain": 200,
    "supervised_finetune": 60,
    "supervised_contrastive": 130,
    "ssl_contrastive": 50,
}

# Previous operating point (max silhouette), for contrast only.
SILHOUETTE_K = {
    "ssl_pretrain": 110,
    "supervised_finetune": 40,
    "supervised_contrastive": 130,
    "ssl_contrastive": 190,
}

PAPER_ORDER = [
    ("ssl_pretrain", "Self-supervised pretraining"),
    ("supervised_finetune", "Supervised finetuning"),
    ("ssl_contrastive", "Self-supervised contrastive learning"),
    ("supervised_contrastive", "Supervised contrastive learning"),
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
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
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
    ax.grid(False)


def _panel(ax, letter: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="bottom",
        ha="left",
    )


def elbow_geometry(k: np.ndarray, inertia: np.ndarray) -> Dict[str, np.ndarray]:
    """Scale-invariant elbow: max perp. distance to the endpoint chord."""
    k = np.asarray(k, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    k_span = max(float(k.max() - k.min()), 1e-12)
    i_span = max(float(inertia.max() - inertia.min()), 1e-12)
    kn = (k - k.min()) / k_span
    in_ = (inertia - inertia.min()) / i_span
    p1 = np.array([kn[0], in_[0]])
    p2 = np.array([kn[-1], in_[-1]])
    line = p2 - p1
    norm = float(np.linalg.norm(line))
    if norm < 1e-12:
        dists = np.zeros(len(k))
        projs = np.repeat(p1[None, :], len(k), axis=0)
        idx = int(np.argmin(inertia))
    else:
        dists = np.empty(len(k))
        projs = np.empty((len(k), 2))
        line2 = np.dot(line, line)
        for i in range(len(k)):
            p = np.array([kn[i], in_[i]])
            # 2-D cross product magnitude / |line|
            dists[i] = abs(line[0] * (p1[1] - p[1]) - line[1] * (p1[0] - p[0])) / norm
            t = float(np.dot(p - p1, line) / line2)
            projs[i] = p1 + t * line
        idx = int(np.argmax(dists))
    k_proj = projs[:, 0] * k_span + k.min()
    i_proj = projs[:, 1] * i_span + inertia.min()
    return {
        "k": k,
        "inertia": inertia,
        "kn": kn,
        "in_": in_,
        "dists": dists,
        "projs": projs,
        "k_proj": k_proj,
        "i_proj": i_proj,
        "p1": p1,
        "p2": p2,
        "idx": idx,
        "k_elbow": int(k[idx]),
    }


def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Model": "model", "k": "k", "Inertia": "inertia", "ARI": "ari"})
    return df.sort_values(["model", "k"]).reset_index(drop=True)


def geometry_for_model(df: pd.DataFrame, model: str) -> Tuple[pd.DataFrame, Dict]:
    g = df.loc[df["model"] == model].sort_values("k").reset_index(drop=True)
    geo = elbow_geometry(g["k"].to_numpy(), g["inertia"].to_numpy(dtype=float))
    expected = EXPECTED_ELBOW[model]
    if int(geo["k_elbow"]) != expected:
        raise SystemExit(
            f"Elbow mismatch for {model}: got {geo['k_elbow']}, expected {expected}"
        )
    return g, geo


def _save(fig: plt.Figure, stem: str, svg: bool = False, tight: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    # bbox_inches=None falls back to rcParams['savefig.bbox'] (tight here),
    # which crops panels unevenly. Override the rcParam when we want the
    # figure's true GridSpec layout.
    if tight:
        ctx = {"savefig.bbox": "tight"}
        kw = {"bbox_inches": "tight", "pad_inches": 0.04}
    else:
        ctx = {"savefig.bbox": None}
        kw = {"pad_inches": 0.02}
    with plt.rc_context(ctx):
        fig.savefig(pdf, **kw)
        fig.savefig(png, **kw)
        if svg:
            svg_path = OUT_DIR / f"{stem}.svg"
            fig.savefig(svg_path, format="svg", **kw)
            print(f"Wrote {svg_path}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    plt.close(fig)


def plot_method_figure(df: pd.DataFrame) -> None:
    """How the elbow rule is applied, using the self-supervised contrastive sweep."""
    g, geo = geometry_for_model(df, "ssl_contrastive")
    k = geo["k"]
    inertia = geo["inertia"]
    idx = int(geo["idx"])
    k_e = int(geo["k_elbow"])
    k_ari = int(g.loc[g["ari"].astype(float).idxmax(), "k"])
    ari_e = float(g.loc[g["k"] == k_e, "ari"].iloc[0])
    ari_o = float(g.loc[g["k"] == k_ari, "ari"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.15))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) Raw WCSS vs k, with endpoint chord.
    ax_a.plot(k, inertia / 1e6, color=C_CURVE, marker="o", ms=3.4, lw=1.15, zorder=3)
    ax_a.plot(
        [k[0], k[-1]],
        [inertia[0] / 1e6, inertia[-1] / 1e6],
        ls=(0, (3.5, 2.2)),
        color=C_CHORD,
        lw=1.15,
        zorder=2,
        label="Endpoint chord",
    )
    ax_a.axvline(k_e, color=C_ELBOW, ls="-", lw=1.05, alpha=0.95, zorder=1)
    ax_a.plot(k[idx], inertia[idx] / 1e6, marker="o", ms=7.5, color=C_ELBOW, zorder=4)
    ax_a.annotate(
        rf"$k_{{\mathrm{{elbow}}}}={k_e}$",
        xy=(k[idx], inertia[idx] / 1e6),
        xytext=(k_e + 28, inertia[idx] / 1e6 + 0.12),
        color=C_ELBOW,
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color=C_ELBOW, lw=0.7),
    )
    ax_a.set_xlabel(r"Number of clusters $k$")
    ax_a.set_ylabel(r"Inertia (WCSS, $\times 10^{6}$)")
    ax_a.set_title("Inertia curve and chord", loc="left", pad=4)
    ax_a.set_xlim(5, 205)
    _spine(ax_a)
    _panel(ax_a, "a")
    ax_a.legend(frameon=False, loc="upper right")

    # (b) Unit-square geometry: this is the space in which distance is measured.
    ax_b.plot(geo["kn"], geo["in_"], color=C_CURVE, marker="o", ms=3.2, lw=1.15, zorder=3)
    ax_b.plot(
        [geo["p1"][0], geo["p2"][0]],
        [geo["p1"][1], geo["p2"][1]],
        ls=(0, (3.5, 2.2)),
        color=C_CHORD,
        lw=1.15,
        zorder=2,
    )
    for i in range(len(k)):
        ax_b.plot(
            [geo["kn"][i], geo["projs"][i, 0]],
            [geo["in_"][i], geo["projs"][i, 1]],
            color=C_MUTED,
            lw=0.55,
            zorder=1,
        )
    ax_b.plot(
        [geo["kn"][idx], geo["projs"][idx, 0]],
        [geo["in_"][idx], geo["projs"][idx, 1]],
        color=C_ELBOW,
        lw=1.7,
        zorder=4,
    )
    ax_b.plot(geo["kn"][idx], geo["in_"][idx], marker="o", ms=7.5, color=C_ELBOW, zorder=5)
    ax_b.plot(
        geo["projs"][idx, 0],
        geo["projs"][idx, 1],
        marker="s",
        ms=4.5,
        color=C_ELBOW,
        zorder=5,
    )
    ax_b.annotate(
        r"$d(k)$ max",
        xy=(
            0.5 * (geo["kn"][idx] + geo["projs"][idx, 0]),
            0.5 * (geo["in_"][idx] + geo["projs"][idx, 1]),
        ),
        xytext=(0.42, 0.62),
        color=C_ELBOW,
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color=C_ELBOW, lw=0.7),
    )
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlim(-0.06, 1.06)
    ax_b.set_ylim(-0.06, 1.06)
    ax_b.set_xlabel(r"Normalized $k$")
    ax_b.set_ylabel("Normalized inertia")
    ax_b.set_title("Scale-invariant geometry", loc="left", pad=4)
    _spine(ax_b)
    _panel(ax_b, "b", x=-0.18)

    # (c) Selection statistic.
    ax_c.plot(k, geo["dists"], color=C_DIST, marker="o", ms=3.4, lw=1.15, zorder=3)
    ax_c.axvline(k_e, color=C_ELBOW, lw=1.05, zorder=1)
    ax_c.plot(k[idx], geo["dists"][idx], marker="o", ms=7.5, color=C_ELBOW, zorder=4)
    ax_c.fill_between(k, 0, geo["dists"], color=C_DIST, alpha=0.08, zorder=0)
    ax_c.set_xlabel(r"Number of clusters $k$")
    ax_c.set_ylabel(r"Perpendicular distance $d(k)$")
    ax_c.set_title("Elbow statistic (max selects $k$)", loc="left", pad=4)
    ax_c.set_xlim(5, 205)
    ax_c.set_ylim(bottom=0)
    _spine(ax_c)
    _panel(ax_c, "c")

    # (d) Diagnostic: ARI was not used to choose k.
    ax_d.plot(g["k"], g["ari"], color=C_CURVE, marker="o", ms=3.4, lw=1.15, zorder=3)
    ax_d.axvline(k_e, color=C_ELBOW, lw=1.05, label=rf"Elbow $k={k_e}$", zorder=1)
    ax_d.axvline(
        k_ari,
        color=C_ORACLE,
        ls=(0, (2.4, 1.6)),
        lw=1.15,
        label=rf"ARI oracle $k={k_ari}$",
        zorder=1,
    )
    ax_d.plot(k_e, ari_e, marker="o", ms=7.0, color=C_ELBOW, zorder=4)
    ax_d.plot(k_ari, ari_o, marker="D", ms=5.5, color=C_ORACLE, zorder=4)
    ax_d.set_xlabel(r"Number of clusters $k$")
    ax_d.set_ylabel("Adjusted Rand index")
    ax_d.set_title("Diagnostic only (labels unused for $k$)", loc="left", pad=4)
    ax_d.set_xlim(5, 205)
    ax_d.legend(frameon=False, loc="lower right")
    _spine(ax_d)
    _panel(ax_d, "d")

    fig.suptitle(
        "Self-supervised contrastive learning  ·  MiniBatch $k$-means inertia elbow",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    _save(fig, "fig_elbow_method")


# Two lines each, same line count so (a)–(d) share one baseline.
PANEL_HEADS = {
    "ssl_pretrain": "Self-supervised\npretraining",
    "supervised_finetune": "Supervised\nfinetuning",
    "ssl_contrastive": "Self-supervised\ncontrastive learning",
    "supervised_contrastive": "Supervised\ncontrastive learning",
}


def plot_four_models(df: pd.DataFrame) -> None:
    geos = [geometry_for_model(df, m)[1] for m in MODEL_ORDER]
    letters = "abcd"
    fig = plt.figure(figsize=(7.16, 2.48))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=[0.11, 0.22, 1.0],
        hspace=0.12,
        wspace=0.28,
        left=0.08,
        right=0.995,
        top=0.97,
        bottom=0.17,
    )
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles=[
            Line2D([0], [0], color=C_CURVE, marker="o", ms=3.2, lw=1.15),
            Line2D([0], [0], color=C_CHORD, ls=(0, (3.5, 2.2)), lw=1.1),
            Line2D([0], [0], color=C_ELBOW, marker="o", ms=5.5, lw=1.15),
        ],
        labels=["Inertia", "Endpoint chord", r"Elbow $k$"],
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.4,
        handlelength=1.4,
        fontsize=8,
        borderaxespad=0.0,
    )

    plot_axes = []
    for col, (model, geo, letter) in enumerate(zip(MODEL_ORDER, geos, letters)):
        tax = fig.add_subplot(gs[1, col])
        tax.set_axis_off()
        tax.set_xlim(0, 1)
        tax.set_ylim(0, 1)
        tax.text(
            0.5,
            1.0,
            f"({letter})",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
            transform=tax.transAxes,
            clip_on=False,
        )
        tax.text(
            0.5,
            0.58,
            PANEL_HEADS[model],
            ha="center",
            va="top",
            fontsize=7.0,
            linespacing=1.15,
            transform=tax.transAxes,
            clip_on=False,
        )

        ax = fig.add_subplot(gs[2, col], sharex=plot_axes[0] if plot_axes else None)
        plot_axes.append(ax)
        k = geo["k"]
        inertia = geo["inertia"]
        idx = int(geo["idx"])
        k_e = int(geo["k_elbow"])
        exp = int(np.floor(np.log10(max(float(np.max(inertia)), 1.0))))
        scale = 10.0 ** exp
        y = inertia / scale

        ax.plot(k, y, color=C_CURVE, marker="o", ms=3.2, lw=1.15, zorder=3)
        ax.plot(
            [k[0], k[-1]],
            [y[0], y[-1]],
            ls=(0, (3.5, 2.2)),
            color=C_CHORD,
            lw=1.1,
            zorder=2,
        )
        ax.axvline(k_e, color=C_ELBOW, lw=1.15, zorder=1)
        ax.plot(k[idx], y[idx], marker="o", ms=7.0, color=C_ELBOW, zorder=4)
        ax.yaxis.get_offset_text().set_visible(False)
        # Journal-style scale factor: ×10^n above the y-axis, outside the frame
        # (matplotlib/AIP offset-text position). Not inside the data area.
        ax.text(
            0.0,
            1.02,
            rf"$\times$10$^{{{exp}}}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
        ax.set_xlim(0, 210)
        ax.set_xlabel(r"$k$")
        if col == 0:
            ax.set_ylabel("Inertia (WCSS)")
        ax.text(
            0.96,
            0.92,
            rf"$k={k_e}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=C_ELBOW,
            fontsize=8.5,
            fontweight="bold",
        )
        _spine(ax)

    _save(fig, "fig_elbow_four_models", svg=True, tight=False)


def plot_elbow_operating_points(df: pd.DataFrame) -> None:
    """Show the new inertia-elbow k (vs old silhouette k) and scores at elbow k."""
    fig, (ax_k, ax_tab) = plt.subplots(
        1, 2, figsize=(7.16, 2.95), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )

    y = np.arange(len(PAPER_ORDER))[::-1]
    x_old, x_new = 0.0, 1.0
    for yi, (model, label) in zip(y, PAPER_ORDER):
        k_old = SILHOUETTE_K[model]
        k_new = EXPECTED_ELBOW[model]
        ax_k.plot([x_old, x_new], [yi, yi], color=C_MUTED, lw=1.0, zorder=1)
        ax_k.plot(x_old, yi, marker="o", ms=6.5, color="#4C4C4C", zorder=3)
        ax_k.plot(x_new, yi, marker="o", ms=7.5, color=C_ELBOW, zorder=3)
        ax_k.text(x_old - 0.08, yi, str(k_old), ha="right", va="center", fontsize=8, color="#4C4C4C")
        ax_k.text(x_new + 0.08, yi, str(k_new), ha="left", va="center", fontsize=8.5, color=C_ELBOW, fontweight="bold")
        ax_k.text(-0.55, yi, label, ha="right", va="center", fontsize=8)

    ax_k.set_xlim(-1.7, 1.45)
    ax_k.set_ylim(-0.55, len(PAPER_ORDER) - 0.45)
    ax_k.set_xticks([x_old, x_new])
    ax_k.set_xticklabels(["Old: max silhouette", "New: inertia elbow"])
    ax_k.set_yticks([])
    ax_k.set_title(r"Operating $k$ after switching the selector", loc="left", pad=4)
    for sp in ax_k.spines.values():
        sp.set_visible(False)
    ax_k.tick_params(length=0)
    _panel(ax_k, "a", x=-0.02, y=1.08)

    headers = [r"Representation", r"$k$", "Sil.", "RI", "ARI", "NMI", "AMI"]
    cell = [headers]
    for model, label in PAPER_ORDER:
        g, _ = geometry_for_model(df, model)
        r = g.loc[g["k"].astype(int) == EXPECTED_ELBOW[model]].iloc[0]
        cell.append(
            [
                label,
                f"{EXPECTED_ELBOW[model]}",
                f"{float(r['Silhouette']):.3f}",
                f"{float(r['RI']):.3f}",
                f"{float(r['ari']):.3f}",
                f"{float(r['NMI']):.3f}",
                f"{float(r['AMI']):.3f}",
            ]
        )

    ax_tab.axis("off")
    ax_tab.set_title("Scores at inertia-elbow $k$", loc="left", pad=4)
    tbl = ax_tab.table(
        cellText=cell[1:],
        colLabels=cell[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.08, 1.6)
    tbl.auto_set_column_width(col=list(range(len(headers))))
    for (row, col), cell_obj in tbl.get_celld().items():
        cell_obj.set_linewidth(0.4)
        cell_obj.set_edgecolor("#CCCCCC")
        if row == 0:
            cell_obj.set_facecolor("#F3F3F3")
            cell_obj.set_text_props(fontweight="bold", fontsize=7)
        elif col == 1:
            cell_obj.set_text_props(color=C_ELBOW, fontweight="bold")
        if col == 0:
            cell_obj.set_text_props(ha="left")
            cell_obj.PAD = 0.08
    _panel(ax_tab, "b", x=-0.02, y=1.08)

    fig.tight_layout(w_pad=1.6)
    _save(fig, "fig_elbow_operating_points")


def write_caption_tex() -> None:
    path = OUT_DIR / "fig_elbow_captions.tex"
    path.write_text(
        r"""% Figure captions for the inertia-elbow plots.

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{figures/fig_elbow_method}
\caption{\label{fig:elbow_method}
Unsupervised selection of $k$ by the MiniBatch $k$-means inertia elbow
(self-supervised contrastive embeddings; $184{,}603$ frames).
The $k$-sweep recorded several scores, but only inertia (WCSS) was used
to choose $k$.
(a)~Inertia versus $k=10,20,\ldots,200$, with the endpoint chord (dashed).
(b)~The same curve after min--max normalizing both axes; gray segments are
perpendiculars to the chord and the orange segment is the longest.
(c)~That distance $d(k)$. The maximizer is $k=60$.
(d)~ARI versus $k$, diagnostic only: labels were not used to choose $k$.}
\end{figure*}

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{figures/fig_elbow_four_models}
\caption{\label{fig:elbow_four_models}
Inertia versus number of clusters $k$ for the four representations
in Sec.~\ref{sec:compared_reps}
(training split, $184{,}603$ frames).
(a)~Self-supervised pretraining, elbow $k=120$.
(b)~Supervised finetuning, $k=40$.
(c)~Self-supervised contrastive learning, $k=60$.
(d)~Supervised contrastive learning, $k=70$ (seed 42, 50-50 noise coin).
The dashed line is the endpoint chord; the orange mark is the inertia elbow.}
\end{figure*}

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{figures/fig_elbow_operating_points}
\caption{\label{fig:elbow_operating_points}
Latest evaluation uses the inertia elbow, not max silhouette.
(a)~Change in operating $k$ when the selector is switched.
(b)~Clustering scores at the inertia-elbow $k$ on the training split.}
\end{figure*}
""",
        encoding="utf-8",
    )
    print(f"Wrote {path}")


def main() -> None:
    _style()
    df = load_sweep(SWEEP_CSV)
    plot_method_figure(df)
    plot_four_models(df)
    plot_elbow_operating_points(df)
    write_caption_tex()


if __name__ == "__main__":
    main()
