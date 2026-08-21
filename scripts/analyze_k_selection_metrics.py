#!/usr/bin/env python3
"""
Compare unsupervised k-selection rules on an existing KMeans sweep CSV.

Selectors (unsupervised):
  - max silhouette
  - elbow on inertia (max distance of curve to chord)
  - max Calinski–Harabasz (if column present)
  - min Davies–Bouldin (if column present)

Oracles (diagnostic only; use labels to score selectors, not to choose k in the method):
  - max ARI
  - max NMI

Example:

  python scripts/analyze_k_selection_metrics.py \\
    --csv results/kmeans_sweep/self_supervised_contrastive/kmeans_sweep.csv \\
    --out-dir results/kmeans_sweep/self_supervised_contrastive/k_selection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COL_ALIASES = {
    "k": ["k", "K"],
    "silhouette": ["silhouette", "Silhouette"],
    "ari": ["ari_vs_class_name", "ARI", "ari"],
    "nmi": ["nmi_vs_class_name", "NMI", "nmi"],
    "ami": ["ami_vs_class_name", "AMI", "ami"],
    "inertia": ["kmeans_inertia", "Inertia", "inertia"],
    "calinski_harabasz": [
        "calinski_harabasz",
        "Calinski_Harabasz",
        "calinski_harabasz_score",
        "CH",
    ],
    "davies_bouldin": [
        "davies_bouldin",
        "Davies_Bouldin",
        "davies_bouldin_score",
        "DB",
    ],
    "model": ["tag", "Model", "model"],
}


def resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for name in COL_ALIASES[key]:
        if name in df.columns:
            return name
    return None


def elbow_k(k: np.ndarray, inertia: np.ndarray) -> int:
    """k at max perpendicular distance from (k, inertia) to the chord endpoints."""
    k = np.asarray(k, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    if len(k) < 3:
        return int(k[int(np.argmin(inertia))])
    # Normalize axes so distance is scale-invariant
    kn = (k - k.min()) / max(k.max() - k.min(), 1e-12)
    in_ = (inertia - inertia.min()) / max(inertia.max() - inertia.min(), 1e-12)
    p1 = np.array([kn[0], in_[0], 0.0])
    p2 = np.array([kn[-1], in_[-1], 0.0])
    line = p2 - p1
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return int(k[int(np.argmin(inertia))])
    dists = []
    for i in range(len(k)):
        p = np.array([kn[i], in_[i], 0.0])
        dists.append(np.linalg.norm(np.cross(line, p1 - p)) / norm)
    return int(k[int(np.argmax(dists))])


def row_at_k(g: pd.DataFrame, k_col: str, k_val: int) -> Optional[pd.Series]:
    m = g[k_col].astype(int) == int(k_val)
    if not m.any():
        return None
    return g.loc[m].iloc[0]


def select_k(
    g: pd.DataFrame,
    cols: Dict[str, str],
) -> List[Tuple[str, str, int]]:
    """Return list of (selector_name, kind, k). kind in {unsupervised, oracle}."""
    k_col = cols["k"]
    out: List[Tuple[str, str, int]] = []

    if cols.get("silhouette"):
        i = g[cols["silhouette"]].astype(float).idxmax()
        out.append(("max_silhouette", "unsupervised", int(g.loc[i, k_col])))

    if cols.get("inertia"):
        ek = elbow_k(g[k_col].to_numpy(), g[cols["inertia"]].to_numpy(dtype=float))
        out.append(("elbow_inertia", "unsupervised", ek))

    if cols.get("calinski_harabasz"):
        i = g[cols["calinski_harabasz"]].astype(float).idxmax()
        out.append(("max_calinski_harabasz", "unsupervised", int(g.loc[i, k_col])))

    if cols.get("davies_bouldin"):
        i = g[cols["davies_bouldin"]].astype(float).idxmin()
        out.append(("min_davies_bouldin", "unsupervised", int(g.loc[i, k_col])))

    if cols.get("ari"):
        i = g[cols["ari"]].astype(float).idxmax()
        out.append(("max_ari_oracle", "oracle", int(g.loc[i, k_col])))

    if cols.get("nmi"):
        i = g[cols["nmi"]].astype(float).idxmax()
        out.append(("max_nmi_oracle", "oracle", int(g.loc[i, k_col])))

    return out


def analyze_group(g: pd.DataFrame, model: str, cols: Dict[str, str]) -> pd.DataFrame:
    selections = select_k(g, cols)
    k_ari = None
    for name, kind, kv in selections:
        if name == "max_ari_oracle":
            k_ari = kv
            break

    rows = []
    for name, kind, kv in selections:
        r = row_at_k(g, cols["k"], kv)
        if r is None:
            continue
        ari = float(r[cols["ari"]]) if cols.get("ari") else float("nan")
        nmi = float(r[cols["nmi"]]) if cols.get("nmi") else float("nan")
        ami = float(r[cols["ami"]]) if cols.get("ami") else float("nan")
        sil = float(r[cols["silhouette"]]) if cols.get("silhouette") else float("nan")
        rows.append(
            {
                "model": model,
                "selector": name,
                "kind": kind,
                "selected_k": int(kv),
                "ari_at_k": ari,
                "nmi_at_k": nmi,
                "ami_at_k": ami,
                "silhouette_at_k": sil,
                "abs_k_minus_k_ari": abs(int(kv) - int(k_ari)) if k_ari is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(g: pd.DataFrame, model: str, cols: Dict[str, str], out_path: Path) -> None:
    k = g[cols["k"]].astype(int).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax = axes.ravel()

    series = [
        (cols.get("silhouette"), "Silhouette", "higher better"),
        (cols.get("inertia"), "Inertia (WCSS)", "elbow"),
        (cols.get("ari"), "ARI (oracle)", "diagnostic"),
        (cols.get("nmi"), "NMI (oracle)", "diagnostic"),
    ]
    for a, (c, title, note) in zip(ax, series):
        if c is None:
            a.set_visible(False)
            continue
        a.plot(k, g[c].astype(float).to_numpy(), marker="o", ms=3)
        a.set_title(f"{title} ({note})")
        a.set_xlabel("k")
        a.grid(True, alpha=0.3)

    # Overlay CH/DB if present (replace empty panels or add twin)
    if cols.get("calinski_harabasz") and ax[1].get_visible():
        ax2 = ax[1].twinx()
        ax2.plot(
            k,
            g[cols["calinski_harabasz"]].astype(float).to_numpy(),
            color="C1",
            marker="s",
            ms=3,
            label="CH",
        )
        ax2.set_ylabel("Calinski–Harabasz", color="C1")
    if cols.get("davies_bouldin"):
        # Prefer bottom-right if NMI panel exists; else use free axis
        a = ax[3] if ax[3].get_visible() else ax[2]
        a2 = a.twinx()
        a2.plot(
            k,
            g[cols["davies_bouldin"]].astype(float).to_numpy(),
            color="C3",
            marker="^",
            ms=3,
            alpha=0.8,
            label="DB",
        )
        a2.set_ylabel("Davies–Bouldin", color="C3")

    fig.suptitle(f"K-selection metrics — {model}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze unsupervised k-selection on a sweep CSV")
    p.add_argument("--csv", required=True, help="Long-form k-sweep CSV")
    p.add_argument("--out-dir", required=True, help="Output directory for tables + plots")
    p.add_argument(
        "--model",
        default=None,
        help="Optional model/tag filter (default: analyze each model in CSV)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    cols: Dict[str, Optional[str]] = {key: resolve_col(df, key) for key in COL_ALIASES}
    if cols["k"] is None:
        raise SystemExit(f"No k column found in {csv_path}. Columns: {list(df.columns)}")

    model_col = cols.get("model")
    if model_col is None:
        df = df.copy()
        df["__model__"] = "default"
        model_col = "__model__"

    if args.model is not None:
        df = df.loc[df[model_col].astype(str) == str(args.model)].copy()
        if df.empty:
            raise SystemExit(f"No rows for model={args.model}")

    # Drop None from cols for select_k
    resolved = {k: v for k, v in cols.items() if v is not None and k != "model"}

    all_cmp = []
    for model, g in df.groupby(model_col):
        g = g.sort_values(resolved["k"]).reset_index(drop=True)
        cmp_df = analyze_group(g, str(model), resolved)
        all_cmp.append(cmp_df)
        plot_metrics(g, str(model), resolved, out_dir / f"metrics_vs_k_{model}.png")
        print(f"\n=== {model} ===")
        print(cmp_df.to_string(index=False))

    summary = pd.concat(all_cmp, ignore_index=True)
    summary_path = out_dir / "selector_comparison.csv"
    summary.to_csv(summary_path, index=False)

    # Rank unsupervised selectors by closeness to ARI-oracle k, then by ARI@k
    uns = summary.loc[summary["kind"] == "unsupervised"].copy()
    if not uns.empty and uns["abs_k_minus_k_ari"].notna().any():
        uns = uns.sort_values(["model", "abs_k_minus_k_ari", "ari_at_k"], ascending=[True, True, False])
        best_path = out_dir / "best_unsupervised_selectors.csv"
        best = uns.groupby("model", as_index=False).first()
        best.to_csv(best_path, index=False)
        print(f"\nBest unsupervised selector(s) by |k - k_ARI| then ARI@k:")
        print(best.to_string(index=False))
        print(f"Wrote {best_path}")

    print(f"\nWrote {summary_path}")
    print(f"Plots under {out_dir}")


if __name__ == "__main__":
    main()
