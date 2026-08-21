#!/usr/bin/env python3
"""
UMAP → MiniBatchKMeans k-sweep for matched-window embeddings.

Fits one UMAP per representation, then sweeps MiniBatchKMeans over
k = 10..200 (step 10) on Z — same family as scripts/kmeans_silhouette_sweep.py.

  python -m experiments.matched_window.evaluate \\
    --out-dir /path/to/data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.matched_window.cluster_eval import (  # noqa: E402
    evaluate_clustering,
    run_minibatch_kmeans,
)
from experiments.matched_window.config import (  # noqa: E402
    DEFAULT_OUT_DIR,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    KMEANS_K_STEP,
    RANDOM_SEED,
)
from experiments.matched_window.sanity import assert_matched_keys  # noqa: E402
from experiments.matched_window.umap_fit import (  # noqa: E402
    fit_umap,
    fit_umap_2d,
    save_umap,
)

REPRESENTATIONS = (
    ("a2v_1s", "a2v_1s", "a2v_1s"),
    ("a2v_3s", "a2v_3s", "a2v_3s"),
    ("birdnet_3s", "birdnet_3s", "birdnet_3s"),
    ("a2v_5s", "a2v_5s", "a2v_5s"),
    ("perch_5s", "perch_5s", "perch_5s"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UMAP + MiniBatchKMeans k-sweep on matched-window outputs"
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--random-state", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--subset",
        choices=["single", "multi", "all"],
        default="single",
        help="Primary eval uses single-species windows; multi/all also written",
    )
    p.add_argument("--k-min", type=int, default=KMEANS_K_MIN)
    p.add_argument("--k-max", type=int, default=KMEANS_K_MAX)
    p.add_argument("--k-step", type=int, default=KMEANS_K_STEP)
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="If set, run only this single k (skip full sweep)",
    )
    return p.parse_args()


def _k_values(k_min: int, k_max: int, k_step: int, single_k: Optional[int]) -> List[int]:
    if single_k is not None:
        return [int(single_k)]
    if k_step <= 0:
        raise ValueError("--k-step must be > 0")
    return list(range(int(k_min), int(k_max) + 1, int(k_step)))


def _load_pair(out_dir: Path, emb_stem: str, meta_stem: str) -> Tuple[np.ndarray, pd.DataFrame]:
    emb_path = out_dir / "embeddings" / f"{emb_stem}.npy"
    meta_path = out_dir / "metadata" / f"{meta_stem}.csv"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    X = np.load(emb_path)
    df = pd.read_csv(meta_path)
    if len(df) != X.shape[0]:
        raise RuntimeError(
            f"Row mismatch for {emb_stem}: meta={len(df)} emb={X.shape[0]}"
        )
    return X, df


def _filter_subset(df: pd.DataFrame, X: np.ndarray, subset: str) -> Tuple[np.ndarray, pd.DataFrame]:
    if "is_single_species" not in df.columns:
        return X, df
    flag = df["is_single_species"]
    if flag.dtype == object:
        flag = flag.astype(str).str.lower().isin(("true", "1", "yes"))
    else:
        flag = flag.astype(bool)

    if subset == "single":
        mask = flag.to_numpy()
    elif subset == "multi":
        mask = (~flag).to_numpy()
    else:
        mask = np.ones(len(df), dtype=bool)

    if mask.sum() == 0:
        raise RuntimeError(f"No rows left after subset={subset}")
    return X[mask], df.loc[mask].reset_index(drop=True)


def _plot_umap_2d(
    Z2: np.ndarray,
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skip plot {out_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    species = df["dominant_species"].astype(str).fillna("").to_numpy()
    sc0 = axes[0].scatter(Z2[:, 0], Z2[:, 1], c=cluster_ids, s=8, cmap="tab20", alpha=0.8)
    axes[0].set_title(f"{title} — cluster_id")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046)

    uniq = sorted({s for s in species if s})
    lut = {s: i for i, s in enumerate(uniq)}
    colors = np.array([lut.get(s, -1) for s in species], dtype=float)
    sc1 = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=colors, s=8, cmap="tab20", alpha=0.8)
    axes[1].set_title(f"{title} — dominant_species")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_sweep(
    name: str,
    X: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    *,
    subset_tag: str,
    random_state: int,
    k_list: Sequence[int],
) -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """
    Fit UMAP once, then MiniBatchKMeans for each k in k_list on Z.
    Returns (metric rows for all k, assignments at best-silhouette k).
    """
    y = df["dominant_species"]
    n = int(X.shape[0])
    if n < 3:
        raise RuntimeError(f"Need >=3 samples for UMAP+KMeans, got {n}")

    reducer, Z = fit_umap(X, random_state=random_state)
    save_umap(reducer, out_dir / "umap" / f"{name}_umap.joblib")
    np.save(out_dir / "umap" / f"{name}_{subset_tag}_Z.npy", Z)

    _, Z2 = fit_umap_2d(X, random_state=random_state)

    metric_rows: List[Dict] = []
    best_sil = -np.inf
    best_assign: Optional[pd.DataFrame] = None
    best_labels: Optional[np.ndarray] = None
    best_k: Optional[int] = None

    for k in k_list:
        if k < 2 or k >= n:
            print(f"  skip k={k} (need 2 <= k < n={n})")
            continue
        t0 = time.time()
        labels, inertia = run_minibatch_kmeans(Z, int(k), random_state=random_state)
        metrics = evaluate_clustering(Z, labels, y.to_numpy(), random_state=random_state)
        elapsed = time.time() - t0
        metrics["inertia"] = float(inertia)
        metrics["k"] = float(k)
        metrics["representation"] = name
        metrics["subset"] = subset_tag
        metrics["embedding_dim"] = float(X.shape[1])
        metrics["umap_dim"] = float(Z.shape[1])
        metrics["elapsed_s"] = float(elapsed)
        metrics["random_state"] = float(random_state)
        metric_rows.append(metrics)

        sil = metrics.get("silhouette", float("nan"))
        sil_f = float(sil) if sil == sil else -np.inf  # NaN-safe
        print(
            f"  k={k:3d}  sil={sil_f:.4f}  ari={metrics.get('ari', float('nan')):.4f}  "
            f"nmi={metrics.get('nmi', float('nan')):.4f}  ({elapsed:.2f}s)"
        )
        if sil_f >= best_sil:
            best_sil = sil_f
            best_k = int(k)
            best_labels = labels
            assign = df.copy()
            assign["cluster_id"] = labels
            assign["representation"] = name
            assign["subset"] = subset_tag
            assign["k"] = int(k)
            best_assign = assign

    if best_assign is not None and best_labels is not None and best_k is not None:
        _plot_umap_2d(
            Z2,
            best_assign,
            best_labels,
            out_dir
            / "results"
            / "umap_visualizations"
            / f"{name}_{subset_tag}_bestk{best_k}.png",
            title=f"{name} [{subset_tag}] best-k={best_k}",
        )
        # Also save a stable basename without k for convenience
        _plot_umap_2d(
            Z2,
            best_assign,
            best_labels,
            out_dir / "results" / "umap_visualizations" / f"{name}_{subset_tag}.png",
            title=f"{name} [{subset_tag}] best-k={best_k}",
        )

    return metric_rows, best_assign


def _gate_matched(out_dir: Path) -> None:
    matched_3s = out_dir / "metadata" / "matched_3s.csv"
    if matched_3s.is_file():
        df = pd.read_csv(matched_3s)
        a = df[df["model"].astype(str) == "a2v"].to_dict(orient="records")
        b = df[df["model"].astype(str) == "birdnet"].to_dict(orient="records")
        if a and b:
            assert_matched_keys(a, b, context="matched_3s")
            print("OK: matched_3s A2V/BirdNET keys agree")
    matched_5s = out_dir / "metadata" / "matched_5s.csv"
    if matched_5s.is_file():
        df = pd.read_csv(matched_5s)
        a = df[df["model"].astype(str) == "a2v"].to_dict(orient="records")
        b = df[df["model"].astype(str) == "perch"].to_dict(orient="records")
        if a and b:
            assert_matched_keys(a, b, context="matched_5s")
            print("OK: matched_5s A2V/Perch keys agree")


def _write_wide_csv(metrics_df: pd.DataFrame, path: Path) -> None:
    """One row per k; columns like a2v_3s_ARI, birdnet_3s_Silhouette, ..."""
    metric_map = [
        ("silhouette", "Silhouette"),
        ("ari", "ARI"),
        ("ami", "AMI"),
        ("nmi", "NMI"),
        ("purity", "Purity"),
        ("v_measure", "VMeasure"),
        ("inertia", "Inertia"),
    ]
    pieces = []
    for src, nice in metric_map:
        if src not in metrics_df.columns:
            continue
        p = metrics_df.pivot_table(
            index="k", columns="representation", values=src, aggfunc="first"
        )
        p = p.rename(columns={c: f"{c}_{nice}" for c in p.columns})
        pieces.append(p)
    if not pieces:
        return
    wide = pieces[0].join(pieces[1:], how="outer").sort_index().reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(path, index=False)
    print(f"Wrote {path}")


def _best_k_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rep, subset), g in metrics_df.groupby(["representation", "subset"]):
        g2 = g.copy()
        sil = pd.to_numeric(g2["silhouette"], errors="coerce")
        if sil.notna().any():
            idx = sil.idxmax()
            best = g2.loc[idx]
        else:
            best = g2.iloc[0]
        rows.append(
            {
                "representation": rep,
                "subset": subset,
                "best_k_by_silhouette": int(best["k"]),
                "silhouette": float(best.get("silhouette", np.nan)),
                "ari": float(best.get("ari", np.nan)),
                "ami": float(best.get("ami", np.nan)),
                "nmi": float(best.get("nmi", np.nan)),
                "purity": float(best.get("purity", np.nan)),
                "v_measure": float(best.get("v_measure", np.nan)),
                "n_points": float(best.get("n_points", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "umap").mkdir(parents=True, exist_ok=True)
    (out_dir / "results" / "umap_visualizations").mkdir(parents=True, exist_ok=True)

    _gate_matched(out_dir)

    k_list = _k_values(args.k_min, args.k_max, args.k_step, args.k)
    print(f"KMeans sweep k values: {k_list[0]}..{k_list[-1]} (n={len(k_list)})")

    subsets = ["single", "multi"] if args.subset == "all" else [args.subset]
    if "single" not in subsets:
        subsets = ["single"] + subsets

    metric_rows: List[Dict] = []
    assign_rows: List[pd.DataFrame] = []

    for name, emb_stem, meta_stem in REPRESENTATIONS:
        emb_path = out_dir / "embeddings" / f"{emb_stem}.npy"
        if not emb_path.is_file():
            print(f"SKIP {name}: missing {emb_path}")
            continue
        X_full, df_full = _load_pair(out_dir, emb_stem, meta_stem)
        for subset in subsets:
            try:
                X, df = _filter_subset(df_full, X_full, subset)
            except RuntimeError as e:
                print(f"SKIP {name}/{subset}: {e}")
                continue
            if X.shape[0] < 3:
                print(f"SKIP {name}/{subset}: n={X.shape[0]} < 3 (need UMAP+KMeans)")
                continue
            print(f"Evaluating {name} subset={subset} n={X.shape[0]} dim={X.shape[1]}")
            try:
                rows, assign = evaluate_sweep(
                    name,
                    X,
                    df,
                    out_dir,
                    subset_tag=subset,
                    random_state=args.random_state,
                    k_list=k_list,
                )
            except Exception as e:
                print(f"SKIP {name}/{subset}: {e}")
                continue
            metric_rows.extend(rows)
            if assign is not None:
                assign_rows.append(assign)

            # Checkpoint long CSV after each representation (like silhouette sweep)
            if metric_rows:
                pd.DataFrame(metric_rows).to_csv(
                    out_dir / "results" / "clustering_metrics.csv", index=False
                )

    if not metric_rows:
        print("ERROR: no representations evaluated", file=sys.stderr)
        return 1

    metrics_df = pd.DataFrame(metric_rows)
    front = [
        "representation",
        "subset",
        "k",
        "n_points",
        "n_labeled",
        "n_species",
        "n_clusters",
        "ari",
        "ami",
        "nmi",
        "purity",
        "homogeneity",
        "completeness",
        "v_measure",
        "silhouette",
        "inertia",
        "elapsed_s",
        "embedding_dim",
        "umap_dim",
        "random_state",
    ]
    cols = [c for c in front if c in metrics_df.columns] + [
        c for c in metrics_df.columns if c not in front
    ]
    metrics_df = metrics_df[cols].sort_values(["representation", "subset", "k"]).reset_index(
        drop=True
    )
    metrics_path = out_dir / "results" / "clustering_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    # Wide table for single-species primary (matches old sweep companion file)
    single = metrics_df[metrics_df["subset"] == "single"]
    if len(single):
        _write_wide_csv(single, out_dir / "results" / "clustering_metrics_wide.csv")
        summary = _best_k_summary(single)
        summary_path = out_dir / "results" / "best_k_by_silhouette.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
        print("\n=== Best k by silhouette (single-species) ===")
        print(summary.to_string(index=False))

        print("\n=== Matched comparisons at each model's best-k ===")
        for left, right in (("a2v_3s", "birdnet_3s"), ("a2v_5s", "perch_5s")):
            L = summary[summary["representation"] == left]
            R = summary[summary["representation"] == right]
            if len(L) and len(R):
                print(
                    f"\n{left} (k={int(L.iloc[0]['best_k_by_silhouette'])}) vs "
                    f"{right} (k={int(R.iloc[0]['best_k_by_silhouette'])}):"
                )
                for m in ("ari", "ami", "nmi", "purity", "v_measure", "silhouette"):
                    print(
                        f"  {m}: {float(L.iloc[0][m]):.4f} vs {float(R.iloc[0][m]):.4f}"
                    )

    if assign_rows:
        assigns = pd.concat(assign_rows, axis=0, ignore_index=True)
        assign_path = out_dir / "results" / "cluster_assignments.csv"
        drop_cols = [c for c in assigns.columns if c.startswith("umap_")]
        assigns.drop(columns=drop_cols, errors="ignore").to_csv(assign_path, index=False)
        print(f"Wrote {assign_path} (best-k assignments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
