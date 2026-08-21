#!/usr/bin/env python3
"""
Voting-scheme evaluation for matched-window A2V dense frames.

Default protocol: cluster at **native** embedding dimension
(no UMAP) and, for Animal2Vec, only frames that overlap annotated vocalizations.
Majority-vote those frame cluster IDs to one label per N-s window, then score
extrinsic metrics on window labels. Does **not** flatten.
BirdNET / Perch remain one-embedding-per-window baselines (native 1024 / 1536).

  python -m experiments.matched_window.evaluate_vote \\
    --out-dir /path/to/data \\
    --cluster-space native --vocal-frames-only --results-subdir vote_native
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
    l2_normalize_rows,
    majority_vote_labels,
    run_minibatch_kmeans,
)
from experiments.matched_window.config import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DEFAULT_WAV_DIR,
    EXPECTED_CLIP_DURATION_S,
    KMEANS_BATCH_SIZE,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    KMEANS_K_STEP,
    RANDOM_SEED,
    UMAP_N_COMPONENTS,
)
from experiments.matched_window.evaluate import (  # noqa: E402
    _filter_subset,
    _gate_matched,
    _k_values,
    _load_pair,
    _plot_umap_2d,
)
from experiments.matched_window.labels import (  # noqa: E402
    frame_centers_seconds,
    frames_in_vocal_intervals,
    load_clip_intervals,
)
from experiments.matched_window.umap_fit import fit_umap, fit_umap_2d, save_umap  # noqa: E402

# A2V: dense frames + vote. Baselines: flat window embeddings (no vote).
A2V_VOTE_REPS = (
    ("a2v_1s_vote", "a2v_1s"),
    ("a2v_3s_vote", "a2v_3s"),
    ("a2v_5s_vote", "a2v_5s"),
)
BASELINE_REPS = (
    ("birdnet_3s", "birdnet_3s", "birdnet_3s"),
    ("perch_5s", "perch_5s", "perch_5s"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frame-cluster + majority-vote k-sweep (matched-window A2V)"
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    p.add_argument("--label-csv-dir", type=Path, default=None)
    p.add_argument("--random-state", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--subset",
        choices=["single", "multi", "all"],
        default="single",
        help="Primary eval uses single-species windows",
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
    p.add_argument(
        "--cluster-space",
        choices=["native", "umap"],
        default="native",
        help="native = K-Means on embedding dim (768/1024/1536); umap = reduce first",
    )
    p.add_argument(
        "--umap-dim",
        type=int,
        default=UMAP_N_COMPONENTS,
        help="UMAP n_components when --cluster-space umap (ignored for native)",
    )
    p.add_argument(
        "--l2-normalize",
        action="store_true",
        default=True,
        help="L2-normalize rows before native K-Means (cosine geometry)",
    )
    p.add_argument(
        "--no-l2-normalize",
        action="store_false",
        dest="l2_normalize",
    )
    p.add_argument(
        "--vocal-frames-only",
        action="store_true",
        default=True,
        help="Cluster only A2V frames that overlap annotated vocalizations",
    )
    p.add_argument(
        "--no-vocal-frames-only",
        action="store_false",
        dest="vocal_frames_only",
    )
    p.add_argument(
        "--results-subdir",
        default=None,
        help="Under results/. Default: vote_native if native, else vote",
    )
    return p.parse_args()


def _results_dir(out_dir: Path, subdir: str = "vote") -> Path:
    d = out_dir / "results" / str(subdir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "umap_visualizations").mkdir(parents=True, exist_ok=True)
    return d


def _load_frames_pair(
    out_dir: Path, stem: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    frames_path = out_dir / "embeddings" / f"{stem}_frames.npy"
    meta_path = out_dir / "metadata" / f"{stem}.csv"
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"Missing {frames_path} — re-run A2V embed to save dense frames"
        )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    X = np.load(frames_path)
    df = pd.read_csv(meta_path)
    if X.ndim != 3:
        raise RuntimeError(f"Expected (N, T, D) frames at {frames_path}, got {X.shape}")
    if len(df) != X.shape[0]:
        raise RuntimeError(
            f"Row mismatch for {stem}: meta={len(df)} frames_N={X.shape[0]}"
        )
    return X, df


def _filter_frames_subset(
    df: pd.DataFrame, frames: np.ndarray, subset: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Filter windows, keep matching (n_win, T, D) frame stacks."""
    if "is_single_species" not in df.columns:
        return frames, df
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
    return frames[mask], df.loc[mask].reset_index(drop=True)


def _expand_frames(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N, T, D) → (N*T, D) and window_id per frame in 0..N-1."""
    frames = np.asarray(frames, dtype=np.float32)
    n_win, t, d = frames.shape
    X = frames.reshape(n_win * t, d)
    window_ids = np.repeat(np.arange(n_win, dtype=np.int64), t)
    return X, window_ids


def _interval_cache_key(wav_path: Path) -> str:
    return str(Path(wav_path).resolve()) if Path(wav_path).exists() else str(wav_path)


def _load_intervals_cached(
    wav_path: Path,
    wav_root: Path,
    label_csv_dir: Optional[Path],
    cache: Dict[str, list],
) -> list:
    key = _interval_cache_key(wav_path)
    if key not in cache:
        intervals, _csv = load_clip_intervals(
            Path(wav_path),
            wav_root,
            label_csv_dir,
            clip_dur_s=EXPECTED_CLIP_DURATION_S,
        )
        cache[key] = list(intervals)
    return cache[key]


def vocal_frame_masks(
    df: pd.DataFrame,
    n_frames: int,
    *,
    wav_root: Path,
    label_csv_dir: Optional[Path],
) -> np.ndarray:
    """Boolean mask (N, T): True iff the frame overlaps an annotated vocalization."""
    n_win = len(df)
    t = int(n_frames)
    masks = np.zeros((n_win, t), dtype=bool)
    cache: Dict[str, list] = {}
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        wav_path = Path(str(row["wav_path"]))
        intervals = _load_intervals_cached(wav_path, wav_root, label_csv_dir, cache)
        ws = float(row["window_start_s"])
        we = float(row["window_end_s"])
        centers = frame_centers_seconds(ws, we, t)
        masks[int(i)] = frames_in_vocal_intervals(centers, intervals)
    return masks


def expand_vocal_frames(
    frames: np.ndarray,
    vocal_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only vocal frames. Returns (X, window_ids, keep_window_idx).

    ``keep_window_idx`` indexes the original window axis; windows with zero
    vocal frames are dropped. Remaining window_ids are 0..n_keep-1.
    """
    frames = np.asarray(frames, dtype=np.float32)
    vocal_mask = np.asarray(vocal_mask, dtype=bool)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (N,T,D), got {frames.shape}")
    if vocal_mask.shape != frames.shape[:2]:
        raise ValueError(
            f"mask shape {vocal_mask.shape} != frames {frames.shape[:2]}"
        )
    xs: List[np.ndarray] = []
    wids: List[np.ndarray] = []
    keep: List[int] = []
    new_i = 0
    for i in range(frames.shape[0]):
        m = vocal_mask[i]
        n_keep = int(m.sum())
        if n_keep == 0:
            continue
        xs.append(frames[i, m])
        wids.append(np.full(n_keep, new_i, dtype=np.int64))
        keep.append(i)
        new_i += 1
    if not xs:
        raise RuntimeError("No vocal frames left after annotation mask")
    X = np.concatenate(xs, axis=0)
    window_ids = np.concatenate(wids, axis=0)
    return X, window_ids, np.asarray(keep, dtype=np.int64)


def prepare_cluster_matrix(
    X: np.ndarray,
    *,
    cluster_space: str,
    umap_dim: int,
    l2_normalize: bool,
    random_state: int,
):
    """Return (reducer_or_None, Z, cluster_dim)."""
    X = np.asarray(X, dtype=np.float32)
    if l2_normalize:
        X = l2_normalize_rows(X)
    if cluster_space == "native":
        return None, X, int(X.shape[1])
    reducer, Z = fit_umap(
        X, n_components=int(umap_dim), random_state=random_state
    )
    return reducer, np.asarray(Z, dtype=np.float32), int(Z.shape[1])


def _write_wide_csv(metrics_df: pd.DataFrame, path: Path) -> None:
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


def _best_k_by_nmi(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rep, subset), g in metrics_df.groupby(["representation", "subset"]):
        g2 = g.copy()
        nmi = pd.to_numeric(g2["nmi"], errors="coerce")
        if nmi.notna().any():
            idx = nmi.idxmax()
            best = g2.loc[idx]
        else:
            best = g2.iloc[0]
        rows.append(
            {
                "representation": rep,
                "subset": subset,
                "best_k_by_nmi": int(best["k"]),
                "nmi": float(best.get("nmi", np.nan)),
                "ami": float(best.get("ami", np.nan)),
                "ari": float(best.get("ari", np.nan)),
                "purity": float(best.get("purity", np.nan)),
                "v_measure": float(best.get("v_measure", np.nan)),
                "silhouette": float(best.get("silhouette", np.nan)),
                "n_windows": float(best.get("n_windows", best.get("n_points", np.nan))),
                "n_frames": float(best.get("n_frames", np.nan)),
                "umap_dim": float(best.get("umap_dim", np.nan)),
                "cluster_space": best.get("cluster_space", ""),
                "cluster_dim": float(best.get("cluster_dim", np.nan)),
                "kmeans_batch_size": float(best.get("kmeans_batch_size", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_vote_sweep(
    name: str,
    frames: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    results_dir: Path,
    *,
    subset_tag: str,
    random_state: int,
    k_list: Sequence[int],
    cluster_space: str = "native",
    umap_dim: int = UMAP_N_COMPONENTS,
    l2_normalize: bool = True,
    vocal_mask: Optional[np.ndarray] = None,
    n_frames_before: Optional[int] = None,
) -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """
    KMeans on dense frames → majority vote → window extrinsic metrics.
    Default: native dim, vocal frames only. Best-k by NMI.
    """
    vocal_mask_kept = None
    if vocal_mask is not None:
        X, window_ids, keep_idx = expand_vocal_frames(frames, vocal_mask)
        vocal_mask_kept = np.asarray(vocal_mask)[keep_idx]
        df = df.iloc[keep_idx].reset_index(drop=True)
        frames = frames[keep_idx]
    else:
        X, window_ids = _expand_frames(frames)
    y_window = df["dominant_species"].astype(str).to_numpy()
    n_win = int(frames.shape[0])
    n_frames = int(X.shape[0])
    t_frames = int(frames.shape[1])
    emb_dim = int(frames.shape[2])
    n_before = int(n_frames_before) if n_frames_before is not None else n_frames

    if n_frames < 3:
        raise RuntimeError(f"Need >=3 frames for KMeans, got {n_frames}")

    print(
        f"  frames: n_windows={n_win} T={t_frames} D={emb_dim} "
        f"→ n_frames={n_frames} (from {n_before}) "
        f"cluster_space={cluster_space}"
    )

    reducer, Z, cluster_dim = prepare_cluster_matrix(
        X,
        cluster_space=cluster_space,
        umap_dim=umap_dim,
        l2_normalize=l2_normalize,
        random_state=random_state,
    )
    if reducer is not None:
        save_umap(reducer, out_dir / "umap" / f"{name}_umap.joblib")
    np.save(results_dir / "umap_visualizations" / f"{name}_{subset_tag}_Z.npy", Z)

    metric_rows: List[Dict] = []
    best_nmi = -np.inf
    best_assign: Optional[pd.DataFrame] = None
    best_k: Optional[int] = None

    for k in k_list:
        if k < 2 or k >= n_frames:
            print(f"  skip k={k} (need 2 <= k < n_frames={n_frames})")
            continue
        t0 = time.time()
        frame_labels, inertia = run_minibatch_kmeans(
            Z, int(k), random_state=random_state
        )
        voted = majority_vote_labels(frame_labels, window_ids)
        if voted.shape[0] != n_win:
            raise RuntimeError(
                f"Vote length {voted.shape[0]} != n_windows {n_win}"
            )

        frame_diag = evaluate_clustering(
            Z, frame_labels, None, random_state=random_state
        )
        win_metrics = evaluate_clustering(
            None, voted, y_window, random_state=random_state
        )
        elapsed = time.time() - t0

        metrics: Dict = dict(win_metrics)
        metrics["silhouette"] = float(frame_diag.get("silhouette", float("nan")))
        metrics["inertia"] = float(inertia)
        metrics["k"] = float(k)
        metrics["representation"] = name
        metrics["subset"] = subset_tag
        metrics["n_windows"] = float(n_win)
        metrics["n_frames"] = float(n_frames)
        metrics["n_points"] = float(n_win)  # extrinsic scored at window level
        metrics["frames_per_window"] = float(t_frames)
        metrics["embedding_dim"] = float(emb_dim)
        metrics["umap_dim"] = (
            float("nan") if cluster_space == "native" else float(Z.shape[1])
        )
        metrics["cluster_space"] = cluster_space
        metrics["cluster_dim"] = float(cluster_dim)
        metrics["l2_normalize"] = float(bool(l2_normalize))
        metrics["vocal_frames_only"] = float(vocal_mask is not None)
        metrics["n_frames_before"] = float(n_before)
        metrics["kmeans_batch_size"] = float(KMEANS_BATCH_SIZE)
        metrics["elapsed_s"] = float(elapsed)
        metrics["random_state"] = float(random_state)
        metrics["aggregation"] = "majority_vote"
        metric_rows.append(metrics)

        nmi_f = float(metrics.get("nmi", float("nan")))
        nmi_safe = nmi_f if nmi_f == nmi_f else -np.inf
        print(
            f"  k={k:3d}  nmi={nmi_f:.4f}  ami={metrics.get('ami', float('nan')):.4f}  "
            f"ari={metrics.get('ari', float('nan')):.4f}  "
            f"frame_sil={metrics.get('silhouette', float('nan')):.4f}  ({elapsed:.2f}s)"
        )
        if nmi_safe >= best_nmi:
            best_nmi = nmi_safe
            best_k = int(k)
            assign = df.copy()
            assign["cluster_id"] = voted
            assign["representation"] = name
            assign["subset"] = subset_tag
            assign["k"] = int(k)
            assign["aggregation"] = "majority_vote"
            best_assign = assign

    if best_assign is not None and best_k is not None:
        # Window-level 2D UMAP for viz only (not used for clustering).
        if vocal_mask_kept is not None:
            mean_X = np.stack(
                [frames[i, vocal_mask_kept[i]].mean(axis=0) for i in range(n_win)]
            )
        else:
            mean_X = frames.mean(axis=1)
        _, Z2 = fit_umap_2d(mean_X, random_state=random_state)
        _plot_umap_2d(
            Z2,
            best_assign,
            best_assign["cluster_id"].to_numpy(),
            results_dir
            / "umap_visualizations"
            / f"{name}_{subset_tag}_bestk{best_k}.png",
            title=f"{name} [{subset_tag}] vote best-NMI k={best_k}",
        )

    return metric_rows, best_assign


def evaluate_baseline_sweep(
    name: str,
    X: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    results_dir: Path,
    *,
    subset_tag: str,
    random_state: int,
    k_list: Sequence[int],
    cluster_space: str = "native",
    umap_dim: int = UMAP_N_COMPONENTS,
    l2_normalize: bool = True,
) -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """One embedding per window (BirdNET/Perch); best-k by NMI."""
    y = df["dominant_species"]
    n = int(X.shape[0])
    if n < 3:
        raise RuntimeError(f"Need >=3 samples for KMeans, got {n}")

    reducer, Z, cluster_dim = prepare_cluster_matrix(
        X,
        cluster_space=cluster_space,
        umap_dim=umap_dim,
        l2_normalize=l2_normalize,
        random_state=random_state,
    )
    if reducer is not None:
        save_umap(reducer, out_dir / "umap" / f"{name}_vote_baseline_umap.joblib")
    np.save(results_dir / "umap_visualizations" / f"{name}_vote_{subset_tag}_Z.npy", Z)
    _, Z2 = fit_umap_2d(X, random_state=random_state)

    metric_rows: List[Dict] = []
    best_nmi = -np.inf
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
        metrics["n_windows"] = float(n)
        metrics["n_frames"] = float("nan")
        metrics["embedding_dim"] = float(X.shape[1])
        metrics["umap_dim"] = (
            float("nan") if cluster_space == "native" else float(Z.shape[1])
        )
        metrics["cluster_space"] = cluster_space
        metrics["cluster_dim"] = float(cluster_dim)
        metrics["l2_normalize"] = float(bool(l2_normalize))
        metrics["vocal_frames_only"] = 0.0
        metrics["kmeans_batch_size"] = float(KMEANS_BATCH_SIZE)
        metrics["elapsed_s"] = float(elapsed)
        metrics["random_state"] = float(random_state)
        metrics["aggregation"] = "window_embedding"
        metric_rows.append(metrics)

        nmi_f = float(metrics.get("nmi", float("nan")))
        nmi_safe = nmi_f if nmi_f == nmi_f else -np.inf
        print(
            f"  k={k:3d}  nmi={nmi_f:.4f}  ami={metrics.get('ami', float('nan')):.4f}  "
            f"ari={metrics.get('ari', float('nan')):.4f}  ({elapsed:.2f}s)"
        )
        if nmi_safe >= best_nmi:
            best_nmi = nmi_safe
            best_k = int(k)
            best_labels = labels
            assign = df.copy()
            assign["cluster_id"] = labels
            assign["representation"] = name
            assign["subset"] = subset_tag
            assign["k"] = int(k)
            assign["aggregation"] = "window_embedding"
            best_assign = assign

    if best_assign is not None and best_labels is not None and best_k is not None:
        _plot_umap_2d(
            Z2,
            best_assign,
            best_labels,
            results_dir
            / "umap_visualizations"
            / f"{name}_{subset_tag}_bestk{best_k}.png",
            title=f"{name} [{subset_tag}] best-NMI k={best_k}",
        )

    return metric_rows, best_assign


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_root = Path(args.wav_dir).expanduser().resolve()
    label_csv_dir = (
        Path(args.label_csv_dir).expanduser().resolve()
        if args.label_csv_dir is not None
        else None
    )
    (out_dir / "umap").mkdir(parents=True, exist_ok=True)
    subdir = args.results_subdir
    if not subdir:
        subdir = "vote_native" if args.cluster_space == "native" else "vote"
    results_dir = _results_dir(out_dir, subdir)

    _gate_matched(out_dir)

    k_list = _k_values(args.k_min, args.k_max, args.k_step, args.k)
    print(f"KMeans sweep k values: {k_list[0]}..{k_list[-1]} (n={len(k_list)})")
    print(
        f"cluster_space={args.cluster_space} "
        f"umap_dim={args.umap_dim if args.cluster_space == 'umap' else 'n/a'} "
        f"l2_normalize={args.l2_normalize} "
        f"vocal_frames_only={args.vocal_frames_only} "
        f"MiniBatchKMeans batch_size={KMEANS_BATCH_SIZE}"
    )
    print(f"results_dir={results_dir}")

    subsets = ["single", "multi"] if args.subset == "all" else [args.subset]
    if "single" not in subsets:
        subsets = ["single"] + subsets

    metric_rows: List[Dict] = []
    assign_rows: List[pd.DataFrame] = []

    for name, stem in A2V_VOTE_REPS:
        frames_path = out_dir / "embeddings" / f"{stem}_frames.npy"
        if not frames_path.is_file():
            print(f"SKIP {name}: missing {frames_path}")
            continue
        frames_full, df_full = _load_frames_pair(out_dir, stem)
        for subset in subsets:
            try:
                frames, df = _filter_frames_subset(df_full, frames_full, subset)
            except RuntimeError as e:
                print(f"SKIP {name}/{subset}: {e}")
                continue
            if frames.shape[0] < 1:
                print(f"SKIP {name}/{subset}: no windows")
                continue
            n_before = int(frames.shape[0] * frames.shape[1])
            vocal_mask = None
            if args.vocal_frames_only:
                print(f"  building vocal-frame mask from {wav_root} ...")
                vocal_mask = vocal_frame_masks(
                    df,
                    int(frames.shape[1]),
                    wav_root=wav_root,
                    label_csv_dir=label_csv_dir,
                )
                n_vocal = int(vocal_mask.sum())
                n_empty = int((~vocal_mask.any(axis=1)).sum())
                print(
                    f"  vocal frames {n_vocal}/{n_before} "
                    f"({100.0 * n_vocal / max(n_before, 1):.1f}%), "
                    f"windows with no vocal frames: {n_empty}"
                )
            print(
                f"Evaluating {name} subset={subset} "
                f"n_windows={frames.shape[0]} T={frames.shape[1]} D={frames.shape[2]}"
            )
            try:
                rows, assign = evaluate_vote_sweep(
                    name,
                    frames,
                    df,
                    out_dir,
                    results_dir,
                    subset_tag=subset,
                    random_state=args.random_state,
                    k_list=k_list,
                    cluster_space=args.cluster_space,
                    umap_dim=args.umap_dim,
                    l2_normalize=args.l2_normalize,
                    vocal_mask=vocal_mask,
                    n_frames_before=n_before,
                )
            except Exception as e:
                print(f"SKIP {name}/{subset}: {e}")
                continue
            metric_rows.extend(rows)
            if assign is not None:
                assign_rows.append(assign)
            if metric_rows:
                pd.DataFrame(metric_rows).to_csv(
                    results_dir / "clustering_metrics.csv", index=False
                )

    for name, emb_stem, meta_stem in BASELINE_REPS:
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
                print(f"SKIP {name}/{subset}: n={X.shape[0]} < 3")
                continue
            print(
                f"Evaluating {name} subset={subset} n={X.shape[0]} "
                f"dim={X.shape[1]} cluster_space={args.cluster_space}"
            )
            try:
                rows, assign = evaluate_baseline_sweep(
                    name,
                    X,
                    df,
                    out_dir,
                    results_dir,
                    subset_tag=subset,
                    random_state=args.random_state,
                    k_list=k_list,
                    cluster_space=args.cluster_space,
                    umap_dim=args.umap_dim,
                    l2_normalize=args.l2_normalize,
                )
            except Exception as e:
                print(f"SKIP {name}/{subset}: {e}")
                continue
            metric_rows.extend(rows)
            if assign is not None:
                assign_rows.append(assign)
            if metric_rows:
                pd.DataFrame(metric_rows).to_csv(
                    results_dir / "clustering_metrics.csv", index=False
                )

    if not metric_rows:
        print("ERROR: no representations evaluated", file=sys.stderr)
        return 1

    metrics_df = pd.DataFrame(metric_rows)
    front = [
        "representation",
        "subset",
        "k",
        "aggregation",
        "n_windows",
        "n_frames",
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
        "cluster_space",
        "cluster_dim",
        "umap_dim",
        "l2_normalize",
        "vocal_frames_only",
        "n_frames_before",
        "kmeans_batch_size",
        "frames_per_window",
        "random_state",
    ]
    cols = [c for c in front if c in metrics_df.columns] + [
        c for c in metrics_df.columns if c not in front
    ]
    metrics_df = metrics_df[cols].sort_values(
        ["representation", "subset", "k"]
    ).reset_index(drop=True)
    metrics_path = results_dir / "clustering_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    single = metrics_df[metrics_df["subset"] == "single"]
    if len(single):
        _write_wide_csv(single, results_dir / "clustering_metrics_wide.csv")
        summary = _best_k_by_nmi(single)
        summary_path = results_dir / "best_k_by_nmi.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
        print("\n=== Best k by NMI (single-species) ===")
        print(summary.to_string(index=False))

        print("\n=== Matched comparisons at each model's best-NMI k ===")
        for left, right in (
            ("a2v_3s_vote", "birdnet_3s"),
            ("a2v_5s_vote", "perch_5s"),
        ):
            L = summary[summary["representation"] == left]
            R = summary[summary["representation"] == right]
            if len(L) and len(R):
                print(
                    f"\n{left} (k={int(L.iloc[0]['best_k_by_nmi'])}) vs "
                    f"{right} (k={int(R.iloc[0]['best_k_by_nmi'])}):"
                )
                for m in ("nmi", "ami", "ari", "purity", "v_measure", "silhouette"):
                    print(
                        f"  {m}: {float(L.iloc[0][m]):.4f} vs {float(R.iloc[0][m]):.4f}"
                    )

    if assign_rows:
        assigns = pd.concat(assign_rows, axis=0, ignore_index=True)
        assign_path = results_dir / "cluster_assignments_windows.csv"
        drop_cols = [c for c in assigns.columns if c.startswith("umap_")]
        assigns.drop(columns=drop_cols, errors="ignore").to_csv(assign_path, index=False)
        print(f"Wrote {assign_path} (best-NMI window assignments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
