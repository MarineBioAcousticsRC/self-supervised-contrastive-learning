"""MiniBatchKMeans on UMAP space + extended clustering metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from experiments.matched_window.config import (
    KMEANS_BATCH_SIZE,
    KMEANS_INIT,
    KMEANS_N_INIT,
    RANDOM_SEED,
)


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted average of per-cluster dominant-label fraction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    total = 0
    correct = 0
    for cid in np.unique(y_pred):
        mask = y_pred == cid
        n = int(mask.sum())
        if n == 0:
            continue
        labels, counts = np.unique(y_true[mask], return_counts=True)
        correct += int(counts.max())
        total += n
    return float(correct) / float(total) if total else float("nan")


def majority_vote_labels(
    frame_cluster_ids: Sequence[int],
    window_ids: Sequence[int],
) -> np.ndarray:
    """
    Majority-vote frame cluster IDs up to one label per window.

    ``window_ids[i]`` is the window index for frame i. Ties break to the
    smallest cluster id. Returns labels ordered by sorted unique window ids.
    """
    frame_cluster_ids = np.asarray(frame_cluster_ids)
    window_ids = np.asarray(window_ids)
    if frame_cluster_ids.shape[0] != window_ids.shape[0]:
        raise ValueError(
            f"Length mismatch: frames={frame_cluster_ids.shape[0]} "
            f"windows={window_ids.shape[0]}"
        )
    if frame_cluster_ids.size == 0:
        return np.asarray([], dtype=int)

    out: list[int] = []
    for wid in np.unique(window_ids):
        cids = frame_cluster_ids[window_ids == wid]
        vals, counts = np.unique(cids, return_counts=True)
        # Deterministic tie-break: highest count, then smallest cluster id
        order = np.lexsort((vals, -counts))
        out.append(int(vals[order[0]]))
    return np.asarray(out, dtype=int)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize so Euclidean K-Means matches cosine geometry."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(nrm, float(eps))).astype(np.float32)


def run_minibatch_kmeans(
    Z: np.ndarray,
    k: int,
    *,
    random_state: int = RANDOM_SEED,
):
    Z = np.asarray(Z, dtype=np.float32)
    k = int(k)
    if k <= 1 or k > Z.shape[0]:
        raise ValueError(f"Invalid k={k} for n={Z.shape[0]}")
    km = MiniBatchKMeans(
        n_clusters=k,
        init=KMEANS_INIT,
        random_state=int(random_state),
        n_init=KMEANS_N_INIT,
        batch_size=KMEANS_BATCH_SIZE,
    )
    labels = km.fit_predict(Z).astype(int)
    return labels, float(km.inertia_)


def evaluate_clustering(
    Z: Optional[np.ndarray],
    cluster_ids: np.ndarray,
    y_true: Optional[np.ndarray],
    *,
    max_silhouette_n: int = 5000,
    random_state: int = RANDOM_SEED,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    labels = np.asarray(cluster_ids)
    out["n_points"] = float(len(labels))
    out["n_clusters"] = float(len(np.unique(labels)))

    if Z is not None and len(labels) >= 3 and len(np.unique(labels)) >= 2:
        Xk = np.asarray(Z)
        yk = labels
        if Xk.shape[0] > max_silhouette_n:
            rng = np.random.default_rng(int(random_state))
            idx = rng.choice(Xk.shape[0], size=max_silhouette_n, replace=False)
            Xk, yk = Xk[idx], yk[idx]
        try:
            out["silhouette"] = float(silhouette_score(Xk, yk))
        except Exception:
            out["silhouette"] = float("nan")
    else:
        out["silhouette"] = float("nan")

    if y_true is not None and len(y_true) == len(labels):
        yt = np.asarray(y_true).astype(str)
        # Drop empty / unknown labels from supervised metrics
        keep = np.array(
            [t.strip() != "" and t.lower() not in ("unknown", "empty", "nan") for t in yt],
            dtype=bool,
        )
        if keep.sum() >= 2 and len(np.unique(yt[keep])) >= 2:
            yt_k = yt[keep]
            yp_k = labels[keep]
            out["ari"] = float(adjusted_rand_score(yt_k, yp_k))
            out["ami"] = float(adjusted_mutual_info_score(yt_k, yp_k))
            out["nmi"] = float(normalized_mutual_info_score(yt_k, yp_k))
            out["homogeneity"] = float(homogeneity_score(yt_k, yp_k))
            out["completeness"] = float(completeness_score(yt_k, yp_k))
            out["v_measure"] = float(v_measure_score(yt_k, yp_k))
            # also via joint call for consistency
            h, c, v = homogeneity_completeness_v_measure(yt_k, yp_k)
            out["homogeneity"] = float(h)
            out["completeness"] = float(c)
            out["v_measure"] = float(v)
            out["purity"] = float(cluster_purity(yt_k, yp_k))
            out["n_labeled"] = float(keep.sum())
            out["n_species"] = float(len(np.unique(yt_k)))
        else:
            for k in (
                "ari",
                "ami",
                "nmi",
                "homogeneity",
                "completeness",
                "v_measure",
                "purity",
            ):
                out[k] = float("nan")
            out["n_labeled"] = float(keep.sum())
            out["n_species"] = 0.0
    return out


def choose_k_from_labels(y_true: pd.Series) -> int:
    """Number of target species in evaluated subset."""
    yt = y_true.astype(str)
    keep = yt.map(
        lambda t: t.strip() != "" and t.lower() not in ("unknown", "empty", "nan")
    )
    n = int(yt[keep].nunique())
    return max(2, n)
