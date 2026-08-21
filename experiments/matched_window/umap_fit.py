"""Separate UMAP fits per representation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from experiments.matched_window.config import (
    RANDOM_SEED,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)


def fit_umap(
    X: np.ndarray,
    *,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = UMAP_METRIC,
    random_state: int = RANDOM_SEED,
):
    try:
        import umap
    except ImportError as e:
        raise SystemExit(
            "umap-learn is required. pip install umap-learn\n" f"Original error: {e}"
        ) from e

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n = int(X.shape[0])
    if n < 3:
        raise ValueError(f"Need at least 3 samples for UMAP, got {n}")

    # Keep UMAP internals valid for small smoke runs.
    nn = int(min(n_neighbors, max(2, n - 1)))
    n_comp = int(min(n_components, max(2, n - 2)))

    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=nn,
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(random_state),
        init="random",  # more stable than spectral on tiny N
    )
    try:
        Z = reducer.fit_transform(X)
    except Exception as e:
        # Last-resort fallback for very small N / sparse spectral failures.
        from sklearn.decomposition import PCA

        print(f"UMAP failed ({e}); falling back to PCA n_components={n_comp}")
        pca = PCA(n_components=n_comp, random_state=int(random_state))
        Z = pca.fit_transform(X)
        reducer = pca
    return reducer, np.asarray(Z, dtype=np.float32)


def fit_umap_2d(X: np.ndarray, **kwargs) -> Tuple[object, np.ndarray]:
    kwargs = dict(kwargs)
    kwargs["n_components"] = 2
    return fit_umap(X, **kwargs)


def save_umap(reducer, path: Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, path)


def load_umap(path: Path):
    import joblib

    return joblib.load(path)
