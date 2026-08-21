#!/usr/bin/env python3
"""
Vote concentration within each N-s window.

At best-NMI k (from results/vote/best_k_by_nmi.csv), cluster dense frames,
then for each window report:
  - majority_frac: fraction of frames assigned to the majority cluster
  - entropy_bits: empirical Shannon entropy of cluster IDs with p_c = n_c / N
    (each of the N frame embeddings contributes weight 1/N)

  python -m experiments.matched_window.vote_concentration \\
    --out-dir /path/to/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.matched_window.cluster_eval import run_minibatch_kmeans  # noqa: E402
from experiments.matched_window.config import DEFAULT_OUT_DIR, RANDOM_SEED  # noqa: E402
from experiments.matched_window.evaluate_vote import (  # noqa: E402
    A2V_VOTE_REPS,
    _expand_frames,
    _filter_frames_subset,
    _load_frames_pair,
    _results_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--subset", default="single", choices=["single", "multi", "all"])
    p.add_argument("--random-state", type=int, default=RANDOM_SEED)
    return p.parse_args()


def window_vote_stats(frame_cluster_ids: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Returns (majority_frac, entropy_bits, majority_cluster_id, n_unique_clusters).

    Entropy uses empirical p_c = n_c / N (each embedding weight 1/N).
    """
    ids = np.asarray(frame_cluster_ids)
    n = int(ids.size)
    if n == 0:
        return float("nan"), float("nan"), -1, 0
    vals, counts = np.unique(ids, return_counts=True)
    majority_frac = float(counts.max()) / float(n)
    maj_id = int(vals[int(np.argmax(counts))])
    # Tie-break majority id to smallest cluster id among max-count (match vote helper)
    max_c = int(counts.max())
    maj_id = int(np.min(vals[counts == max_c]))
    p = counts.astype(np.float64) / float(n)
    # natural bits via log2; ignore zero (none present after unique)
    entropy = float(-np.sum(p * np.log2(p)))
    return majority_frac, entropy, maj_id, int(vals.size)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    results_dir = _results_dir(out_dir)
    best_path = results_dir / "best_k_by_nmi.csv"
    if not best_path.is_file():
        print(f"ERROR: missing {best_path}", file=sys.stderr)
        return 1
    best = pd.read_csv(best_path)
    best = best[best["subset"].astype(str) == str(args.subset)]

    per_window_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for name, stem in A2V_VOTE_REPS:
        row = best[best["representation"] == name]
        if row.empty:
            print(f"SKIP {name}: no best_k_by_nmi row")
            continue
        k = int(row.iloc[0]["best_k_by_nmi"])

        z_path = out_dir / "umap" / f"{name}_{args.subset}_Z.npy"
        frames_full, df_full = _load_frames_pair(out_dir, stem)
        frames, df = _filter_frames_subset(df_full, frames_full, args.subset)
        X, window_ids = _expand_frames(frames)
        n_win = int(frames.shape[0])
        t = int(frames.shape[1])

        if z_path.is_file():
            Z = np.load(z_path)
            if Z.shape[0] != X.shape[0]:
                raise RuntimeError(
                    f"{name}: Z rows {Z.shape[0]} != frames {X.shape[0]}"
                )
            print(f"{name}: loaded Z {Z.shape}, k={k}")
        else:
            raise FileNotFoundError(
                f"Missing {z_path}; re-run evaluate_vote first"
            )

        labels, _ = run_minibatch_kmeans(Z, k, random_state=args.random_state)

        maj_fracs = []
        ents = []
        for wid in range(n_win):
            cids = labels[window_ids == wid]
            maj_frac, ent, maj_id, n_uniq = window_vote_stats(cids)
            maj_fracs.append(maj_frac)
            ents.append(ent)
            meta = df.iloc[wid]
            per_window_rows.append(
                {
                    "representation": name,
                    "subset": args.subset,
                    "k": k,
                    "window_idx": wid,
                    "sample_key": meta.get("sample_key", ""),
                    "dominant_species": meta.get("dominant_species", ""),
                    "n_frames": t,
                    "majority_cluster_id": maj_id,
                    "n_unique_clusters_in_window": n_uniq,
                    "majority_frac": maj_frac,
                    "entropy_bits": ent,
                }
            )

        maj_arr = np.asarray(maj_fracs, dtype=float)
        ent_arr = np.asarray(ents, dtype=float)
        summary_rows.append(
            {
                "representation": name,
                "subset": args.subset,
                "k": k,
                "n_windows": n_win,
                "frames_per_window": t,
                "majority_frac_mean": float(np.mean(maj_arr)),
                "majority_frac_median": float(np.median(maj_arr)),
                "majority_frac_std": float(np.std(maj_arr)),
                "majority_frac_p25": float(np.percentile(maj_arr, 25)),
                "majority_frac_p75": float(np.percentile(maj_arr, 75)),
                "entropy_bits_mean": float(np.mean(ent_arr)),
                "entropy_bits_median": float(np.median(ent_arr)),
                "entropy_bits_std": float(np.std(ent_arr)),
                "entropy_bits_p25": float(np.percentile(ent_arr, 25)),
                "entropy_bits_p75": float(np.percentile(ent_arr, 75)),
                # high concentration: majority >= 0.5 / 0.75
                "frac_windows_majority_ge_0.5": float(np.mean(maj_arr >= 0.5)),
                "frac_windows_majority_ge_0.75": float(np.mean(maj_arr >= 0.75)),
            }
        )
        print(
            f"{name} k={k}: majority_frac mean={np.mean(maj_arr):.3f} "
            f"median={np.median(maj_arr):.3f}; "
            f"entropy_bits mean={np.mean(ent_arr):.3f} median={np.median(ent_arr):.3f}"
        )

    if not summary_rows:
        print("ERROR: nothing computed", file=sys.stderr)
        return 1

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(per_window_rows)
    summary_path = results_dir / "vote_concentration_summary.csv"
    detail_path = results_dir / "vote_concentration_by_window.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print("\n=== Vote concentration summary ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
