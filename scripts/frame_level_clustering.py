#!/usr/bin/env python3
"""
Frame-level embedding clustering with MiniBatch k-means (or HDBSCAN).

Loads .h5 embeddings, optionally assigns CSV ground-truth labels, clusters with
K-Means or HDBSCAN, and writes metrics (NMI, ARI, AMI, RI, silhouette).
"""

from __future__ import annotations

import os

# Must be set before numpy/sklearn import (OpenBLAS thread oversubscription segfaults).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
)

try:
    import hdbscan  # type: ignore

    _HAS_HDBSCAN = True
except ImportError:
    hdbscan = None
    _HAS_HDBSCAN = False

_SEGMENT_WINDOW_RE = re.compile(r"_(\d{5})s_(\d{5})s$")


def load_h5_embeddings_for_clustering(emb_dir: str) -> pd.DataFrame:
    emb_dir_p = Path(emb_dir.strip())
    if not emb_dir_p.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    h5_paths = sorted(emb_dir_p.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in: {emb_dir}")

    rows: List[Dict] = []
    for h5_path in h5_paths:
        try:
            with h5py.File(h5_path, "r") as f:
                if "embedding" not in f or "time" not in f:
                    continue
                emb = np.asarray(f["embedding"][:], dtype=np.float32)
                t = np.asarray(f["time"][:], dtype=np.float32)
                if emb.ndim != 2 or t.ndim != 1 or emb.shape[0] != t.shape[0]:
                    continue

                wav_name = None
                if "filename" in f:
                    try:
                        wav_name = f["filename"][()]
                        if isinstance(wav_name, (bytes, np.bytes_)):
                            wav_name = wav_name.decode("utf-8", errors="ignore")
                        wav_name = str(wav_name)
                    except Exception:
                        wav_name = None

                if not wav_name:
                    wav_name = h5_path.name.split("_embeddings_")[0]

                if len(t) >= 2:
                    dt = float(np.median(np.diff(t)))
                    dt = dt if np.isfinite(dt) and dt > 0 else 0.0
                else:
                    dt = 0.0
                dt = max(dt, 1e-3)

                for i in range(emb.shape[0]):
                    ti = float(t[i])
                    rows.append(
                        {
                            "wav": os.path.basename(wav_name),
                            "start_s": ti,
                            "end_s": ti + dt,
                            "time_s": ti,
                            "h5_path": str(h5_path),
                            "h5_name": h5_path.name,
                            "frame_idx": int(i),
                            "embedding_vec": emb[i],
                        }
                    )
        except Exception:
            continue

    if not rows:
        raise ValueError(
            "No valid (embedding,time) pairs found in .h5 files. "
            "Expected datasets: 'embedding' (T,D) and 'time' (T,)."
        )

    return pd.DataFrame(rows)


def recording_stem_from_segment_h5_name(h5_name: str) -> Optional[str]:
    name = Path(h5_name).name
    if ".wav_embeddings_" not in name:
        return None
    left = name.split(".wav_embeddings_", 1)[0]
    left = Path(left).stem
    if not _SEGMENT_WINDOW_RE.search(left):
        return None
    return _SEGMENT_WINDOW_RE.sub("", left)


def wav_stem_from_h5_name(h5_name: str) -> str:
    name = Path(h5_name).name
    if ".wav_embeddings_" in name:
        left = name.split(".wav_embeddings_", 1)[0]
    else:
        left = name.split("_embeddings_", 1)[0]
    return Path(left).stem


def resolve_csv_path(
    wav_csv_dir: Path,
    *,
    h5_name: str,
    wav_name: str,
    segment_h5_use_recording_csv: bool,
) -> Tuple[Optional[Path], str, str]:
    stem_candidates: List[str] = []
    h5_stem = wav_stem_from_h5_name(h5_name)
    if h5_stem:
        stem_candidates.append(h5_stem)
    if wav_name:
        wav_stem = Path(wav_name).stem
        if wav_stem and wav_stem not in stem_candidates:
            stem_candidates.append(wav_stem)

    for stem in stem_candidates:
        segment_csv = wav_csv_dir / f"{stem}.csv"
        if segment_csv.exists():
            return segment_csv, stem, "file_csv"

    if segment_h5_use_recording_csv or ".wav_embeddings_" in h5_name:
        rec_stem = recording_stem_from_segment_h5_name(h5_name)
        if rec_stem is not None:
            recording_csv = wav_csv_dir / f"{rec_stem}.csv"
            if recording_csv.exists():
                return recording_csv, rec_stem, "recording_csv"

    return None, (stem_candidates[0] if stem_candidates else ""), "none"


def read_label_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        try:
            df = pd.read_csv(csv_path, header=None, sep=r"\s+")
        except Exception:
            return None
    if df.shape[0] == 0 or df.shape[1] < 3:
        return None
    df = df.iloc[:, :3].copy()
    df.columns = ["start", "duration", "cls"]
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["cls"] = df["cls"].astype(str).str.strip()
    df = df.dropna(subset=["start", "duration", "cls"])
    df = df[df["duration"] > 0]
    return df if len(df) > 0 else None


def filter_annotation_segments(
    df: pd.DataFrame, *, only_call: bool, max_segment_duration: Optional[float]
) -> pd.DataFrame:
    out = df.copy()
    if only_call:
        c = out["cls"].str.lower()
        out = out[c.str.contains("call", na=False) & ~c.str.contains("unknown", na=False)]
    if max_segment_duration is not None:
        out = out[out["duration"] <= float(max_segment_duration)]
    return out


def assign_frames_to_segments(
    time_sec: np.ndarray,
    segments: pd.DataFrame,
    recording_id: str,
) -> pd.DataFrame:
    n = len(time_sec)
    out = pd.DataFrame(
        {
            "time_s": time_sec.astype(np.float32),
            "class_name": ["Unknown"] * n,
            "segment_start_s": np.full(n, np.nan, dtype=np.float32),
            "segment_end_s": np.full(n, np.nan, dtype=np.float32),
            "segment_uid": [None] * n,
        }
    )

    for seg_idx, row in segments.reset_index(drop=True).iterrows():
        start_s = float(row["start"])
        end_s = float(row["start"]) + float(row["duration"])
        cls = str(row["cls"]).strip()

        left = int(np.searchsorted(time_sec, start_s, side="left"))
        right = int(np.searchsorted(time_sec, end_s, side="right"))
        left = max(0, min(left, n))
        right = max(left, min(right, n))
        if right <= left:
            continue

        seg_uid = f"{recording_id}__{start_s:.5f}__{end_s:.5f}__{seg_idx}"
        current = out.iloc[left:right]["class_name"].to_numpy()
        mask = current == "Unknown"
        idx = np.arange(left, right)[mask]
        if len(idx) == 0:
            continue

        out.loc[idx, "class_name"] = cls
        out.loc[idx, "segment_start_s"] = start_s
        out.loc[idx, "segment_end_s"] = end_s
        out.loc[idx, "segment_uid"] = seg_uid

    return out


def filter_by_min_label_count(df: pd.DataFrame, min_label_count: int) -> pd.DataFrame:
    vals, counts = np.unique(df["class_name"].astype(str).to_numpy(), return_counts=True)
    keep = set(vals[counts >= int(min_label_count)].tolist())
    return df[df["class_name"].astype(str).isin(keep)].copy()


def subsample_max_per_label(df: pd.DataFrame, max_samples_per_label: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    keep_idx: List[int] = []
    y = df["class_name"].astype(str).to_numpy()
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) <= int(max_samples_per_label):
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=int(max_samples_per_label), replace=False).tolist())
    keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
    return df.iloc[keep_idx].copy()


def maybe_add_labels_from_csv(
    df_frames: pd.DataFrame,
    *,
    wav_csv_dir: str,
    segment_h5_use_recording_csv: bool,
    ignore_test_files: bool,
    only_call: bool,
    max_segment_duration: Optional[float],
) -> pd.DataFrame:
    wav_csv_dir_p = Path(wav_csv_dir)
    if not wav_csv_dir_p.exists():
        raise FileNotFoundError(f"WAV/CSV directory not found: {wav_csv_dir}")

    out_parts: List[pd.DataFrame] = []
    for _, g in df_frames.groupby("h5_path", sort=False):
        g = g.sort_values("frame_idx")
        h5_name = str(g["h5_name"].iloc[0])
        wav_name = str(g["wav"].iloc[0]).strip()

        csv_path, recording_id, _match_mode = resolve_csv_path(
            wav_csv_dir_p,
            h5_name=h5_name,
            wav_name=wav_name,
            segment_h5_use_recording_csv=segment_h5_use_recording_csv,
        )

        if ignore_test_files and (
            "testfile" in recording_id.lower() or "testfile" in wav_name.lower()
        ):
            continue

        if csv_path is None or not csv_path.exists():
            continue

        df_seg = read_label_csv(csv_path)
        if df_seg is None:
            continue

        df_seg = filter_annotation_segments(
            df_seg,
            only_call=only_call,
            max_segment_duration=max_segment_duration,
        )
        if len(df_seg) == 0:
            continue

        t = g["time_s"].to_numpy(dtype=np.float32)
        frame_seg_df = assign_frames_to_segments(t, df_seg, recording_id=recording_id)

        g2 = g.copy().reset_index(drop=True)
        g2["class_name"] = frame_seg_df["class_name"].astype(str)
        g2["segment_start_s"] = frame_seg_df["segment_start_s"]
        g2["segment_end_s"] = frame_seg_df["segment_end_s"]
        g2["segment_uid"] = frame_seg_df["segment_uid"]

        labeled_count = int((g2["class_name"].astype(str).str.lower() != "unknown").sum())
        if labeled_count == 0:
            continue

        out_parts.append(g2)

    if not out_parts:
        raise ValueError("No frames could be labeled from CSVs.")

    return pd.concat(out_parts, axis=0, ignore_index=True)


def cluster_embeddings_df(
    df_frames: pd.DataFrame,
    algorithm: str,
    random_state: int,
    k: int = 60,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> pd.DataFrame:
    X = np.stack(df_frames["embedding_vec"].to_numpy())

    if algorithm.lower() in {"kmeans", "k-means", "k_means"}:
        if k <= 1 or k > X.shape[0]:
            raise ValueError(f"Invalid k={k} for n_frames={X.shape[0]}")
        km = MiniBatchKMeans(
            n_clusters=int(k),
            init="k-means++",
            random_state=int(random_state),
            n_init=10,
            batch_size=128,
        )
        labels = km.fit_predict(X).astype(int)
        extra = {"kmeans_inertia": float(km.inertia_)}
    elif algorithm.lower() in {"hdbscan", "hdb"}:
        if not _HAS_HDBSCAN or hdbscan is None:
            raise ImportError("HDBSCAN not installed. pip install hdbscan")
        ms = None if (min_samples is None or int(min_samples) <= 0) else int(min_samples)
        cl = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=ms,
            metric=str(metric),
            cluster_selection_method=str(cluster_selection_method),
        )
        labels = cl.fit_predict(X).astype(int)
        extra = {"hdbscan_noise_frac": float(np.mean(labels == -1))}
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    out = df_frames.copy()
    out["cluster_id"] = labels
    if "class_name" not in out.columns:
        out["class_name"] = "Unknown"
    out.attrs["cluster_metrics"] = extra
    return out


def summarize_clustering(
    df_results: pd.DataFrame,
    *,
    X_for_silhouette: Optional[np.ndarray] = None,
    max_silhouette_n: int = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    labels = df_results["cluster_id"].to_numpy()
    out["n_points"] = float(len(labels))
    out["n_clusters_including_noise"] = float(len(np.unique(labels)))
    out["n_noise"] = float(np.sum(labels == -1))

    if X_for_silhouette is not None and len(labels) >= 3:
        keep = labels != -1
        Xk = X_for_silhouette[keep]
        yk = labels[keep]
        if len(np.unique(yk)) >= 2 and Xk.shape[0] >= 10:
            if Xk.shape[0] > max_silhouette_n:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(Xk.shape[0], size=max_silhouette_n, replace=False)
                Xk, yk = Xk[idx], yk[idx]
            try:
                out["silhouette"] = float(silhouette_score(Xk, yk))
            except Exception:
                out["silhouette"] = float("nan")
            try:
                out["calinski_harabasz"] = float(calinski_harabasz_score(Xk, yk))
            except Exception:
                out["calinski_harabasz"] = float("nan")
            try:
                out["davies_bouldin"] = float(davies_bouldin_score(Xk, yk))
            except Exception:
                out["davies_bouldin"] = float("nan")
        else:
            out["silhouette"] = float("nan")
            out["calinski_harabasz"] = float("nan")
            out["davies_bouldin"] = float("nan")
    else:
        out["silhouette"] = float("nan")
        out["calinski_harabasz"] = float("nan")
        out["davies_bouldin"] = float("nan")

    if "class_name" in df_results.columns:
        y_true = df_results["class_name"].astype(str)
        if not bool((y_true.str.lower() == "unknown").all()):
            yt = y_true.to_numpy()
            out["ri_vs_class_name"] = float(rand_score(yt, labels))
            out["nmi_vs_class_name"] = float(normalized_mutual_info_score(yt, labels))
            out["ami_vs_class_name"] = float(adjusted_mutual_info_score(yt, labels))
            out["ari_vs_class_name"] = float(adjusted_rand_score(yt, labels))

    return out


def parse_args():
    p = argparse.ArgumentParser(description="Frame-level embedding clustering (UI Tab 2 equivalent)")
    p.add_argument("--emb-dir", required=True, help="Directory of .h5 embedding files")
    p.add_argument("--wav-csv-dir", default=None, help="WAV/CSV root for ground-truth labels")
    p.add_argument("--algorithm", choices=["hdbscan", "kmeans"], default="kmeans")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--k", type=int, default=60, help="K-Means clusters")
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--min-samples", type=int, default=0, help="0 = HDBSCAN default (None)")
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    p.add_argument("--use-csv-labels", action="store_true", help="Assign labels from CSV (for NMI/ARI)")
    p.add_argument("--segment-h5-recording-csv", action="store_true", default=True)
    p.add_argument("--no-segment-h5-recording-csv", action="store_false", dest="segment_h5_recording_csv")
    p.add_argument("--ignore-test-files", action="store_true", default=True)
    p.add_argument("--no-ignore-test-files", action="store_false", dest="ignore_test_files")
    p.add_argument("--filter-noise", action="store_true", default=True)
    p.add_argument("--no-filter-noise", action="store_false", dest="filter_noise")
    p.add_argument("--only-call", action="store_true")
    p.add_argument("--max-segment-duration", type=float, default=0.0)
    p.add_argument("--min-label-count", type=int, default=0)
    p.add_argument("--max-samples-per-label", type=int, default=0)
    p.add_argument("--out-json", required=True, help="Write metrics JSON here")
    p.add_argument("--out-csv", default=None, help="Optional per-frame cluster CSV (no embedding_vec)")
    return p.parse_args()


def main():
    args = parse_args()

    df_frames = load_h5_embeddings_for_clustering(args.emb_dir)
    label_config = {}

    if args.use_csv_labels:
        if not args.wav_csv_dir:
            raise SystemExit("--use-csv-labels requires --wav-csv-dir")
        max_dur = None if args.max_segment_duration <= 0 else float(args.max_segment_duration)
        df_frames = maybe_add_labels_from_csv(
            df_frames,
            wav_csv_dir=args.wav_csv_dir,
            segment_h5_use_recording_csv=bool(args.segment_h5_recording_csv),
            ignore_test_files=bool(args.ignore_test_files),
            only_call=bool(args.only_call),
            max_segment_duration=max_dur,
        )
        if args.filter_noise:
            m = df_frames["class_name"].astype(str).str.lower().to_numpy() != "unknown"
            df_frames = df_frames.loc[m].copy()
        if args.min_label_count > 0:
            df_frames = filter_by_min_label_count(df_frames, args.min_label_count)
        if args.max_samples_per_label > 0:
            df_frames = subsample_max_per_label(
                df_frames, args.max_samples_per_label, args.random_state
            )
        if len(df_frames) == 0:
            raise SystemExit("No frames left after label filters.")

        label_config = {
            "use_csv_labels": True,
            "filter_noise": args.filter_noise,
            "ignore_test_files": args.ignore_test_files,
            "only_call": args.only_call,
            "max_segment_duration": args.max_segment_duration,
            "min_label_count": args.min_label_count,
            "max_samples_per_label": args.max_samples_per_label,
        }

    X_all = np.stack(df_frames["embedding_vec"].to_numpy())
    if args.algorithm == "kmeans":
        df_results = cluster_embeddings_df(
            df_frames,
            algorithm="kmeans",
            random_state=args.random_state,
            k=args.k,
        )
        cluster_params = {"k": args.k, "random_state": args.random_state}
    else:
        df_results = cluster_embeddings_df(
            df_frames,
            algorithm="hdbscan",
            random_state=args.random_state,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric=args.metric,
            cluster_selection_method=args.cluster_selection_method,
        )
        cluster_params = {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "metric": args.metric,
            "cluster_selection_method": args.cluster_selection_method,
        }

    diag = summarize_clustering(df_results, X_for_silhouette=X_all, random_state=args.random_state)

    payload = {
        "emb_dir": str(args.emb_dir),
        "algorithm": args.algorithm,
        "cluster_params": cluster_params,
        "label_config": label_config,
        "n_h5_files": int(df_frames["h5_path"].nunique()),
        "metrics": diag,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_results.drop(columns=["embedding_vec"], errors="ignore").to_csv(out_csv, index=False)

    print(json.dumps(diag, indent=2))
    print(f"Wrote metrics to {out_json}")


if __name__ == "__main__":
    main()
