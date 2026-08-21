#!/usr/bin/env python3
"""
Sweep MiniBatch k-means over embedding directories and record clustering metrics.

Example:

  python scripts/kmeans_sweep.py \\
    --emb-dir /path/to/embeddings/self_supervised_pretrain \\
    --emb-dir /path/to/embeddings/self_supervised_contrastive \\
    --wav-csv-dir /path/to/nips4bplus \\
    --use-csv-labels \\
    --k-min 10 --k-max 200 --k-step 10 \\
    --out-csv /path/to/kmeans_sweep.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from frame_level_clustering import (  # noqa: E402
    cluster_embeddings_df,
    filter_by_min_label_count,
    load_h5_embeddings_for_clustering,
    maybe_add_labels_from_csv,
    subsample_max_per_label,
    summarize_clustering,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MiniBatch k-means sweep over embedding directories (inertia, ARI, AMI, NMI, silhouette)",
    )
    p.add_argument(
        "--emb-dir",
        action="append",
        dest="emb_dirs",
        required=True,
        help="Embeddings directory with .h5 files (repeat for multiple runs)",
    )
    p.add_argument("--wav-csv-dir", default=None, help="WAV/CSV root for ground-truth labels")
    p.add_argument("--use-csv-labels", action="store_true", help="Assign CSV labels (needed for NMI/ARI/RI/AMI)")
    p.add_argument("--k-min", type=int, default=10)
    p.add_argument("--k-max", type=int, default=200)
    p.add_argument(
        "--k-step",
        type=int,
        default=10,
        help="Step between k values (use 1 for every integer from k-min..k-max)",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-silhouette-n", type=int, default=5000)
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
    p.add_argument(
        "--out-csv",
        required=True,
        help="Long-form CSV (one row per emb_dir × k). Also writes a wide CSV + Excel workbook beside it.",
    )
    p.add_argument(
        "--out-xlsx",
        default=None,
        help="Optional Excel path. Default: same stem as --out-csv with .xlsx",
    )
    p.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        help="Optional short name per --emb-dir (same order). Defaults to directory basename.",
    )
    return p.parse_args()


METRIC_COLS = [
    ("silhouette", "Silhouette"),
    ("ri_vs_class_name", "RI"),
    ("ari_vs_class_name", "ARI"),
    ("nmi_vs_class_name", "NMI"),
    ("ami_vs_class_name", "AMI"),
    ("kmeans_inertia", "Inertia"),
    ("calinski_harabasz", "Calinski_Harabasz"),
    ("davies_bouldin", "Davies_Bouldin"),
]


def long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Excel-friendly long table: clear headers, sorted by tag then k."""
    out = df.copy()
    rename = {
        "tag": "Model",
        "emb_dir": "Embeddings_Dir",
        "k": "k",
        "n_points": "N_Points",
        "n_clusters": "N_Clusters",
        "n_gt_classes": "N_GT_Classes",
        "silhouette": "Silhouette",
        "ri_vs_class_name": "RI",
        "ari_vs_class_name": "ARI",
        "nmi_vs_class_name": "NMI",
        "ami_vs_class_name": "AMI",
        "kmeans_inertia": "Inertia",
        "calinski_harabasz": "Calinski_Harabasz",
        "davies_bouldin": "Davies_Bouldin",
        "elapsed_s": "Elapsed_s",
        "random_state": "Random_State",
        "max_silhouette_n": "Silhouette_Subsample_N",
    }
    out = out.rename(columns=rename)
    col_order = [c for c in rename.values() if c in out.columns]
    out = out[col_order].sort_values(["Model", "k"]).reset_index(drop=True)
    return out


def wide_metric_table(df: pd.DataFrame, value_col: str, nice_name: str) -> pd.DataFrame:
    """One row per k; one column per model for a single metric."""
    pivot = df.pivot_table(index="k", columns="tag", values=value_col, aggfunc="first")
    pivot = pivot.sort_index().reset_index()
    pivot.columns = ["k"] + [f"{c}_{nice_name}" for c in pivot.columns[1:]]
    return pivot


def wide_all_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per k; columns like pretrain_Silhouette, supervised_ARI, ..."""
    pieces = []
    for src, nice in METRIC_COLS:
        if src not in df.columns:
            continue
        p = df.pivot_table(index="k", columns="tag", values=src, aggfunc="first")
        p = p.rename(columns={c: f"{c}_{nice}" for c in p.columns})
        pieces.append(p)
    if not pieces:
        return pd.DataFrame({"k": sorted(df["k"].unique())})
    wide = pieces[0].join(pieces[1:], how="outer").sort_index().reset_index()
    return wide


def save_analysis_tables(rows: list, out_csv: Path, out_xlsx: Optional[Path]) -> None:
    """Write long CSV, wide CSV, and multi-sheet Excel workbook."""
    if not rows:
        return
    raw = pd.DataFrame(rows)
    long_df = long_table(raw)
    wide_df = wide_all_metrics_table(raw)

    long_df.to_csv(out_csv, index=False)
    wide_csv = out_csv.with_name(out_csv.stem + "_wide.csv")
    wide_df.to_csv(wide_csv, index=False)

    xlsx_path = out_xlsx if out_xlsx is not None else out_csv.with_suffix(".xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            long_df.to_excel(writer, sheet_name="All_Results", index=False)
            wide_df.to_excel(writer, sheet_name="Wide_by_k", index=False)
            for src, nice in METRIC_COLS:
                if src not in raw.columns:
                    continue
                wide_metric_table(raw, src, nice).to_excel(writer, sheet_name=nice[:31], index=False)
            # Best-k summary sheet
            summary_rows = []
            for tag, g in raw.groupby("tag"):
                best_sil = g.loc[g["silhouette"].astype(float).idxmax()]
                summary_rows.append(
                    {
                        "Model": tag,
                        "Best_k_by_Silhouette": int(best_sil["k"]),
                        "Silhouette": float(best_sil["silhouette"]),
                        "RI": float(best_sil.get("ri_vs_class_name", np.nan)),
                        "ARI": float(best_sil.get("ari_vs_class_name", np.nan)),
                        "NMI": float(best_sil.get("nmi_vs_class_name", np.nan)),
                        "AMI": float(best_sil.get("ami_vs_class_name", np.nan)),
                    }
                )
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Best_k_Summary", index=False)
        print(f"  Excel: {xlsx_path}")
    except Exception as e:
        print(f"  (Excel .xlsx skipped: {e}. Install openpyxl for Excel output.)")

    print(f"  Long CSV:  {out_csv}")
    print(f"  Wide CSV:  {wide_csv}")


def prepare_frames(args: argparse.Namespace, emb_dir: str) -> pd.DataFrame:
    df = load_h5_embeddings_for_clustering(emb_dir)
    if not args.use_csv_labels:
        return df
    if not args.wav_csv_dir:
        raise SystemExit("--use-csv-labels requires --wav-csv-dir")

    max_dur = None if args.max_segment_duration <= 0 else float(args.max_segment_duration)
    df = maybe_add_labels_from_csv(
        df,
        wav_csv_dir=args.wav_csv_dir,
        segment_h5_use_recording_csv=bool(args.segment_h5_recording_csv),
        ignore_test_files=bool(args.ignore_test_files),
        only_call=bool(args.only_call),
        max_segment_duration=max_dur,
    )
    if args.filter_noise:
        m = df["class_name"].astype(str).str.lower().to_numpy() != "unknown"
        df = df.loc[m].copy()
    if args.min_label_count > 0:
        df = filter_by_min_label_count(df, args.min_label_count)
    if args.max_samples_per_label > 0:
        df = subsample_max_per_label(df, args.max_samples_per_label, args.random_state)
    if len(df) == 0:
        raise SystemExit(f"No frames left after label filters for: {emb_dir}")
    return df


def k_values(k_min: int, k_max: int, k_step: int) -> List[int]:
    if k_min < 2:
        raise SystemExit("--k-min must be >= 2")
    if k_max < k_min:
        raise SystemExit("--k-max must be >= --k-min")
    if k_step < 1:
        raise SystemExit("--k-step must be >= 1")
    return list(range(int(k_min), int(k_max) + 1, int(k_step)))


def main() -> None:
    args = parse_args()
    emb_dirs = [str(Path(p).expanduser()) for p in args.emb_dirs]
    tags = args.tags
    if tags is not None and len(tags) != len(emb_dirs):
        raise SystemExit(f"--tag count ({len(tags)}) must match --emb-dir count ({len(emb_dirs)})")
    if tags is None:
        tags = [Path(p).name for p in emb_dirs]

    ks = k_values(args.k_min, args.k_max, args.k_step)
    out_csv = Path(args.out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_xlsx = Path(args.out_xlsx).expanduser() if args.out_xlsx else out_csv.with_suffix(".xlsx")

    rows = []
    print(f"Sweep: {len(emb_dirs)} emb dirs × {len(ks)} k values")
    print(f"k grid: {ks[0]}..{ks[-1]} step {args.k_step} ({len(ks)} values)")
    print(f"Outputs: {out_csv}  |  {out_csv.stem}_wide.csv  |  {out_xlsx}")

    for emb_dir, tag in zip(emb_dirs, tags):
        print(f"\n=== [{tag}] loading {emb_dir} ===")
        t0 = time.time()
        df_frames = prepare_frames(args, emb_dir)
        X_all = np.stack(df_frames["embedding_vec"].to_numpy())
        n_gt = (
            int(df_frames["class_name"].astype(str).nunique())
            if "class_name" in df_frames.columns
            else 0
        )
        print(
            f"[{tag}] frames={len(df_frames):,} h5={df_frames['h5_path'].nunique()} "
            f"gt_classes={n_gt} load_s={time.time() - t0:.1f}"
        )

        for k in ks:
            if k > X_all.shape[0]:
                print(f"[{tag}] skip k={k} (n_frames={X_all.shape[0]})")
                continue
            t1 = time.time()
            df_results = cluster_embeddings_df(
                df_frames,
                algorithm="kmeans",
                random_state=args.random_state,
                k=int(k),
            )
            diag = summarize_clustering(
                df_results,
                X_for_silhouette=X_all,
                max_silhouette_n=args.max_silhouette_n,
                random_state=args.random_state,
            )
            inertia = float(df_results.attrs.get("cluster_metrics", {}).get("kmeans_inertia", np.nan))
            row = {
                "tag": tag,
                "emb_dir": emb_dir,
                "k": int(k),
                "n_points": int(diag.get("n_points", 0)),
                "n_clusters": int(diag.get("n_clusters_including_noise", 0)),
                "n_gt_classes": n_gt,
                "silhouette": diag.get("silhouette", np.nan),
                "ri_vs_class_name": diag.get("ri_vs_class_name", np.nan),
                "ari_vs_class_name": diag.get("ari_vs_class_name", np.nan),
                "nmi_vs_class_name": diag.get("nmi_vs_class_name", np.nan),
                "ami_vs_class_name": diag.get("ami_vs_class_name", np.nan),
                "kmeans_inertia": inertia,
                "calinski_harabasz": diag.get("calinski_harabasz", np.nan),
                "davies_bouldin": diag.get("davies_bouldin", np.nan),
                "elapsed_s": round(time.time() - t1, 2),
                "random_state": int(args.random_state),
                "max_silhouette_n": int(args.max_silhouette_n),
            }
            rows.append(row)
            # Checkpoint after every k (CSV + Excel-friendly tables)
            save_analysis_tables(rows, out_csv, out_xlsx)
            print(
                f"[{tag}] k={k:3d} sil={row['silhouette']:.4f} "
                f"CH={row['calinski_harabasz']:.1f} DB={row['davies_bouldin']:.4f} "
                f"ARI={row['ari_vs_class_name']:.4f} NMI={row['nmi_vs_class_name']:.4f} "
                f"({row['elapsed_s']}s)"
            )

    print(f"\nDone. Wrote {len(rows)} rows.")
    if rows:
        save_analysis_tables(rows, out_csv, out_xlsx)
        df = pd.DataFrame(rows)
        for tag, g in df.groupby("tag"):
            best = g.loc[g["silhouette"].astype(float).idxmax()]
            print(
                f"  best silhouette [{tag}]: k={int(best['k'])} "
                f"sil={float(best['silhouette']):.4f} "
                f"ARI={float(best['ari_vs_class_name']):.4f} "
                f"NMI={float(best['nmi_vs_class_name']):.4f}"
            )


if __name__ == "__main__":
    main()
