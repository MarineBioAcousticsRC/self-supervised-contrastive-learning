#!/usr/bin/env python3
"""
N-trial end-to-end driver for the four Table II representations.

Frozen models (ssl_pretrain, supervised_finetune): reuse checkpoints/embeddings;
re-run MiniBatch k-means + inertia elbow each trial.

Contrastive models (ssl_contrastive, supervised_contrastive): retrain with the
trial seed, embed, then k-sweep + elbow.

Works with the GitHub flattened layout and the lab nested Mar-Lab-Animal2vec copy.
Run from the paper repository root so subprocesses can `import nn`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_PAPER_ROOT = _SCRIPTS.parent

ALL_MODELS = (
    "ssl_pretrain",
    "supervised_finetune",
    "ssl_contrastive",
    "supervised_contrastive",
)
TRAIN_MODELS = frozenset({"ssl_contrastive", "supervised_contrastive"})

SWEEP_METRIC_ALIASES = {
    "k": ["k", "K"],
    "ari": ["ARI", "ari_vs_class_name", "ari"],
    "nmi": ["NMI", "nmi_vs_class_name", "nmi"],
    "ami": ["AMI", "ami_vs_class_name", "ami"],
    "ri": ["RI", "ri_vs_class_name", "ri"],
    "silhouette": ["Silhouette", "silhouette"],
    "calinski_harabasz": ["Calinski_Harabasz", "calinski_harabasz"],
    "davies_bouldin": ["Davies_Bouldin", "davies_bouldin"],
    "inertia": ["Inertia", "kmeans_inertia", "inertia"],
    "n_points": ["N_Points", "n_points"],
    "n_gt_classes": ["N_GT_Classes", "n_gt_classes"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N-trial train/embed/elbow pipeline")
    p.add_argument("--out-root", required=True, help="Output root (created if missing)")
    p.add_argument("--n-trials", type=int, default=2)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Comma-separated model ids",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrain-ckpt", required=True)
    p.add_argument("--supervised-ckpt", default=None, help="Unused for clustering; recorded in config")
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--label-csv-dir", required=True)
    p.add_argument("--wav-dir", required=True, help="WAV root for animal2vec_inference.py")
    p.add_argument("--wav-csv-dir", required=True, help="WAV/CSV root for clustering labels")
    p.add_argument("--emb-ssl-pretrain", required=True)
    p.add_argument("--emb-supervised-finetune", required=True)
    p.add_argument(
        "--emb-ssl-contrastive",
        default=None,
        help="Existing SSL contrastive embeddings (used when --skip-embed)",
    )
    p.add_argument(
        "--emb-supervised-contrastive",
        default=None,
        help="Existing class-aware contrastive embeddings (used when --skip-embed)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-updates", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-updates", type=int, default=500)
    p.add_argument("--save-interval-updates", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--k-min", type=int, default=10)
    p.add_argument("--k-max", type=int, default=200)
    p.add_argument("--k-step", type=int, default=10)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--skip-cluster", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run trials that already have trial_metrics.json")
    p.add_argument(
        "--keep-embeddings",
        action="store_true",
        default=True,
        help="Keep per-trial .h5 embeddings (default on)",
    )
    p.add_argument(
        "--no-keep-embeddings",
        action="store_false",
        dest="keep_embeddings",
        help="Delete per-trial .h5 after the k-sweep (saves disk for large N)",
    )
    p.add_argument(
        "--prune-intermediate-checkpoints",
        action="store_true",
        help="Delete checkpoint_1000.pt … after training; always keep checkpoint_last.pt",
    )
    p.add_argument("--paper-root", default=str(_PAPER_ROOT))
    return p.parse_args()


def detect_layout(paper_root: Path) -> Dict[str, Path | str]:
    nested = paper_root / "Mar-Lab-Animal2vec"
    nested_train = nested / "scripts" / "vocal_contrastive_finetune.py"
    flat_train = paper_root / "scripts" / "ssl_contrastive_finetune.py"
    if nested_train.is_file():
        code_root = nested
        sweep = code_root / "scripts" / "kmeans_silhouette_sweep.py"
        if not sweep.is_file():
            sweep = paper_root / "scripts" / "kmeans_sweep.py"
        analyze = code_root / "scripts" / "analyze_k_selection_metrics.py"
        if not analyze.is_file():
            analyze = paper_root / "scripts" / "analyze_k_selection_metrics.py"
        return {
            "code_root": code_root,
            "train_script": nested_train,
            "class_aware_flag": "--class-aware",
            "sweep_script": sweep,
            "analyze_script": analyze,
            "infer_script": code_root / "animal2vec_inference.py",
            "layout": "nested",
        }
    if flat_train.is_file():
        return {
            "code_root": paper_root,
            "train_script": flat_train,
            "class_aware_flag": "--supervised-contrastive",
            "sweep_script": paper_root / "scripts" / "kmeans_sweep.py",
            "analyze_script": paper_root / "scripts" / "analyze_k_selection_metrics.py",
            "infer_script": paper_root / "animal2vec_inference.py",
            "layout": "flat",
        }
    raise SystemExit(
        f"Could not find training scripts under {paper_root} "
        "(expected scripts/ssl_contrastive_finetune.py or "
        "Mar-Lab-Animal2vec/scripts/vocal_contrastive_finetune.py)"
    )


def trial_name(idx: int, seed: int) -> str:
    return f"trial_{idx:02d}_seed{seed}"


def run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[dict] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"  $ {' '.join(cmd)}")
    print(f"    log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}). See {log_path}")


def delete_intermediate_checkpoints(run_dir: Path) -> None:
    last = run_dir / "checkpoint_last.pt"
    for path in run_dir.glob("checkpoint_*.pt"):
        if path.name == "checkpoint_last.pt":
            continue
        if path.name in {"checkpoint_best.pt"}:
            continue
        path.unlink(missing_ok=True)
    if not last.is_file():
        print(f"  warning: missing {last}")


def delete_h5(emb_dir: Path) -> None:
    n = 0
    for path in emb_dir.glob("*.h5"):
        path.unlink()
        n += 1
    print(f"  deleted {n} .h5 files under {emb_dir}")


def metric_from_row(row: pd.Series, key: str):
    for name in SWEEP_METRIC_ALIASES[key]:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def extract_elbow_metrics(sweep_csv: Path, selection_csv: Path, model: str) -> Dict:
    sel = pd.read_csv(selection_csv)
    elbow = sel.loc[sel["selector"].astype(str) == "elbow_inertia"]
    if elbow.empty:
        raise SystemExit(f"No elbow_inertia row in {selection_csv}")
    er = elbow.iloc[0]
    selected_k = int(er["selected_k"])

    sweep = pd.read_csv(sweep_csv)
    k_col = "k" if "k" in sweep.columns else "K"
    model_col = None
    for cand in ("Model", "tag", "model"):
        if cand in sweep.columns:
            model_col = cand
            break
    g = sweep
    if model_col is not None:
        matched = g[g[model_col].astype(str) == str(model)]
        if not matched.empty:
            g = matched
    at_k = g.loc[g[k_col].astype(int) == selected_k]
    if at_k.empty:
        raise SystemExit(f"No sweep row at k={selected_k} in {sweep_csv}")
    row = at_k.iloc[0]

    def fnum(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def inum(val) -> Optional[int]:
        x = fnum(val)
        return None if x is None else int(x)

    return {
        "selector": "elbow_inertia",
        "selected_k": selected_k,
        "ari": fnum(metric_from_row(row, "ari")),
        "ami": fnum(metric_from_row(row, "ami")),
        "nmi": fnum(metric_from_row(row, "nmi")),
        "ri": fnum(metric_from_row(row, "ri")),
        "silhouette": fnum(metric_from_row(row, "silhouette")),
        "calinski_harabasz": fnum(metric_from_row(row, "calinski_harabasz")),
        "davies_bouldin": fnum(metric_from_row(row, "davies_bouldin")),
        "inertia": fnum(metric_from_row(row, "inertia")),
        "n_points": inum(metric_from_row(row, "n_points")),
        "n_gt_classes": inum(metric_from_row(row, "n_gt_classes")),
    }


def existing_emb_for(args: argparse.Namespace, model: str) -> Optional[Path]:
    mapping = {
        "ssl_pretrain": args.emb_ssl_pretrain,
        "supervised_finetune": args.emb_supervised_finetune,
        "ssl_contrastive": args.emb_ssl_contrastive,
        "supervised_contrastive": args.emb_supervised_contrastive,
    }
    raw = mapping[model]
    return Path(raw).expanduser() if raw else None


def run_train(
    args: argparse.Namespace,
    layout: Dict,
    model: str,
    run_dir: Path,
    seed: int,
) -> None:
    cmd = [
        args.python,
        str(layout["train_script"]),
        "--pretrain-ckpt",
        args.pretrain_ckpt,
        "--manifest-dir",
        args.manifest_dir,
        "--train-subset",
        "train_0",
        "--label-csv-dir",
        args.label_csv_dir,
        "--label-csv-format",
        "auto",
        "--label-index-rate-hz",
        "44100",
        "--save-dir",
        str(run_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-updates",
        str(args.max_updates),
        "--lr",
        str(args.lr),
        "--warmup-updates",
        str(args.warmup_updates),
        "--save-interval-updates",
        str(args.save_interval_updates),
        "--seed",
        str(seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if model == "supervised_contrastive":
        cmd.append(str(layout["class_aware_flag"]))
    run_cmd(cmd, cwd=layout["code_root"], log_path=run_dir / "train.log")
    if args.prune_intermediate_checkpoints:
        delete_intermediate_checkpoints(run_dir)


def run_embed(
    args: argparse.Namespace,
    layout: Dict,
    run_dir: Path,
    ckpt: Path,
    emb_dir: Path,
) -> None:
    emb_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        str(layout["infer_script"]),
        "--path",
        args.wav_dir,
        "--model-path",
        str(ckpt),
        "--out-path",
        str(emb_dir),
        "--sample-rate",
        "24000",
        "--device",
        args.device,
        "--write-embeddings",
        "True",
        "--average_top_k_layers",
        "12",
        "--unique-values",
        "[]",
        "--overwrite-previous-preds",
        "True",
    ]
    run_cmd(cmd, cwd=layout["code_root"], log_path=run_dir / "embed.log")


def run_cluster(
    args: argparse.Namespace,
    layout: Dict,
    model: str,
    run_dir: Path,
    emb_dir: Path,
    seed: int,
) -> Dict:
    sweep_csv = run_dir / "kmeans_sweep.csv"
    k_sel_dir = run_dir / "k_selection"
    cmd = [
        args.python,
        str(layout["sweep_script"]),
        "--emb-dir",
        str(emb_dir),
        "--tag",
        model,
        "--wav-csv-dir",
        args.wav_csv_dir,
        "--use-csv-labels",
        "--k-min",
        str(args.k_min),
        "--k-max",
        str(args.k_max),
        "--k-step",
        str(args.k_step),
        "--random-state",
        str(seed),
        "--out-csv",
        str(sweep_csv),
    ]
    run_cmd(cmd, cwd=layout["code_root"], log_path=run_dir / "cluster.log")

    k_sel_dir.mkdir(parents=True, exist_ok=True)
    cmd_an = [
        args.python,
        str(layout["analyze_script"]),
        "--csv",
        str(sweep_csv),
        "--out-dir",
        str(k_sel_dir),
        "--model",
        model,
    ]
    run_cmd(cmd_an, cwd=layout["code_root"], log_path=run_dir / "cluster.log")
    selection_csv = k_sel_dir / "selector_comparison.csv"
    return extract_elbow_metrics(sweep_csv, selection_csv, model)


def run_one_model(
    args: argparse.Namespace,
    layout: Dict,
    model: str,
    trial_idx: int,
    seed: int,
) -> Optional[Path]:
    run_dir = Path(args.out_root) / model / trial_name(trial_idx, seed)
    metrics_path = run_dir / "trial_metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"=== skip existing {model} {trial_name(trial_idx, seed)}")
        return metrics_path

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {model} {trial_name(trial_idx, seed)}")
    trains = model in TRAIN_MODELS
    emb_dir: Optional[Path] = None
    ckpt: Optional[Path] = None

    if trains and not args.skip_train:
        run_train(args, layout, model, run_dir, seed)
        ckpt = run_dir / "checkpoint_last.pt"
    elif trains:
        ckpt = run_dir / "checkpoint_last.pt"
        if not ckpt.is_file():
            print(f"  skip-train: missing {ckpt}")
            ckpt = None

    if trains and not args.skip_embed:
        if ckpt is None or not ckpt.is_file():
            raise SystemExit(f"Need checkpoint_last.pt to embed {run_dir}")
        emb_dir = run_dir / "embeddings"
        run_embed(args, layout, run_dir, ckpt, emb_dir)
    else:
        existing = existing_emb_for(args, model)
        if existing is None or not existing.is_dir():
            raise SystemExit(
                f"{model} needs embeddings: pass --emb-{model.replace('_', '-')} "
                f"or run without --skip-embed"
            )
        emb_dir = existing
        print(f"  reuse embeddings: {emb_dir}")

    metrics = {
        "model": model,
        "trial_idx": trial_idx,
        "seed": seed,
        "emb_dir": str(emb_dir),
        "checkpoint": str(ckpt) if ckpt else None,
        "skip_train": bool(args.skip_train or not trains),
        "skip_embed": bool(args.skip_embed or not trains),
    }
    if not args.skip_cluster:
        elbow = run_cluster(args, layout, model, run_dir, emb_dir, seed)
        metrics.update(elbow)
        if trains and not args.skip_embed and not args.keep_embeddings:
            delete_h5(run_dir / "embeddings")

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote {metrics_path}")
    return metrics_path


def maybe_aggregate(args: argparse.Namespace) -> None:
    agg = _SCRIPTS / "aggregate_n_trials.py"
    if not agg.is_file():
        print("aggregate_n_trials.py not found; skip aggregate")
        return
    cmd = [args.python, str(agg), "--out-root", str(Path(args.out_root).expanduser())]
    subprocess.run(cmd, check=False)


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise SystemExit("--n-trials must be >= 1")
    paper_root = Path(args.paper_root).expanduser().resolve()
    layout = detect_layout(paper_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Choose from {ALL_MODELS}")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_root = str(out_root)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_root": str(paper_root),
        "layout": layout["layout"],
        "code_root": str(layout["code_root"]),
        "n_trials": args.n_trials,
        "base_seed": args.base_seed,
        "models": models,
        "max_updates": args.max_updates,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "k_step": args.k_step,
        "skip_train": args.skip_train,
        "skip_embed": args.skip_embed,
        "keep_embeddings": args.keep_embeddings,
        "prune_intermediate_checkpoints": args.prune_intermediate_checkpoints,
        "paths": {
            "pretrain_ckpt": args.pretrain_ckpt,
            "supervised_ckpt": args.supervised_ckpt,
            "manifest_dir": args.manifest_dir,
            "label_csv_dir": args.label_csv_dir,
            "wav_dir": args.wav_dir,
            "wav_csv_dir": args.wav_csv_dir,
            "emb_ssl_pretrain": args.emb_ssl_pretrain,
            "emb_supervised_finetune": args.emb_supervised_finetune,
            "emb_ssl_contrastive": args.emb_ssl_contrastive,
            "emb_supervised_contrastive": args.emb_supervised_contrastive,
        },
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Layout: {layout['layout']}  code_root={layout['code_root']}")
    print(f"Out: {out_root}")

    for i in range(1, args.n_trials + 1):
        seed = args.base_seed + i - 1
        for model in models:
            run_one_model(args, layout, model, i, seed)
        maybe_aggregate(args)

    maybe_aggregate(args)
    print("Done.")


if __name__ == "__main__":
    main()
