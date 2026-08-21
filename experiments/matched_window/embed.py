#!/usr/bin/env python3
"""
Embed NIPS4Bplus train clips for matched-window experiments.

Examples:

  python -m experiments.matched_window.embed --experiment all --model a2v --sanity-n 10
  python -m experiments.matched_window.embed --experiment 2 --model birdnet
  python -m experiments.matched_window.embed --experiment 3 --model perch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.matched_window.audio_slices import (  # noqa: E402
    all_windows_fit,
    clip_covers_windows,
    expected_n_samples,
    fitting_windows,
    load_mono,
    slice_window,
)
from experiments.matched_window.checkpoint import (  # noqa: E402
    discover_ssl_contrastive_checkpoint,
)
from experiments.matched_window.config import (  # noqa: E402
    A2V_SAMPLE_RATE,
    BIRDNET_SAMPLE_RATE,
    BIRDNET_WINDOWS,
    DEFAULT_OUT_DIR,
    DEFAULT_WAV_DIR,
    EXPECTED_CLIP_DURATION_S,
    MIN_CLIP_DURATION_S,
    MIN_EXP1_CLIP_DURATION_S,
    ONE_SECOND_WINDOWS,
    PERCH_SAMPLE_RATE,
    PERCH_WINDOWS,
    ExperimentConfig,
    sample_key,
)
from experiments.matched_window.labels import (  # noqa: E402
    label_window,
    labels_to_csv_cell,
    load_clip_intervals,
)
from experiments.matched_window.recordings import (  # noqa: E402
    RecordingClip,
    assert_clips_nonempty,
    list_train_wavs,
)
from experiments.matched_window.sanity import (  # noqa: E402
    assert_matched_keys,
    print_sanity_report,
)

META_COLUMNS = [
    "sample_key",
    "recording_id",
    "window_start_s",
    "window_end_s",
    "row_idx",
    "model",
    "embedding_dim",
    "a2v_num_frames",
    "a2v_trim_frames",
    "window_labels",
    "dominant_species",
    "is_single_species",
    "wav_path",
    "waveform_samples",
    "source_sr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-window embedding experiments")
    p.add_argument("--experiment", choices=["1", "2", "3", "all"], required=True)
    p.add_argument("--model", choices=["a2v", "birdnet", "perch"], required=True)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--label-csv-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--a2v-checkpoint", type=Path, default=None)
    p.add_argument(
        "--checkpoint-search-root",
        type=Path,
        default=None,
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--sanity-n", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--birdnet-version", default=None)
    p.add_argument("--perch-model-name", default="perch_v2")
    return p.parse_args()


def _experiments_for(choice: str) -> List[str]:
    return ["1", "2", "3"] if choice == "all" else [choice]


def _validate_model_experiment(model: str, experiment: str) -> None:
    ok = {
        ("a2v", "1"),
        ("a2v", "2"),
        ("a2v", "3"),
        ("birdnet", "2"),
        ("perch", "3"),
    }
    if (model, experiment) not in ok:
        raise SystemExit(
            f"Model {model!r} cannot run experiment {experiment}. "
            "Valid: a2v∈{1,2,3}, birdnet∈{2}, perch∈{3}."
        )


def _label_meta(
    clip: RecordingClip,
    wav_root: Path,
    label_csv_dir: Optional[Path],
    start_s: float,
    end_s: float,
) -> Dict[str, Any]:
    intervals, _ = load_clip_intervals(
        clip.wav_path,
        wav_root,
        label_csv_dir,
        clip_dur_s=EXPECTED_CLIP_DURATION_S,
    )
    lab = label_window(intervals, start_s, end_s)
    return {
        "window_labels": labels_to_csv_cell(lab.window_labels),
        "dominant_species": lab.dominant_species or "",
        "is_single_species": bool(lab.is_single_species),
    }


def _base_row(
    clip: RecordingClip,
    start_s: float,
    end_s: float,
    *,
    model: str,
    embedding_dim: int,
    waveform_samples: int,
    source_sr: int,
    row_idx: int,
    a2v_num_frames: Optional[int] = None,
    a2v_trim_frames: int = 0,
    label_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lab = label_fields or {
        "window_labels": "",
        "dominant_species": "",
        "is_single_species": False,
    }
    return {
        "sample_key": sample_key(clip.recording_id, start_s, end_s),
        "recording_id": clip.recording_id,
        "window_start_s": float(start_s),
        "window_end_s": float(end_s),
        "row_idx": int(row_idx),
        "model": model,
        "embedding_dim": int(embedding_dim),
        "a2v_num_frames": "" if a2v_num_frames is None else int(a2v_num_frames),
        "a2v_trim_frames": int(a2v_trim_frames),
        "window_labels": lab["window_labels"],
        "dominant_species": lab["dominant_species"],
        "is_single_species": lab["is_single_species"],
        "wav_path": str(clip.wav_path),
        "waveform_samples": int(waveform_samples),
        "source_sr": int(source_sr),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {c: r.get(c, "") for c in META_COLUMNS}
            w.writerow(out)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _save_stack(
    out_dir: Path,
    name: str,
    vectors: List[np.ndarray],
    rows: List[Dict[str, Any]],
) -> None:
    emb_dir = out_dir / "embeddings"
    meta_dir = out_dir / "metadata"
    emb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if not vectors:
        raise RuntimeError(f"No embeddings to save for {name}")
    X = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    np.save(emb_dir / f"{name}.npy", X)
    _write_csv(meta_dir / f"{name}.csv", rows)
    print(f"Wrote {X.shape} -> embeddings/{name}.npy and metadata/{name}.csv ({len(rows)} rows)")


def _save_frames_stack(
    out_dir: Path,
    name: str,
    frames_list: List[np.ndarray],
) -> None:
    """Save dense frames as float32 (N, T, D) — no flatten."""
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    if not frames_list:
        raise RuntimeError(f"No frame stacks to save for {name}")
    shapes = {tuple(np.asarray(f).shape) for f in frames_list}
    if len(shapes) != 1:
        raise RuntimeError(f"Inconsistent frame shapes for {name}: {shapes}")
    X = np.stack([np.asarray(f, dtype=np.float32) for f in frames_list], axis=0)
    if X.ndim != 3:
        raise RuntimeError(f"Expected (N, T, D) frames, got {X.shape}")
    path = emb_dir / f"{name}_frames.npy"
    np.save(path, X)
    print(f"Wrote {X.shape} -> embeddings/{name}_frames.npy")


def _append_matched_meta(
    out_dir: Path,
    matched_name: str,
    rows: List[Dict[str, Any]],
    *,
    model: str,
) -> None:
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{matched_name}.csv"
    kept: List[Dict[str, Any]] = []
    if path.exists():
        for r in _read_csv(path):
            if str(r.get("model", "")) != model:
                kept.append(r)
    kept.extend(rows)
    kept.sort(key=lambda r: (str(r.get("sample_key", "")), str(r.get("model", ""))))
    _write_csv(path, kept)
    print(f"Updated metadata/{matched_name}.csv ({len(kept)} rows)")


def run_a2v(
    experiment: str,
    clips: Sequence[RecordingClip],
    cfg: ExperimentConfig,
    *,
    print_sanity: bool,
) -> None:
    from experiments.matched_window.a2v_embedder import (
        Animal2VecEmbedder,
        apply_optional_trim,
        resolve_frame_count,
    )

    ckpt = discover_ssl_contrastive_checkpoint(
        cfg.checkpoint_search_root, cfg.a2v_checkpoint
    )
    print(f"A2V checkpoint: {ckpt}")
    embedder = Animal2VecEmbedder(str(ckpt), device=cfg.device)

    windows = {
        "1": ONE_SECOND_WINDOWS,
        "2": BIRDNET_WINDOWS,
        "3": PERCH_WINDOWS,
    }[experiment]
    name = {"1": "a2v_1s", "2": "a2v_3s", "3": "a2v_5s"}[experiment]
    duration_s = float(windows[0][1] - windows[0][0])

    frames_list: List[np.ndarray] = []
    meta_partial: List[Dict[str, Any]] = []
    frame_counts: List[int] = []
    sanity_done = False
    n_skip_short = 0

    for clip in tqdm(clips, desc=f"A2V exp{experiment}"):
        audio, sr, dur = load_mono(clip.wav_path, A2V_SAMPLE_RATE)
        # Exp 1: keep every complete 1s window that fits (e.g. 4.2s → 0–1…3–4).
        # Exp 2/3: require all matched windows (need full ~5s coverage).
        if experiment == "1":
            use_windows = fitting_windows(audio.shape[0], sr, windows)
            if dur + 1e-6 < MIN_EXP1_CLIP_DURATION_S or not use_windows:
                n_skip_short += 1
                continue
        else:
            if not clip_covers_windows(
                dur, windows, min_duration_s=MIN_CLIP_DURATION_S
            ) or not all_windows_fit(audio.shape[0], sr, windows):
                n_skip_short += 1
                continue
            use_windows = list(windows)
            if abs(dur - EXPECTED_CLIP_DURATION_S) > 0.15:
                print(
                    f"WARN {clip.wav_path.name}: duration={dur:.4f}s "
                    f"(expected ~{EXPECTED_CLIP_DURATION_S}s)"
                )

        for start_s, end_s in use_windows:
            need = expected_n_samples(end_s - start_s, sr)
            chunk = slice_window(audio, sr, start_s, end_s, strict=True)
            frames = embedder.embed_waveform(chunk)
            frames_list.append(frames)
            frame_counts.append(int(frames.shape[0]))
            lab = _label_meta(clip, cfg.wav_dir, cfg.label_csv_dir, start_s, end_s)
            meta_partial.append(
                {
                    "clip": clip,
                    "start_s": start_s,
                    "end_s": end_s,
                    "need": need,
                    "sr": sr,
                    "lab": lab,
                }
            )

            if print_sanity and not sanity_done:
                print_sanity_report(
                    {
                        "recording_id": clip.recording_id,
                        "wav_path": str(clip.wav_path),
                        "original_waveform_length_samples": int(audio.shape[0]),
                        "original_duration_s": float(dur),
                        "source_sr": sr,
                        "windows": [
                            {
                                "window_boundaries": (start_s, end_s),
                                "window_sample_count": int(chunk.shape[0]),
                                "Animal2Vec_frame_matrix_shape": tuple(frames.shape),
                                "Flattened_Animal2Vec_shape": (int(frames.size),),
                                "window_labels": lab["window_labels"],
                                "dominant_species": lab["dominant_species"],
                            }
                        ],
                    }
                )
                sanity_done = True

    if n_skip_short:
        reason = (
            "clips with no complete 1s window"
            if experiment == "1"
            else "short clips"
        )
        print(f"Skipped {n_skip_short} {reason} for A2V exp{experiment}")
    if not frames_list:
        raise RuntimeError(f"No A2V embeddings produced for experiment {experiment}")

    target_t, trim_rec = resolve_frame_count(frame_counts, duration_s, allow_trim=True)
    print(
        f"A2V duration={duration_s}s frame counts={trim_rec.observed_counts} "
        f"-> target_t={target_t} trimmed={trim_rec.trimmed}"
    )
    trim_path = cfg.trim_config_path()
    existing = json.loads(trim_path.read_text()) if trim_path.exists() else {}
    existing[str(duration_s)] = {
        "duration_s": trim_rec.duration_s,
        "observed_counts": trim_rec.observed_counts,
        "min_frames": trim_rec.min_frames,
        "trimmed": trim_rec.trimmed,
    }
    trim_path.parent.mkdir(parents=True, exist_ok=True)
    trim_path.write_text(json.dumps(existing, indent=2))

    vectors: List[np.ndarray] = []
    trimmed_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for frames, mp in zip(frames_list, meta_partial):
        trimmed_frames, was_trimmed = apply_optional_trim(frames, target_t)
        flat = trimmed_frames.reshape(-1)
        vectors.append(flat)
        trimmed_list.append(np.asarray(trimmed_frames, dtype=np.float32))
        rows.append(
            _base_row(
                mp["clip"],
                mp["start_s"],
                mp["end_s"],
                model="a2v",
                embedding_dim=int(flat.shape[0]),
                waveform_samples=int(mp["need"]),
                source_sr=int(mp["sr"]),
                row_idx=len(rows),
                a2v_num_frames=int(trimmed_frames.shape[0]),
                a2v_trim_frames=int(was_trimmed),
                label_fields=mp["lab"],
            )
        )

    assert len(set(int(r["a2v_num_frames"]) for r in rows)) == 1
    _save_stack(cfg.out_dir, name, vectors, rows)
    _save_frames_stack(cfg.out_dir, name, trimmed_list)

    if experiment == "2":
        _append_matched_meta(cfg.out_dir, "matched_3s", rows, model="a2v")
        bird_meta = cfg.out_dir / "metadata" / "birdnet_3s.csv"
        if bird_meta.exists():
            assert_matched_keys(rows, _read_csv(bird_meta), context="exp2 a2v vs birdnet")
    elif experiment == "3":
        _append_matched_meta(cfg.out_dir, "matched_5s", rows, model="a2v")
        perch_meta = cfg.out_dir / "metadata" / "perch_5s.csv"
        if perch_meta.exists():
            assert_matched_keys(rows, _read_csv(perch_meta), context="exp3 a2v vs perch")


def run_birdnet(
    clips: Sequence[RecordingClip],
    cfg: ExperimentConfig,
    *,
    version: Optional[str],
    print_sanity: bool,
) -> None:
    from experiments.matched_window.birdnet_embedder import BirdNETEmbedder

    embedder = BirdNETEmbedder(version=version)
    vectors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    sanity_done = False
    n_skip_short = 0

    for clip in tqdm(clips, desc="BirdNET 3s"):
        audio, sr, dur = load_mono(clip.wav_path, BIRDNET_SAMPLE_RATE)
        if not clip_covers_windows(dur, BIRDNET_WINDOWS, min_duration_s=MIN_CLIP_DURATION_S) or not all_windows_fit(
            audio.shape[0], sr, BIRDNET_WINDOWS
        ):
            n_skip_short += 1
            continue
        for start_s, end_s in BIRDNET_WINDOWS:
            need = expected_n_samples(end_s - start_s, sr)
            chunk = slice_window(audio, sr, start_s, end_s, strict=True)
            vec = embedder.embed_3s(chunk)
            lab = _label_meta(clip, cfg.wav_dir, cfg.label_csv_dir, start_s, end_s)
            vectors.append(vec)
            rows.append(
                _base_row(
                    clip,
                    start_s,
                    end_s,
                    model="birdnet",
                    embedding_dim=int(vec.shape[0]),
                    waveform_samples=int(need),
                    source_sr=int(sr),
                    row_idx=len(rows),
                    label_fields=lab,
                )
            )
            if print_sanity and not sanity_done:
                print_sanity_report(
                    {
                        "recording_id": clip.recording_id,
                        "wav_path": str(clip.wav_path),
                        "original_waveform_length_samples": int(audio.shape[0]),
                        "original_duration_s": float(dur),
                        "source_sr": sr,
                        "windows": [
                            {
                                "window_boundaries": (start_s, end_s),
                                "window_sample_count": int(chunk.shape[0]),
                                "BirdNET_embedding_shape": tuple(vec.shape),
                                "window_labels": lab["window_labels"],
                            }
                        ],
                    }
                )
                sanity_done = True

    if n_skip_short:
        print(f"Skipped {n_skip_short} short clips for BirdNET")
    if not vectors:
        raise RuntimeError("No BirdNET embeddings produced")

    _save_stack(cfg.out_dir, "birdnet_3s", vectors, rows)
    _append_matched_meta(cfg.out_dir, "matched_3s", rows, model="birdnet")

    a2v_meta = cfg.out_dir / "metadata" / "a2v_3s.csv"
    if a2v_meta.exists():
        assert_matched_keys(_read_csv(a2v_meta), rows, context="exp2 a2v vs birdnet")


def run_perch(
    clips: Sequence[RecordingClip],
    cfg: ExperimentConfig,
    *,
    model_name: str,
    print_sanity: bool,
) -> None:
    from experiments.matched_window.perch_embedder import PerchEmbedder

    embedder = PerchEmbedder(model_name=model_name)
    vectors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    sanity_done = False
    n_skip_short = 0

    for clip in tqdm(clips, desc="Perch 5s"):
        audio, sr, dur = load_mono(clip.wav_path, PERCH_SAMPLE_RATE)
        if not clip_covers_windows(dur, PERCH_WINDOWS, min_duration_s=MIN_CLIP_DURATION_S) or not all_windows_fit(
            audio.shape[0], sr, PERCH_WINDOWS
        ):
            n_skip_short += 1
            continue
        for start_s, end_s in PERCH_WINDOWS:
            need = expected_n_samples(end_s - start_s, sr)
            chunk = slice_window(audio, sr, start_s, end_s, strict=True)
            vec = embedder.embed_5s(chunk)
            lab = _label_meta(clip, cfg.wav_dir, cfg.label_csv_dir, start_s, end_s)
            vectors.append(vec)
            rows.append(
                _base_row(
                    clip,
                    start_s,
                    end_s,
                    model="perch",
                    embedding_dim=int(vec.shape[0]),
                    waveform_samples=int(need),
                    source_sr=int(sr),
                    row_idx=len(rows),
                    label_fields=lab,
                )
            )
            if print_sanity and not sanity_done:
                print_sanity_report(
                    {
                        "recording_id": clip.recording_id,
                        "wav_path": str(clip.wav_path),
                        "original_waveform_length_samples": int(audio.shape[0]),
                        "original_duration_s": float(dur),
                        "source_sr": sr,
                        "windows": [
                            {
                                "window_boundaries": (start_s, end_s),
                                "window_sample_count": int(chunk.shape[0]),
                                "Perch_embedding_shape": tuple(vec.shape),
                                "window_labels": lab["window_labels"],
                            }
                        ],
                    }
                )
                sanity_done = True

    if n_skip_short:
        print(f"Skipped {n_skip_short} short clips for Perch")
    if not vectors:
        raise RuntimeError("No Perch embeddings produced")

    _save_stack(cfg.out_dir, "perch_5s", vectors, rows)
    _append_matched_meta(cfg.out_dir, "matched_5s", rows, model="perch")

    a2v_meta = cfg.out_dir / "metadata" / "a2v_5s.csv"
    if a2v_meta.exists():
        assert_matched_keys(_read_csv(a2v_meta), rows, context="exp3 a2v vs perch")


def main() -> int:
    args = parse_args()
    limit = args.sanity_n if args.sanity_n is not None else args.limit
    cfg = ExperimentConfig(
        wav_dir=Path(args.wav_dir).expanduser().resolve(),
        label_csv_dir=(
            Path(args.label_csv_dir).expanduser().resolve() if args.label_csv_dir else None
        ),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        a2v_checkpoint=(
            Path(args.a2v_checkpoint).expanduser().resolve() if args.a2v_checkpoint else None
        ),
        checkpoint_search_root=Path(args.checkpoint_search_root).expanduser().resolve(),
        device=args.device,
        sanity_n=limit,
    )
    cfg.ensure_dirs()

    clips = list_train_wavs(cfg.wav_dir, limit=limit)
    assert_clips_nonempty(clips, cfg.wav_dir)
    print(f"Train clips: {len(clips)} under {cfg.wav_dir}")

    print_sanity = limit is not None and int(limit) <= 20

    for exp in _experiments_for(args.experiment):
        _validate_model_experiment(args.model, exp)
        if args.model == "a2v":
            run_a2v(exp, clips, cfg, print_sanity=print_sanity)
        elif args.model == "birdnet":
            run_birdnet(clips, cfg, version=args.birdnet_version, print_sanity=print_sanity)
        else:
            run_perch(
                clips, cfg, model_name=args.perch_model_name, print_sanity=print_sanity
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
