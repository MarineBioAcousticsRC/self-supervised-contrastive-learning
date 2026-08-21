"""Exact window slicing + per-model resampling."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int, float]:
    """Load mono float32 audio resampled to target_sr. Returns (audio, sr, duration_s)."""
    import librosa

    audio, _ = librosa.load(str(path), sr=int(target_sr), mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(target_sr)
    duration_s = float(audio.shape[0]) / float(sr)
    return audio, sr, duration_s


def load_native(path: Path) -> Tuple[np.ndarray, int, float]:
    """Load mono at file sample rate (no resample)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    duration_s = float(audio.shape[0]) / float(sr)
    return audio, int(sr), duration_s


def expected_n_samples(duration_s: float, sample_rate: int) -> int:
    return int(round(float(duration_s) * float(sample_rate)))


def max_pad_samples(sample_rate: int) -> int:
    """Allow tiny rounding shortfall (~2 ms), not content padding."""
    return max(2, int(round(0.002 * float(sample_rate))))


def window_fits(
    audio_len: int,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> bool:
    need = expected_n_samples(float(end_s) - float(start_s), sample_rate)
    start_i = int(round(float(start_s) * int(sample_rate)))
    return start_i >= 0 and (audio_len - start_i + max_pad_samples(sample_rate)) >= need


def clip_covers_windows(
    duration_s: float,
    windows: list,
    *,
    min_duration_s: float = 5.0,
) -> bool:
    """True if reported duration is long enough for every window end."""
    if float(duration_s) + 1e-6 < float(min_duration_s):
        return False
    max_end = max(float(e) for _, e in windows)
    return float(duration_s) + 1e-6 >= max_end


def all_windows_fit(
    audio_len: int,
    sample_rate: int,
    windows: list,
) -> bool:
    """True if every [start,end) window fits in the resampled waveform (in samples)."""
    return all(
        window_fits(int(audio_len), int(sample_rate), float(s), float(e))
        for s, e in windows
    )


def fitting_windows(
    audio_len: int,
    sample_rate: int,
    windows: list,
) -> list:
    """Return the subset of windows that fully fit in the resampled waveform."""
    return [
        (float(s), float(e))
        for s, e in windows
        if window_fits(int(audio_len), int(sample_rate), float(s), float(e))
    ]

def slice_window(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    *,
    strict: bool = True,
) -> np.ndarray:
    """
    Return exact samples for [start_s, end_s).

    Pads with zeros only for tiny rounding shortfalls (~2 ms). Raises if the
    shortfall is larger when strict=True (short / truncated clips).
    """
    sr = int(sample_rate)
    duration_s = float(end_s) - float(start_s)
    if duration_s <= 0:
        raise ValueError(f"Invalid window [{start_s}, {end_s})")

    need = expected_n_samples(duration_s, sr)
    start_i = int(round(float(start_s) * sr))
    end_i = start_i + need
    pad_budget = max_pad_samples(sr)

    if start_i < 0:
        raise ValueError(f"Negative start index for window [{start_s}, {end_s})")

    if end_i <= audio.shape[0]:
        chunk = audio[start_i:end_i]
    else:
        available = audio[start_i:]
        shortfall = need - available.shape[0]
        if shortfall <= 0:
            chunk = available[:need]
        elif shortfall <= pad_budget or not strict:
            chunk = np.pad(available, (0, shortfall), mode="constant")
        else:
            raise ValueError(
                f"Window [{start_s}, {end_s}) needs {need} samples @ {sr} Hz "
                f"but only {available.shape[0]} available from index {start_i} "
                f"(audio len={audio.shape[0]}, shortfall={shortfall})."
            )

    if chunk.shape[0] != need:
        if chunk.shape[0] > need:
            chunk = chunk[:need]
        else:
            chunk = np.pad(chunk, (0, need - chunk.shape[0]), mode="constant")

    if strict and chunk.shape[0] != need:
        raise AssertionError(
            f"Expected {need} samples for duration {duration_s}s @ {sr} Hz, "
            f"got {chunk.shape[0]}"
        )
    return np.asarray(chunk, dtype=np.float32)


def write_temp_wav(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sample_rate))
    return path
