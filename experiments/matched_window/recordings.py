"""List NIPS4Bplus train segment wavs and resolve recording IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from experiments.matched_window.labels import parse_segment_offset_seconds

_SEGMENT_WAV_RE = re.compile(
    r"^(?P<stem>.+)_(?P<start>\d{5})s_(?P<end>\d{5})s$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecordingClip:
    wav_path: Path
    recording_id: str
    segment_offset_s: float
    segment_end_s: float
    is_train: bool
    is_test: bool


def is_train_filename(name: str) -> bool:
    n = name.lower()
    return "trainfile" in n and "testfile" not in n


def is_test_filename(name: str) -> bool:
    return "testfile" in name.lower()


def list_train_wavs(
    wav_dir: Path,
    *,
    limit: Optional[int] = None,
    glob_pattern: str = "*.wav",
) -> List[RecordingClip]:
    """Return sorted train segment clips (exclude testfile*)."""
    root = Path(wav_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {root}")

    paths = sorted(root.glob(glob_pattern))
    if glob_pattern == "*.wav":
        paths = sorted(set(paths) | set(root.glob("*.WAV")))

    clips: List[RecordingClip] = []
    for p in paths:
        if not p.is_file():
            continue
        if not is_train_filename(p.name):
            continue
        stem = p.stem
        rec_id, offset_s = parse_segment_offset_seconds(stem)
        m = _SEGMENT_WAV_RE.match(stem)
        end_s = float(int(m.group("end"))) if m else offset_s + 5.0
        clips.append(
            RecordingClip(
                wav_path=p,
                recording_id=rec_id,
                segment_offset_s=float(offset_s),
                segment_end_s=float(end_s),
                is_train=True,
                is_test=False,
            )
        )

    if limit is not None:
        clips = clips[: max(0, int(limit))]
    return clips


def assert_clips_nonempty(clips: Sequence[RecordingClip], wav_dir: Path) -> None:
    if not clips:
        raise RuntimeError(
            f"No trainfile*.wav found under {wav_dir}. "
            "Expected NIPS segment exports like nips4b_birds_trainfile685_00000s_00005s.wav"
        )
