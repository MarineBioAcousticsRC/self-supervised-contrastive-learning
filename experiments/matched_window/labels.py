"""Overlap-based window labels from NIPS temporal annotations (torch-free)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

ClassKey = Union[int, str]

_SEGMENT_WINDOW_RE = re.compile(r"_(\d{5})s_(\d{5})s$")

NIPS_CLASS_NAMES = [
    "Empty", "Aegcau_call", "Alaarv_song", "Anttri_song", "Butbut_call", "Carcan_call",
    "Carcan_song", "Carcar_call", "Carcar_song", "Cerbra_call", "Cerbra_song", "Cetcet_song",
    "Chlchl_call", "Cicatr_song", "Cicorn_song", "Cisjun_song", "Colpal_song", "Corcor_call",
    "Denmaj_call", "Denmaj_drum", "Embcir_call", "Embcir_song", "Erirub_call", "Erirub_song",
    "Fricoe_call", "Fricoe_song", "Galcri_call", "Galcri_song", "Galthe_call", "Galthe_song",
    "Gargla_call", "Hirrus_call", "Jyntor_song", "Lopcri_call", "Loxcur_call", "Lularb_song",
    "Lusmeg_call", "Lusmeg_song", "Lyrple_song", "Motcin_call", "Musstr_call", "Oriori_call",
    "Oriori_song", "Parate_call", "Parate_song", "Parcae_call", "Parcae_song", "Parmaj_call",
    "Parmaj_song", "Pasdom_call", "Pelgra_call", "Petpet_call", "Petpet_song", "Phofem_song",
    "Phycol_call", "Phycol_song", "Picpic_call", "Plaaff_song", "Plasab_song", "Poepal_call",
    "Poepal_song", "Prumod_song", "Ptehey_song", "Pyrpyr_call", "Regign_call", "Regign_song",
    "Serser_call", "Serser_song", "Siteur_call", "Siteur_song", "Strdec_song", "Strtur_song",
    "Stuvul_call", "Sylatr_call", "Sylatr_song", "Sylcan_call", "Sylcan_song", "Sylmel_call",
    "Sylmel_song", "Sylund_call", "Sylund_song", "Tetpyg_song", "Tibtom_song", "Trotro_song",
    "Turmer_call", "Turmer_song", "Turphi_call", "Turphi_song", "Unknown",
]
EMPTY_CLASS_ID = 0
UNKNOWN_CLASS_ID = len(NIPS_CLASS_NAMES) - 1
DEFAULT_EXCLUDE = ("empty", "unknown", "silence", "unk")


def class_key_to_name(class_key) -> str:
    if isinstance(class_key, int) and 0 <= class_key < len(NIPS_CLASS_NAMES):
        return NIPS_CLASS_NAMES[class_key]
    return str(class_key)


def parse_segment_offset_seconds(wav_stem: str) -> Tuple[str, float]:
    m = _SEGMENT_WINDOW_RE.search(wav_stem)
    if not m:
        return wav_stem, 0.0
    rec_stem = _SEGMENT_WINDOW_RE.sub("", wav_stem)
    return rec_stem, float(int(m.group(1)))


def resolve_label_csv(
    wav_path: Path, wav_root: Path, label_csv_dir: Optional[Path]
) -> Optional[Path]:
    stem = wav_path.stem
    rec_stem, _ = parse_segment_offset_seconds(stem)
    search_dirs: List[Path] = []
    if label_csv_dir is not None:
        search_dirs.append(Path(label_csv_dir))
    search_dirs.extend([wav_path.parent, wav_root / "csv", wav_root])
    seen = set()
    for d in search_dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        for name in (f"{stem}.csv", f"{rec_stem}.csv"):
            p = Path(d) / name
            if p.is_file():
                return p
    return None


def _split_line(line: str) -> List[str]:
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    return [p.strip() for p in line.split(",")]


def _norm(name: str) -> str:
    return name.strip().lower()


def _label_from_field(field: str):
    f = field.strip()
    if not f:
        return None, None
    try:
        cid = int(float(f))
        if 0 <= cid < len(NIPS_CLASS_NAMES):
            return NIPS_CLASS_NAMES[cid], cid
        return None, cid
    except ValueError:
        return f, None


def _parse_ts(value: str) -> Optional[float]:
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if ":" not in s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return None
    return None


def _detect_fmt(lines: Sequence[str]) -> str:
    for line in lines:
        parts = _split_line(line.strip())
        if len(parts) < 2:
            continue
        head = " ".join(parts[:3]).lower()
        if "name" in parts[0].lower() and "start" in head:
            return "audacity"
        if len(parts) >= 3:
            _, cid = _label_from_field(parts[2])
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            if cid is not None and b > a and (a >= 50 or b >= 50):
                return "indices"
            if cid is not None and 0 <= cid < len(NIPS_CLASS_NAMES) and b > a and b > 1:
                return "indices"
        try:
            float(parts[0])
            float(parts[1])
            return "seconds"
        except ValueError:
            continue
    return "seconds"


def _excluded(label_name, class_id) -> bool:
    exclude_names = {_norm(c) for c in DEFAULT_EXCLUDE}
    exclude_ids = {EMPTY_CLASS_ID, UNKNOWN_CLASS_ID}
    if class_id is not None and class_id in exclude_ids:
        return True
    if label_name:
        base = label_name.split()[0]
        if _norm(base) in exclude_names or _norm(label_name) in exclude_names:
            return True
    return False


def read_intervals(
    csv_path: Path,
    *,
    index_rate_hz: float = 44100.0,
    segment_offset_s: float = 0.0,
    clip_dur_s: Optional[float] = 5.0,
) -> List[Tuple[float, float, ClassKey]]:
    if not Path(csv_path).is_file():
        return []
    raw = [ln.strip() for ln in Path(csv_path).read_text().splitlines() if ln.strip()]
    fmt = _detect_fmt(raw)
    intervals: List[Tuple[float, float, ClassKey]] = []
    skip_header = fmt == "audacity"
    for line in raw:
        parts = _split_line(line)
        if len(parts) < 2:
            continue
        if skip_header and "name" in parts[0].lower() and "start" in " ".join(parts[:3]).lower():
            skip_header = False
            continue
        label_name = None
        class_id = None
        start_s = end_s = None
        if fmt == "audacity":
            label_name = parts[0]
            start_s = _parse_ts(parts[1])
            dur_s = _parse_ts(parts[2]) if len(parts) > 2 else None
            if start_s is None or dur_s is None or dur_s <= 0:
                continue
            end_s = start_s + dur_s
        else:
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            if len(parts) >= 3:
                label_name, class_id = _label_from_field(parts[2])
            if fmt == "indices":
                start_s = a / float(index_rate_hz)
                end_s = b / float(index_rate_hz)
            else:
                if b <= 0:
                    continue
                start_s = a
                end_s = a + b
        if start_s is None or end_s is None or end_s <= start_s:
            continue
        if _excluded(label_name, class_id):
            continue
        ck: ClassKey = class_id if class_id is not None else (
            _norm(label_name.split()[0]) if label_name else "unknown"
        )
        intervals.append((start_s, end_s, ck))

    intervals.sort(key=lambda x: x[0])
    if segment_offset_s > 0:
        intervals = [
            (s - segment_offset_s, e - segment_offset_s, ck)
            for s, e, ck in intervals
            if e > segment_offset_s and s < (clip_dur_s or float("inf")) + segment_offset_s
        ]
    if clip_dur_s is not None:
        out = []
        for s, e, ck in intervals:
            s = max(0.0, s)
            e = min(float(clip_dur_s), e)
            if e > s:
                out.append((s, e, ck))
        intervals = out
    return intervals


@dataclass
class WindowLabelResult:
    window_labels: List[str]
    dominant_species: Optional[str]
    is_single_species: bool
    overlap_by_species: dict


def window_overlap(
    window_start: float,
    window_end: float,
    event_start: float,
    event_end: float,
) -> float:
    return max(0.0, min(window_end, event_end) - max(window_start, event_start))


def frame_centers_seconds(
    window_start: float,
    window_end: float,
    n_frames: int,
) -> np.ndarray:
    """Uniform frame centers across ``[window_start, window_end]``."""
    n = int(n_frames)
    ws = float(window_start)
    we = float(window_end)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.asarray([(ws + we) * 0.5], dtype=np.float64)
    dur = we - ws
    return ws + (np.arange(n, dtype=np.float64) + 0.5) * (dur / float(n))


def frames_in_vocal_intervals(
    t_centers: np.ndarray,
    intervals: Sequence[Tuple[float, float, object]],
    *,
    half_width_s: float = 0.0025,
) -> np.ndarray:
    """True where a frame center (plus a half-hop) overlaps any vocal interval."""
    t = np.asarray(t_centers, dtype=np.float64)
    mask = np.zeros(t.shape[0], dtype=bool)
    if t.size == 0 or not intervals:
        return mask
    half = float(half_width_s)
    lo = t - half
    hi = t + half
    for es, ee, _ck in intervals:
        mask |= (hi > float(es)) & (lo < float(ee))
    return mask


def label_window(
    intervals: Sequence[Tuple[float, float, object]],
    window_start: float,
    window_end: float,
) -> WindowLabelResult:
    totals: dict = {}
    for es, ee, ck in intervals:
        ov = window_overlap(window_start, window_end, float(es), float(ee))
        if ov <= 0:
            continue
        name = class_key_to_name(ck)
        totals[name] = totals.get(name, 0.0) + float(ov)
    labels = sorted(totals.keys())
    dominant = max(totals.items(), key=lambda kv: (kv[1], kv[0]))[0] if totals else None
    return WindowLabelResult(
        window_labels=labels,
        dominant_species=dominant,
        is_single_species=len(labels) == 1,
        overlap_by_species=totals,
    )


def load_clip_intervals(
    wav_path: Path,
    wav_root: Path,
    label_csv_dir: Optional[Path],
    *,
    clip_dur_s: float = 5.0,
    sample_rate: float = 24000.0,  # unused; kept for API compatibility
) -> Tuple[List[Tuple[float, float, object]], Optional[Path]]:
    del sample_rate
    _rec, seg_offset = parse_segment_offset_seconds(wav_path.stem)
    csv_path = resolve_label_csv(wav_path, wav_root, label_csv_dir)
    if csv_path is None:
        return [], None
    intervals = read_intervals(
        csv_path,
        index_rate_hz=44100.0,
        segment_offset_s=float(seg_offset),
        clip_dur_s=float(clip_dur_s),
    )
    return intervals, csv_path


def labels_to_csv_cell(labels: Sequence[str]) -> str:
    return "|".join(labels)
