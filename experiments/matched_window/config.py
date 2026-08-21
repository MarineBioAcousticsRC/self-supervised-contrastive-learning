"""Constants and output layout for matched-window experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ONE_SECOND_WINDOWS: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
]
BIRDNET_WINDOWS: List[Tuple[float, float]] = [
    (0.0, 3.0),
    (2.0, 5.0),
]
PERCH_WINDOWS: List[Tuple[float, float]] = [
    (0.0, 5.0),
]

EXPECTED_CLIP_DURATION_S = 5.0
# Exp 2/3 (and strict full-window protocols): require enough audio for max window end.
# Exp 1 uses fit-what-fits (every complete 1s window); see embed.run_a2v.
MIN_CLIP_DURATION_S = 5.0
# Exp 1: skip only if no complete 1s window fits.
MIN_EXP1_CLIP_DURATION_S = 1.0
RANDOM_SEED = 42

A2V_SAMPLE_RATE = 24000
BIRDNET_SAMPLE_RATE = 48000
PERCH_SAMPLE_RATE = 32000

A2V_AVERAGE_TOP_K_LAYERS = 12
A2V_EMBEDDING_DIM = 768

UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

KMEANS_N_INIT = 10
KMEANS_BATCH_SIZE = 128
KMEANS_INIT = "k-means++"
# Same defaults as scripts/kmeans_sweep.py
KMEANS_K_MIN = 10
KMEANS_K_MAX = 200
KMEANS_K_STEP = 10

DEFAULT_WAV_DIR = None
DEFAULT_OUT_DIR = None
DEFAULT_CHECKPOINT_SEARCH_ROOT = None


@dataclass
class FrameTrimRecord:
    duration_s: float
    observed_counts: List[int]
    min_frames: int
    trimmed: bool


@dataclass
class ExperimentConfig:
    wav_dir: Path = DEFAULT_WAV_DIR
    label_csv_dir: Path | None = None
    out_dir: Path = DEFAULT_OUT_DIR
    a2v_checkpoint: Path | None = None
    checkpoint_search_root: Path = DEFAULT_CHECKPOINT_SEARCH_ROOT
    random_seed: int = RANDOM_SEED
    device: str = "cuda"
    sanity_n: int | None = None
    a2v_frame_trim: Dict[str, FrameTrimRecord] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for sub in ("embeddings", "metadata", "umap", "results/umap_visualizations"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

    def emb_path(self, name: str) -> Path:
        return self.out_dir / "embeddings" / f"{name}.npy"

    def meta_path(self, name: str) -> Path:
        return self.out_dir / "metadata" / f"{name}.csv"

    def umap_path(self, name: str) -> Path:
        return self.out_dir / "umap" / f"{name}_umap.joblib"

    def trim_config_path(self) -> Path:
        return self.out_dir / "a2v_frame_trim.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wav_dir"] = str(self.wav_dir)
        d["label_csv_dir"] = str(self.label_csv_dir) if self.label_csv_dir else None
        d["out_dir"] = str(self.out_dir)
        d["a2v_checkpoint"] = str(self.a2v_checkpoint) if self.a2v_checkpoint else None
        d["checkpoint_search_root"] = str(self.checkpoint_search_root)
        return d


def sample_key(recording_id: str, start_s: float, end_s: float) -> str:
    return f"{recording_id}_{start_s:g}_{end_s:g}"
