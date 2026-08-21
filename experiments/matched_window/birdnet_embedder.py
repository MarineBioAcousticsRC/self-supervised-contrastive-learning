"""BirdNET native 3-second embeddings for matched windows."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines_1s.embed_birdnet import (  # noqa: E402
    extract_embeddings_birdnetlib,
    load_analyzer,
)
from experiments.matched_window.audio_slices import write_temp_wav  # noqa: E402
from experiments.matched_window.config import BIRDNET_SAMPLE_RATE  # noqa: E402


class BirdNETEmbedder:
    def __init__(self, version: Optional[str] = None) -> None:
        self.analyzer = load_analyzer(version)
        self.sample_rate = BIRDNET_SAMPLE_RATE

    def embed_3s(self, audio_3s: np.ndarray) -> np.ndarray:
        """
        Embed one exact 3-second mono waveform @ 48 kHz.
        Returns shape (birdnet_dim,) — one native vector, no temporal split.
        """
        audio = np.asarray(audio_3s, dtype=np.float32).reshape(-1)
        need = int(round(3.0 * self.sample_rate))
        if audio.shape[0] != need:
            raise ValueError(
                f"BirdNET expects {need} samples for 3s @ {self.sample_rate} Hz, "
                f"got {audio.shape[0]}"
            )

        with tempfile.TemporaryDirectory(prefix="birdnet_mw_") as td:
            tmp = Path(td) / "window_3s.wav"
            write_temp_wav(audio, self.sample_rate, tmp)
            # hop == window ⇒ one window, no overlap
            embs, _centers = extract_embeddings_birdnetlib(
                self.analyzer, tmp, window_s=3.0, hop_s=3.0
            )

        if embs.ndim != 2 or embs.shape[0] < 1:
            raise RuntimeError(f"BirdNET returned unexpected shape {getattr(embs, 'shape', None)}")
        if embs.shape[0] != 1:
            # If library still returned multiple, take the first full window.
            embs = embs[:1]
        return np.asarray(embs[0], dtype=np.float32).reshape(-1)
