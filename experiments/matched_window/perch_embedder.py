"""Perch native 5-second mean embeddings (no spatial→1s remapping)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines_1s.embed_perch import load_perch_model  # noqa: E402
from experiments.matched_window.config import PERCH_SAMPLE_RATE  # noqa: E402


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def mean_embedding_from_outputs(outputs) -> np.ndarray:
    """Extract a single pooled Perch embedding vector from model.embed() outputs."""
    # perch-hoplite may return a namedtuple / object with .embedding, or a dict.
    if isinstance(outputs, dict):
        for key in ("embedding", "embeddings", "pooled_embedding", "output"):
            if key in outputs:
                vec = _to_numpy(outputs[key])
                break
        else:
            # Take first array-like value
            vec = _to_numpy(next(iter(outputs.values())))
    else:
        if hasattr(outputs, "embedding"):
            vec = _to_numpy(outputs.embedding)
        elif hasattr(outputs, "embeddings"):
            vec = _to_numpy(outputs.embeddings)
        else:
            vec = _to_numpy(outputs)

    vec = np.asarray(vec, dtype=np.float32)
    # Common shapes: (1, D), (D,), (n_windows, D) with n_windows==1 for 5s
    if vec.ndim == 2:
        if vec.shape[0] == 1:
            vec = vec[0]
        else:
            # Mean over windows if model somehow returned multiple
            vec = vec.mean(axis=0)
    elif vec.ndim > 2:
        vec = vec.reshape(vec.shape[0], -1).mean(axis=0)
    return np.asarray(vec, dtype=np.float32).reshape(-1)


class PerchEmbedder:
    def __init__(
        self,
        model_name: str = "perch_v2",
        *,
        window_size_s: float = 5.0,
        hop_size_s: float = 5.0,
    ) -> None:
        self.model = load_perch_model(model_name, hop_size_s=hop_size_s, window_size_s=window_size_s)
        self.sample_rate = int(getattr(self.model, "sample_rate", PERCH_SAMPLE_RATE))
        self.window_size_s = float(window_size_s)

    def embed_5s(self, audio_5s: np.ndarray) -> np.ndarray:
        """
        Embed one exact ~5s mono waveform at Perch sample rate.
        Returns shape (perch_dim,) — native pooled vector via model.embed().
        """
        audio = np.asarray(audio_5s, dtype=np.float32).reshape(-1)
        need = int(round(self.window_size_s * self.sample_rate))
        if abs(audio.shape[0] - need) > 1:
            raise ValueError(
                f"Perch expects ~{need} samples for {self.window_size_s}s @ "
                f"{self.sample_rate} Hz, got {audio.shape[0]}"
            )
        if audio.shape[0] < need:
            audio = np.pad(audio, (0, need - audio.shape[0]), mode="constant")
        elif audio.shape[0] > need:
            audio = audio[:need]

        outputs = self.model.embed(audio)
        vec = mean_embedding_from_outputs(outputs)
        if vec.ndim != 1 or vec.size == 0:
            raise RuntimeError(f"Unexpected Perch embedding shape: {vec.shape}")
        return vec
