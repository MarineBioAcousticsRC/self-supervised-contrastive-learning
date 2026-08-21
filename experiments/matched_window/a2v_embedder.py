"""Exact-length Animal2Vec dense-frame embeddings + temporal flatten."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.matched_window.config import (
    A2V_AVERAGE_TOP_K_LAYERS,
    A2V_SAMPLE_RATE,
    FrameTrimRecord,
)


@dataclass
class A2VEmbedResult:
    frames: np.ndarray  # (T, D)
    flat: np.ndarray  # (T*D,)
    num_frames: int
    embedding_dim: int
    trimmed: bool


class Animal2VecEmbedder:
    """Load a Fairseq Animal2Vec checkpoint and embed exact-length waveforms."""

    def __init__(
        self,
        checkpoint: str,
        *,
        sample_rate: int = A2V_SAMPLE_RATE,
        average_top_k_layers: int = A2V_AVERAGE_TOP_K_LAYERS,
        device: str = "cuda",
        normalize: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.average_top_k_layers = int(average_top_k_layers)
        self.normalize = bool(normalize)

        use_cuda = torch.cuda.is_available() and str(device).lower() == "cuda"
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Register custom Fairseq task/model (audio_ccas, data2vec_multi, ...)
        import nn  # noqa: F401
        from fairseq import checkpoint_utils

        models, _ = checkpoint_utils.load_model_ensemble([str(checkpoint)])
        self.model = models[0].to(self.device)
        self.model.eval()
        self._finetuned = hasattr(self.model, "w2v_encoder")

    @torch.inference_mode()
    def embed_waveform(self, audio: np.ndarray) -> np.ndarray:
        """
        Return dense frames (T, D) for a mono float32 waveform already at
        ``self.sample_rate``. Does not pad to 10s segment length.
        """
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            raise ValueError("Empty waveform")

        x = torch.from_numpy(wav).float()
        if self.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        batch = x.view(1, -1).to(self.device)

        layer_results = self.model.extract_features(source=batch)["layer_results"]
        # Each entry is typically (B, T, D) or a tuple for finetuned models.
        avg_layer = self.average_top_k_layers
        if self._finetuned:
            # Finetuned models wrap layer tensors in tuples; take [0].
            target = []
            for l in layer_results[-avg_layer:]:
                if isinstance(l, (tuple, list)):
                    target.append(l[0])
                else:
                    target.append(l)
        else:
            target = list(layer_results[-avg_layer:])

        # Normalize shapes to (B, T, D) then squeeze B.
        stacked = []
        for t in target:
            tt = t
            if tt.dim() == 2:
                # (T, D) — add batch
                tt = tt.unsqueeze(0)
            elif tt.dim() == 3:
                pass
            else:
                raise RuntimeError(f"Unexpected layer tensor shape: {tuple(tt.shape)}")
            stacked.append(tt)

        avg = (sum(stacked) / len(stacked)).float()  # (B, T, D)
        frames = avg.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if frames.ndim != 2:
            raise RuntimeError(f"Expected frames (T,D), got {frames.shape}")
        return frames

    def embed_and_flatten(self, audio: np.ndarray) -> A2VEmbedResult:
        frames = self.embed_waveform(audio)
        flat = frames.reshape(-1)
        return A2VEmbedResult(
            frames=frames,
            flat=flat,
            num_frames=int(frames.shape[0]),
            embedding_dim=int(frames.shape[1]),
            trimmed=False,
        )


def center_trim_frames(frames: np.ndarray, target_t: int) -> np.ndarray:
    """Center-trim (T, D) to target_t frames. Never pads."""
    t = int(frames.shape[0])
    target_t = int(target_t)
    if t == target_t:
        return frames
    if t < target_t:
        raise ValueError(f"Cannot center-trim {t} frames up to {target_t}")
    start = (t - target_t) // 2
    return frames[start : start + target_t]


def resolve_frame_count(
    frame_counts: Sequence[int],
    duration_s: float,
    *,
    allow_trim: bool = True,
) -> Tuple[int, FrameTrimRecord]:
    """
    Assert fixed frame count for a duration. If counts differ by model behavior,
    return min observed count and mark trimmed=True when allow_trim.
    """
    counts = [int(c) for c in frame_counts]
    unique = sorted(set(counts))
    if len(unique) == 1:
        rec = FrameTrimRecord(
            duration_s=float(duration_s),
            observed_counts=unique,
            min_frames=unique[0],
            trimmed=False,
        )
        return unique[0], rec

    if not allow_trim:
        raise AssertionError(
            f"Animal2Vec frame counts differ for duration={duration_s}s: {unique}. "
            "Verify waveform sample counts / resampling before trimming."
        )

    # Prefer consistent center-trim to minimum when difference is small.
    min_t = min(unique)
    max_t = max(unique)
    if max_t - min_t > 2:
        raise AssertionError(
            f"Animal2Vec frame counts differ too much for duration={duration_s}s: "
            f"{unique}. Fix slicing/resampling instead of trimming."
        )
    rec = FrameTrimRecord(
        duration_s=float(duration_s),
        observed_counts=unique,
        min_frames=min_t,
        trimmed=True,
    )
    return min_t, rec


def apply_optional_trim(
    frames: np.ndarray,
    target_t: Optional[int],
) -> Tuple[np.ndarray, bool]:
    if target_t is None or int(frames.shape[0]) == int(target_t):
        return frames, False
    return center_trim_frames(frames, int(target_t)), True
