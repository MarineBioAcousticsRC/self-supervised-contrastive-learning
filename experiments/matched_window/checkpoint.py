"""Discover a self-supervised contrastive animal2vec checkpoint under a search root."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

_VOCAL_DIR_RE = re.compile(r"vocal.?contrastive|ssl.?contrastive|self.?supervised.?contrastive", re.IGNORECASE)
PREFERRED_NAMES = ("checkpoint_last.pt", "checkpoint_best.pt")


def discover_ssl_contrastive_checkpoint(
    search_root: Path,
    explicit: Optional[Path] = None,
) -> Path:
    """Return explicit path if given, else newest matching checkpoint under search_root."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"A2V checkpoint not found: {p}")
        return p

    root = Path(search_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint search root not found: {root}")

    candidates: List[Path] = []
    for pt in root.rglob("*.pt"):
        # Prefer directories whose name mentions vocal contrastive.
        parts = {seg.lower() for seg in pt.parts}
        path_str = str(pt)
        if not (
            _VOCAL_DIR_RE.search(path_str)
            or any("vocal" in s and "contrast" in s for s in parts)
            or "ssl_contrastive" in path_str.lower()
            or "self_supervised_contrastive" in path_str.lower()
        ):
            continue
        candidates.append(pt)

    if not candidates:
        # Fallback: any checkpoint_last.pt under Results with "contrastive" in path.
        for pt in root.rglob("checkpoint_last.pt"):
            if "contrastive" in str(pt).lower() or "vocal" in str(pt).lower():
                candidates.append(pt)

    if not candidates:
        raise FileNotFoundError(
            f"No self-supervised contrastive checkpoint found under {root}. "
            "Pass --a2v-checkpoint explicitly."
        )

    def rank(p: Path):
        # Prefer same-span dense run over class_aware; then preferred filenames; then newest.
        path_l = str(p).lower()
        if "self_supervised_contrastive" in path_l or "ssl_contrastive" in path_l:
            dir_rank = 0
        elif "nips_vocal_contrastive_dense" in path_l and "class_aware" not in path_l:
            dir_rank = 1
        elif "vocal_contrastive" in path_l and "class_aware" not in path_l:
            dir_rank = 2
        else:
            dir_rank = 3
        if p.name in PREFERRED_NAMES:
            name_rank = PREFERRED_NAMES.index(p.name)
        else:
            name_rank = 100
        try:
            mtime = -p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (dir_rank, name_rank, mtime, str(p))

    candidates.sort(key=rank)
    return candidates[0].resolve()
