"""Sanity-gate helpers for matched-window embeddings."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def print_sanity_report(report: Mapping[str, Any]) -> None:
    print("=" * 72)
    print("MATCHED-WINDOW SANITY REPORT")
    print("=" * 72)
    for key in (
        "recording_id",
        "wav_path",
        "original_waveform_length_samples",
        "original_duration_s",
        "source_sr",
    ):
        if key in report:
            print(f"  {key}: {report[key]}")

    windows = report.get("windows") or []
    for i, w in enumerate(windows):
        print(f"\n  --- window {i} ---")
        for k, v in w.items():
            print(f"    {k}: {v}")
    print("=" * 72)


def assert_matched_keys(
    a_rows: Sequence[Mapping[str, Any]],
    b_rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Abort if paired A2V / baseline rows disagree on recording_id or window bounds."""
    a_by_key = {str(r["sample_key"]): r for r in a_rows}
    b_by_key = {str(r["sample_key"]): r for r in b_rows}

    only_a = sorted(set(a_by_key) - set(b_by_key))
    only_b = sorted(set(b_by_key) - set(a_by_key))
    if only_a or only_b:
        raise RuntimeError(
            f"[{context}] sample_key mismatch.\n"
            f"  only in A ({len(only_a)}): {only_a[:5]}...\n"
            f"  only in B ({len(only_b)}): {only_b[:5]}..."
        )

    for key in a_by_key:
        a = a_by_key[key]
        b = b_by_key[key]
        if str(a["recording_id"]) != str(b["recording_id"]):
            raise RuntimeError(
                f"[{context}] Matched samples disagree on recording_id for key={key}: "
                f"A={a['recording_id']!r} B={b['recording_id']!r}. Stopping."
            )
        for field in ("window_start_s", "window_end_s"):
            try:
                if abs(float(a[field]) - float(b[field])) < 1e-6:
                    continue
            except (TypeError, ValueError):
                if str(a[field]) == str(b[field]):
                    continue
            raise RuntimeError(
                f"[{context}] Matched samples disagree on {field} for key={key}: "
                f"A={a[field]!r} B={b[field]!r}. Stopping."
            )


def collect_frame_counts(counts: Iterable[int]) -> List[int]:
    return sorted({int(c) for c in counts})
