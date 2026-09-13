#!/usr/bin/env python3
"""Smoke tests for vocal contrastive module (no GPU / checkpoint required)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from nn.contrastive import (
    LabelConfig,
    compute_contrastive_loss,
    intervals_to_encoder_frame_lists,
    read_vocal_intervals_seconds,
    sample_class_aware_negative,
    sample_negative_frame,
)


def test_interval_mapping():
    conv = [(127, 63, 1), (512, 10, 5), (512, 3, 4), (512, 3, 3), (512, 3, 2), (512, 3, 1), (512, 2, 1), (512, 2, 1)]
    sr = 24000
    wav_len = sr * 5  # 5 s clip
    intervals = [(1.0, 2.0, 3), (3.0, 3.5, 5)]
    vocal, non_vocal, classes = intervals_to_encoder_frame_lists(intervals, wav_len, sr, conv)
    assert len(vocal) == 2, vocal
    assert len(classes) == 2
    assert all(len(v) >= 2 for v in vocal)
    assert len(non_vocal) > 0
    print("OK test_interval_mapping:", [len(v) for v in vocal], "non_vocal", len(non_vocal), "classes", classes)


def test_csv_indices():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1000,5000,3\n")
        f.write("8000,12000,5\n")
        path = f.name
    cfg = LabelConfig(label_csv_format="indices", label_index_rate_hz="44100", sample_rate=24000)
    iv = read_vocal_intervals_seconds(path, cfg, true_dur_s=10.0, clip_dur_s=10.0)
    assert len(iv) == 2
    assert iv[0][0] < iv[0][1]
    assert iv[0][2] == 3
    assert iv[1][2] == 5
    print("OK test_csv_indices:", iv)


def test_sample_negative_frame_no_same_clip_other_vocal():
    rng = np.random.default_rng(0)
    # clip 0: two vocal spans; clip 1: one span — other-file negatives always available.
    vocal = [
        [[10, 11, 12, 13], [20, 21, 22]],
        [[5, 6, 7, 8]],
    ]
    non_vocal = [list(range(0, 5)), list(range(40, 50))]
    anchor_span_frames = set(vocal[0][1])
    for _ in range(200):
        neg_b, neg_frame, source = sample_negative_frame(0, 1, vocal, non_vocal, rng, noise_prob=0.0)
        if source == "other_file":
            assert neg_b == 1
            assert neg_frame in vocal[1][0]
        else:
            assert neg_frame not in anchor_span_frames
            assert neg_frame not in vocal[0][0]
    print("OK test_sample_negative_frame_no_same_clip_other_vocal")


def test_class_aware_negative_no_same_clip_other_vocal():
    rng = np.random.default_rng(1)
    vocal = [
        [[10, 11, 12, 13], [20, 21, 22, 23]],  # clip 0: class 1 and class 2
        [[5, 6, 7, 8]],
    ]
    classes = [[1, 2], [1]]
    non_vocal = [list(range(0, 5)), list(range(40, 50))]
    anchor_span_frames = set(vocal[0][0])
    other_same_clip_span = set(vocal[0][1])
    for _ in range(200):
        neg_b, neg_frame, diff_cls, source = sample_class_aware_negative(
            0, 0, 1, vocal, classes, non_vocal, rng, noise_prob=0.0
        )
        assert neg_frame not in anchor_span_frames
        assert neg_frame not in other_same_clip_span or source != "diff_class_vocal"
        if source == "diff_class_vocal":
            assert neg_b != 0
        if neg_b == 0:
            assert neg_frame not in vocal[0][0] and neg_frame not in vocal[0][1]
    print("OK test_class_aware_negative_no_same_clip_other_vocal")


def _class_aware_coin_fixture():
    # clip 0: class 1 + class 2 (same-clip other vocal must never be used)
    # clip 1: class 1 (same class — must not be used as the vocal negative)
    # clip 2: class 3 (different class — vocal branch)
    vocal = [
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [[5, 6, 7, 8]],
        [[30, 31, 32, 33]],
    ]
    classes = [[1, 2], [1], [3]]
    non_vocal = [list(range(0, 5)), list(range(40, 50)), list(range(0, 4))]
    return vocal, classes, non_vocal


def test_class_aware_negative_forced_noise():
    vocal, classes, non_vocal = _class_aware_coin_fixture()
    rng = np.random.default_rng(0)
    for _ in range(200):
        neg_b, neg_frame, diff_cls, source = sample_class_aware_negative(
            0, 0, 1, vocal, classes, non_vocal, rng, noise_prob=1.0
        )
        assert source == "noise_same_clip"
        assert diff_cls is False
        assert neg_b == 0
        assert neg_frame in non_vocal[0]
    print("OK test_class_aware_negative_forced_noise")


def test_class_aware_negative_forced_diff_class():
    vocal, classes, non_vocal = _class_aware_coin_fixture()
    rng = np.random.default_rng(0)
    for _ in range(200):
        neg_b, neg_frame, diff_cls, source = sample_class_aware_negative(
            0, 0, 1, vocal, classes, non_vocal, rng, noise_prob=0.0
        )
        assert source == "diff_class_vocal"
        assert diff_cls is True
        assert neg_b == 2
        assert neg_frame in vocal[2][0]
    print("OK test_class_aware_negative_forced_diff_class")


def test_class_aware_negative_coin_flip_mix():
    vocal, classes, non_vocal = _class_aware_coin_fixture()
    rng = np.random.default_rng(0)
    sources = []
    for _ in range(400):
        neg_b, neg_frame, diff_cls, source = sample_class_aware_negative(
            0, 0, 1, vocal, classes, non_vocal, rng, noise_prob=0.5
        )
        sources.append(source)
        assert source in {"noise_same_clip", "diff_class_vocal"}
        assert neg_frame not in set(vocal[0][0])
        assert neg_frame not in set(vocal[0][1])
        if source == "noise_same_clip":
            assert neg_b == 0 and neg_frame in non_vocal[0] and diff_cls is False
        else:
            assert neg_b == 2 and neg_frame in vocal[2][0] and diff_cls is True
    assert "noise_same_clip" in sources and "diff_class_vocal" in sources
    print("OK test_class_aware_negative_coin_flip_mix")


def test_contrastive_loss():
    b, t, d = 4, 50, 32
    s = torch.randn(b, t, d, requires_grad=True)
    vocal = [[[10, 11, 12, 13]], [[5, 6, 7]], [[20, 21, 22]], [[30, 31, 32, 33]]]
    non_vocal = [list(range(0, 5)), list(range(40, 50)), list(range(0, 4)), list(range(35, 50))]
    classes = [[3], [5], [7], [9]]
    loss, stats = compute_contrastive_loss(
        s, vocal, non_vocal, margin=0.2, vocal_span_classes=classes
    )
    assert stats["valid_triplets"] == 14
    assert "neg_diff_class_rate" in stats
    assert "neg_by_source" in stats
    assert loss.requires_grad
    loss.backward()
    print("OK test_contrastive_loss:", stats)


def test_class_aware_contrastive_loss():
    b, t, d = 3, 50, 32
    s = torch.randn(b, t, d, requires_grad=True)
    vocal = [
        [[10, 11, 12, 13]],
        [[5, 6, 7, 8]],
        [[20, 21, 22, 23]],
    ]
    classes = [[1], [2], [1]]
    non_vocal = [list(range(0, 5)), list(range(40, 50)), list(range(35, 50))]
    rng = np.random.default_rng(0)
    loss, stats = compute_contrastive_loss(
        s,
        vocal,
        non_vocal,
        margin=0.2,
        rng=rng,
        class_aware=True,
        vocal_span_classes=classes,
    )
    assert stats["valid_triplets"] == 12
    # Positives stay same-span; labels are used only for negatives.
    assert stats.get("pos_same_class", 0) == 0
    assert stats.get("pos_same_span_fallback", 0) == 12
    assert stats.get("neg_diff_class", 0) > 0
    assert stats.get("neg_by_source", {}).get("noise_same_clip", 0) > 0
    assert stats.get("neg_by_source", {}).get("diff_class_vocal", 0) > 0
    assert "per_class_sampling" in stats
    assert loss.requires_grad
    loss.backward()
    print("OK test_class_aware_contrastive_loss:", stats)


if __name__ == "__main__":
    test_interval_mapping()
    test_csv_indices()
    test_sample_negative_frame_no_same_clip_other_vocal()
    test_class_aware_negative_no_same_clip_other_vocal()
    test_class_aware_negative_forced_noise()
    test_class_aware_negative_forced_diff_class()
    test_class_aware_negative_coin_flip_mix()
    test_contrastive_loss()
    test_class_aware_contrastive_loss()
    print("All smoke tests passed.")
