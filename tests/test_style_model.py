"""Tests for groovebot.style.model (StyleCNN forward / shape contract)."""
from __future__ import annotations

import numpy as np
import torch

from groovebot.style.model import GENRES, MOODS, StyleCNN


def test_forward_returns_logits_for_each_head():
    model = StyleCNN(n_mels=64)
    x = torch.randn(2, 1, 64, 200)
    out = model(x)
    assert set(out.keys()) == {"genre", "mood"}
    assert out["genre"].shape == (2, len(GENRES))
    assert out["mood"].shape == (2, len(MOODS))


def test_forward_accepts_3d_input_by_unsqueezing_channel():
    model = StyleCNN(n_mels=64)
    x = torch.randn(3, 64, 150)
    out = model(x)
    assert out["genre"].shape == (3, len(GENRES))


def test_forward_rejects_bad_rank():
    model = StyleCNN(n_mels=64)
    bad = torch.randn(64, 150)  # 2-D, no batch
    try:
        model(bad)
    except ValueError as e:
        assert "shape" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on rank-2 input")


def test_predict_probs_sums_to_one():
    model = StyleCNN(n_mels=64)
    x = torch.randn(1, 1, 64, 200)
    probs = model.predict_probs(x)
    for head, p in probs.items():
        s = float(p.sum(dim=-1).item())
        assert abs(s - 1.0) < 1e-5, f"{head} probs sum {s} != 1"


def test_variable_time_dim_does_not_change_output_shape():
    model = StyleCNN(n_mels=64)
    short = model(torch.randn(1, 1, 64, 100))
    long_ = model(torch.randn(1, 1, 64, 500))
    assert short["genre"].shape == long_["genre"].shape
    assert short["mood"].shape == long_["mood"].shape


def test_smaller_n_mels_constructor():
    model = StyleCNN(n_mels=32)
    out = model(torch.randn(1, 1, 32, 100))
    assert out["genre"].shape[-1] == len(GENRES)


def test_genre_and_mood_vocab_sizes_match_spec():
    # spec §14 module note: GTZAN 10 genres, 6 mood classes
    assert len(GENRES) == 10
    assert len(MOODS) == 6
    assert set(MOODS) == {"aggressive", "happy", "sad", "calm", "dark", "epic"}
