"""Tests for the v3 StyleHead (MLP on a frozen embedding)."""
from __future__ import annotations

import torch

from groovebot.style.model import GENRES, MOODS, StyleHead


def test_forward_returns_logits_for_each_head():
    head = StyleHead(emb_dim=2048)
    x = torch.randn(3, 2048)
    out = head(x)
    assert set(out.keys()) == {"genre", "mood"}
    assert out["genre"].shape == (3, len(GENRES))
    assert out["mood"].shape == (3, len(MOODS))


def test_forward_unsqueezes_single_embedding():
    head = StyleHead(emb_dim=2048)
    x = torch.randn(2048)
    out = head(x)
    assert out["genre"].shape == (1, len(GENRES))


def test_predict_probs_sums_to_one():
    head = StyleHead(emb_dim=2048)
    probs = head.predict_probs(torch.randn(2, 2048))
    for k, p in probs.items():
        s = p.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_emb_dim_mismatch_raises():
    head = StyleHead(emb_dim=2048)
    try:
        head(torch.randn(2, 512))
    except ValueError as e:
        assert "emb_dim" in str(e)
    else:
        raise AssertionError("expected ValueError on emb_dim mismatch")


def test_dropout_train_mode_keeps_logits_finite():
    head = StyleHead(emb_dim=2048, hidden=128, dropout=0.7)
    head.train()
    out = head(torch.randn(4, 2048))
    assert torch.isfinite(out["genre"]).all()
    assert torch.isfinite(out["mood"]).all()


def test_smaller_emb_dim_supported():
    head = StyleHead(emb_dim=512, hidden=64)
    out = head(torch.randn(2, 512))
    assert out["genre"].shape == (2, len(GENRES))
