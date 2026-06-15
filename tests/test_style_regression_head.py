"""Tests for v3 StyleRegressionHead (arousal/valence on PANNs embedding)."""
from __future__ import annotations

import torch

from groovebot.style.model import REGRESSION_TARGETS, StyleRegressionHead


def test_forward_returns_one_scalar_per_target():
    head = StyleRegressionHead(emb_dim=2048)
    out = head(torch.randn(4, 2048))
    assert set(out.keys()) == set(REGRESSION_TARGETS)
    for k, v in out.items():
        assert v.shape == (4,), f"{k}: got {v.shape}"


def test_forward_unsqueezes_single_embedding():
    head = StyleRegressionHead(emb_dim=2048)
    out = head(torch.randn(2048))
    for k, v in out.items():
        assert v.shape == (1,)


def test_predict_disables_dropout():
    head = StyleRegressionHead(emb_dim=2048, dropout=0.9)
    head.train()
    x = torch.randn(8, 2048)
    a = head.predict(x)
    b = head.predict(x)
    for k in a:
        assert torch.allclose(a[k], b[k])


def test_emb_dim_mismatch_raises():
    head = StyleRegressionHead(emb_dim=2048)
    try:
        head(torch.randn(2, 256))
    except ValueError as e:
        assert "emb_dim" in str(e)
    else:
        raise AssertionError("expected ValueError on emb_dim mismatch")


def test_can_train_a_constant_target():
    """Sanity that the loss path works: regressing the mean of a
    1-d signal converges in a few steps to the constant."""
    torch.manual_seed(0)
    head = StyleRegressionHead(emb_dim=64, hidden=32, dropout=0.0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    x = torch.randn(64, 64)
    y_arousal = torch.full((64,), 5.0)
    y_valence = torch.full((64,), 6.0)
    for _ in range(60):
        out = head(x)
        loss = (
            torch.nn.functional.mse_loss(out["arousal"], y_arousal)
            + torch.nn.functional.mse_loss(out["valence"], y_valence)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = head(x)
    assert out["arousal"].mean().item() == \
        __import__("pytest").approx(5.0, abs=0.4)
    assert out["valence"].mean().item() == \
        __import__("pytest").approx(6.0, abs=0.4)


def test_custom_targets_supported():
    head = StyleRegressionHead(emb_dim=128, targets=("arousal",))
    out = head(torch.randn(3, 128))
    assert set(out.keys()) == {"arousal"}
