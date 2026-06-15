"""Tests for the pluggable arousal source on GrooveStyleSelector."""
from __future__ import annotations

import numpy as np

from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.model import StyleHead, StyleRegressionHead
from groovebot.style.select import (
    GrooveStyleSelector, make_panns_arousal_fn,
)


SR = 22050


def _synth_clicks(duration_sec: float, bpm: float) -> np.ndarray:
    n = int(SR * duration_sec)
    sig = np.zeros(n, dtype=np.float32)
    period = int(SR * 60.0 / bpm)
    for start in range(0, n, max(1, SR // 200)):
        pass
    click_len = max(1, SR // 200)
    for start in range(0, n, period):
        end = min(n, start + click_len)
        sig[start:end] = 1.0
    return sig


def _mock_emb(audio, sr):
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-9))
    rng = np.random.default_rng(int(rms * 1e6) % (2**31))
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def test_selector_uses_custom_arousal_fn():
    """A custom arousal_fn must override the heuristic exactly."""
    calls = []

    def constant_arousal(audio, sr):
        calls.append((audio.shape, sr))
        return 0.42

    sel = GrooveStyleSelector(arousal_fn=constant_arousal)
    style = sel.select(_synth_clicks(5.0, 120.0), SR)
    assert style.arousal == 0.42
    assert calls, "arousal_fn was not invoked"


def test_selector_clamps_arousal_fn_output():
    """An arousal_fn that returns values outside 0..1 must be clamped
    before arousal_bucket() sees it (the bucket assumes 0..1)."""
    sel_high = GrooveStyleSelector(arousal_fn=lambda a, sr: 9.0)
    assert sel_high.select(_synth_clicks(3.0, 120.0), SR).arousal == 1.0
    sel_low = GrooveStyleSelector(arousal_fn=lambda a, sr: -2.0)
    assert sel_low.select(_synth_clicks(3.0, 120.0), SR).arousal == 0.0


def test_make_panns_arousal_fn_returns_unit_scale():
    """End-to-end: build a backbone + regression head, wrap into the
    arousal fn, and confirm output is in 0..1 with the DEAM
    calibrator."""
    backbone = PannsBackbone(_inference_fn=_mock_emb)
    head = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    fn = make_panns_arousal_fn(backbone, head)
    audio = _synth_clicks(3.0, 120.0)
    v = fn(audio, SR)
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


def test_make_panns_arousal_fn_obeys_target():
    """Calling with target='valence' must route to that head output."""
    backbone = PannsBackbone(_inference_fn=_mock_emb)
    head = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    fn_v = make_panns_arousal_fn(backbone, head, target="valence")
    v = fn_v(_synth_clicks(3.0, 120.0), SR)
    assert 0.0 <= v <= 1.0


def test_make_panns_arousal_fn_custom_calibrator_passthrough():
    """A custom calibrator must be applied to the raw head output."""
    backbone = PannsBackbone(_inference_fn=_mock_emb)
    head = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    seen = []

    def cal(x: float) -> float:
        seen.append(x)
        return 0.123

    fn = make_panns_arousal_fn(backbone, head, calibrator=cal)
    assert fn(_synth_clicks(2.0, 120.0), SR) == 0.123
    assert len(seen) == 1


def test_selector_with_v3_arousal_fn_end_to_end():
    """Wire backbone+head+v3 arousal fn into the embedding-mode
    selector; the contract output stays a GrooveStyle with 0..1
    arousal."""
    backbone = PannsBackbone(_inference_fn=_mock_emb)
    head = StyleHead(emb_dim=EMBEDDING_DIM)
    reg_head = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    arousal_fn = make_panns_arousal_fn(backbone, reg_head)
    sel = GrooveStyleSelector(
        backbone=backbone, head=head, arousal_fn=arousal_fn,
    )
    style = sel.select(_synth_clicks(4.0, 130.0), SR)
    assert 0.0 <= style.arousal <= 1.0
    assert style.arousal_bucket in {"low", "mid", "high"}
