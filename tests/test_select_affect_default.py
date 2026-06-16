"""Tests for the DEAM-learned affect default + mood_source switch
in GrooveStyleSelector.

The PANNs backbone is mocked everywhere so the suite runs without the
340 MB checkpoint. The mock embeds a query into a deterministic
2048-d vector keyed by audio RMS; the heads run real PyTorch forward
passes on top, so the wiring assertions are end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from groovebot.style.attributes import estimate_arousal
from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.deam import DEAM_SAM_HI, DEAM_SAM_LO
from groovebot.style.model import (
    GENRES, MOODS, StyleCNN, StyleHead, StyleRegressionHead,
)
from groovebot.style.mood_from_va import (
    DEFAULT_QUADRANT_PROTOTYPES,
    PROTOTYPES_WITH_AUX,
    MoodPrototype,
    mood_probs_from_va,
)
from groovebot.style.select import GrooveStyleSelector
from groovebot.style.table import MOVES


SR = 22050


def _synth_click_track(sr: int, duration_sec: float, bpm: float) -> np.ndarray:
    n = int(sr * duration_sec)
    sig = np.zeros(n, dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    click_len = max(1, sr // 200)
    for start in range(0, n, period):
        end = min(n, start + click_len)
        sig[start:end] = 1.0
    return sig


def _mock_panns_emb(audio, sr):
    """Deterministic 2048-d embedding seeded from audio RMS — gives
    reproducible head outputs across runs."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-9))
    rng = np.random.default_rng(int(rms * 1e6) % (2**31))
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def _v3_selector(
    *,
    regression_head: StyleRegressionHead | None = None,
    arousal_fn=None,
    valence_fn=None,
    mood_source="head",
    mood_va_prototypes=DEFAULT_QUADRANT_PROTOTYPES,
):
    backbone = PannsBackbone(_inference_fn=_mock_panns_emb)
    head = StyleHead(emb_dim=EMBEDDING_DIM)
    return GrooveStyleSelector(
        backbone=backbone, head=head,
        regression_head=regression_head,
        arousal_fn=arousal_fn,
        valence_fn=valence_fn,
        mood_source=mood_source,
        mood_va_prototypes=mood_va_prototypes,
    )


# ----------------------------------------------------------- affect default

def test_no_regression_head_falls_back_to_heuristic_arousal():
    """v3 selector without `regression_head` must use the v2 heuristic
    (the spec's fallback path)."""
    sel = _v3_selector()
    audio = _synth_click_track(SR, 4.0, 130.0)
    style = sel.select(audio, SR)
    expected = max(0.0, min(1.0, estimate_arousal(audio, SR)))
    assert style.arousal == pytest.approx(expected, abs=1e-6)


def test_regression_head_becomes_default_arousal_source():
    """When a regression head is wired in and no explicit arousal_fn,
    the learned head produces the arousal — NOT the heuristic."""
    reg = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    sel = _v3_selector(regression_head=reg)
    audio = _synth_click_track(SR, 4.0, 130.0)
    style = sel.select(audio, SR)
    # Must differ from heuristic (random head ≠ RMS×onset density).
    # We assert it's in 0..1 and is bit-equal to what we'd compute
    # directly through the head + DEAM calibrator.
    assert 0.0 <= style.arousal <= 1.0
    backbone = sel.backbone
    emb = backbone.embed(audio, SR)
    x = torch.from_numpy(emb).unsqueeze(0)
    with torch.no_grad():
        a_raw = float(reg(x)["arousal"].item())
    from groovebot.style.deam import sam_to_unit
    assert style.arousal == pytest.approx(sam_to_unit(a_raw), abs=1e-6)


def test_explicit_arousal_fn_overrides_regression_head_default():
    """An explicit arousal_fn must win over the regression-head default
    (lets users mix sources during ablation)."""
    reg = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    sel = _v3_selector(
        regression_head=reg, arousal_fn=lambda a, sr: 0.314,
    )
    style = sel.select(_synth_click_track(SR, 4.0, 130.0), SR)
    assert style.arousal == pytest.approx(0.314)


def test_arousal_and_valence_clamp_to_unit_interval():
    """The selector must clamp affect to 0..1 before bucket / VA map
    (a hostile regression head returning out-of-range values must not
    crash arousal_bucket())."""
    sel_high = _v3_selector(
        arousal_fn=lambda a, sr: 12.0, valence_fn=lambda a, sr: -5.0,
        mood_source="va",
    )
    style = sel_high.select(_synth_click_track(SR, 3.0, 120.0), SR)
    assert style.arousal == 1.0
    # mood from VA at (V=0, A=1) -> aggressive
    assert style.mood == "aggressive"


# ----------------------------------------------- mood_source switch

def test_mood_source_head_keeps_head_softmax():
    sel = _v3_selector()
    style = sel.select(_synth_click_track(SR, 4.0, 130.0), SR)
    # Mood is whatever the random head argmaxes to (vocabulary check
    # only; no accuracy claim)
    assert style.mood in MOODS
    assert abs(sum(style.mood_probs.values()) - 1.0) < 1e-4


def test_mood_source_va_returns_va_distribution():
    """mood_source='va' replaces the head softmax with the V/A map.
    The probs must match `mood_probs_from_va` exactly."""
    reg = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    sel = _v3_selector(regression_head=reg, mood_source="va")
    audio = _synth_click_track(SR, 4.0, 130.0)
    style = sel.select(audio, SR)
    expected = mood_probs_from_va(
        style.arousal, _valence_from_reg(reg, sel.backbone, audio, SR),
    )
    for m in MOODS:
        assert style.mood_probs[m] == pytest.approx(expected[m], abs=1e-6)


def test_mood_source_va_uses_explicit_aux_prototypes():
    """Passing PROTOTYPES_WITH_AUX must let epic/dark fire (the
    default 4-quadrant prototypes leave them at 0)."""
    reg = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    sel_default = _v3_selector(regression_head=reg, mood_source="va")
    sel_aux = _v3_selector(
        regression_head=reg, mood_source="va",
        mood_va_prototypes=PROTOTYPES_WITH_AUX,
    )
    audio = _synth_click_track(SR, 4.0, 130.0)
    s_d = sel_default.select(audio, SR)
    s_a = sel_aux.select(audio, SR)
    assert s_d.mood_probs["epic"] == 0.0
    assert s_d.mood_probs["dark"] == 0.0
    assert s_a.mood_probs["epic"] > 0.0
    assert s_a.mood_probs["dark"] > 0.0


def test_mood_source_va_requires_valence_source_at_construction():
    """mood_source='va' without a regression_head and without a
    valence_fn must error early (clear feedback to the caller)."""
    with pytest.raises(ValueError, match="valence"):
        GrooveStyleSelector(mood_source="va")


def test_mood_source_va_accepts_explicit_valence_fn():
    """Without PANNs, an explicit valence_fn is also valid."""
    sel = GrooveStyleSelector(
        mood_source="va",
        arousal_fn=lambda a, sr: 0.9,   # high arousal
        valence_fn=lambda a, sr: 0.1,   # low valence -> aggressive
    )
    style = sel.select(_synth_click_track(SR, 3.0, 120.0), SR)
    assert style.mood == "aggressive"


@pytest.mark.parametrize(
    "arousal_const,valence_const,expected_mood",
    [
        (0.95, 0.95, "happy"),
        (0.95, 0.05, "aggressive"),
        (0.05, 0.95, "calm"),
        (0.05, 0.05, "sad"),
    ],
)
def test_va_quadrants_drive_table_through_full_select(
    arousal_const, valence_const, expected_mood,
):
    """End-to-end: fixed (A, V) into the v1/v2 CNN selector with the
    VA mood source must surface the correct quadrant mood, and the
    table.select_move call must produce a valid move."""
    sel = GrooveStyleSelector(
        mood_source="va",
        arousal_fn=lambda a, sr: arousal_const,
        valence_fn=lambda a, sr: valence_const,
    )
    style = sel.select(_synth_click_track(SR, 3.0, 120.0), SR)
    assert style.mood == expected_mood
    assert style.move in MOVES


# ----------------------------------------------- guard rails / validation

def test_regression_head_without_v3_path_rejected():
    """Passing a regression head with the v1/v2 CNN selector is a
    construction error — the user needs to wire backbone+head too."""
    with pytest.raises(ValueError, match="v3"):
        GrooveStyleSelector(
            regression_head=StyleRegressionHead(emb_dim=EMBEDDING_DIM),
        )


def test_embedding_shared_across_head_and_regression_head(monkeypatch):
    """When regression_head co-wired with backbone+head, the backbone
    must embed exactly once per select() call (perf-critical for the
    M2 realtime loop)."""
    reg = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
    sel = _v3_selector(regression_head=reg)
    calls = {"n": 0}
    real_embed = sel.backbone.embed

    def counting_embed(audio, sr):
        calls["n"] += 1
        return real_embed(audio, sr)

    monkeypatch.setattr(sel.backbone, "embed", counting_embed)
    sel.select(_synth_click_track(SR, 4.0, 130.0), SR)
    assert calls["n"] == 1


# --------------------------------------------------------------- helpers

def _valence_from_reg(reg, backbone, audio, sr):
    from groovebot.style.deam import sam_to_unit
    emb = backbone.embed(audio, sr)
    x = torch.from_numpy(emb).unsqueeze(0)
    with torch.no_grad():
        v_raw = float(reg(x)["valence"].item())
    return max(0.0, min(1.0, sam_to_unit(v_raw)))
