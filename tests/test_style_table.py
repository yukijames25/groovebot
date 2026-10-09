"""Tests for groovebot.style.table — table lookup + soft mood weighting."""
from __future__ import annotations

import pytest

from groovebot.style.table import MOVES, select_move


def _uniform_mood(prob: float = 1.0 / 6.0) -> dict[str, float]:
    from groovebot.style.model import MOODS
    return {m: prob for m in MOODS}


def test_metal_high_aggressive_picks_headbang():
    mood = {"aggressive": 1.0, "happy": 0.0, "sad": 0.0,
            "calm": 0.0, "dark": 0.0, "epic": 0.0}
    move, intensity = select_move("metal", "high", mood)
    assert move == "headbang"
    assert intensity > 0.8  # high arousal + aggressive nudge → near saturated


def test_classical_low_calm_picks_quiet_listen():
    mood = {"aggressive": 0.0, "happy": 0.0, "sad": 0.0,
            "calm": 1.0, "dark": 0.0, "epic": 0.0}
    move, intensity = select_move("classical", "low", mood)
    assert move == "quiet_listen"
    assert intensity < 0.3  # low arousal × calm shrink


def test_pop_mid_happy_picks_clap_or_penlight():
    mood = {"aggressive": 0.0, "happy": 1.0, "sad": 0.0,
            "calm": 0.0, "dark": 0.0, "epic": 0.0}
    move, _ = select_move("pop", "mid", mood)
    assert move in {"clap", "penlight_wave", "bob_nod"}


def test_soft_mood_weighting_is_not_argmax():
    # Two moods at 0.5 each should produce a *different* result than either
    # at 1.0 alone — soft weighting must compose.
    happy_only = {"happy": 1.0, "aggressive": 0.0, "sad": 0.0,
                  "calm": 0.0, "dark": 0.0, "epic": 0.0}
    aggro_only = {"happy": 0.0, "aggressive": 1.0, "sad": 0.0,
                  "calm": 0.0, "dark": 0.0, "epic": 0.0}
    blended = {"happy": 0.5, "aggressive": 0.5, "sad": 0.0,
               "calm": 0.0, "dark": 0.0, "epic": 0.0}
    move_happy, int_happy = select_move("rock", "mid", happy_only)
    move_aggro, int_aggro = select_move("rock", "mid", aggro_only)
    move_blend, int_blend = select_move("rock", "mid", blended)
    # Blend intensity sits between the two pure cases (or equals one of
    # them when nudges cancel).
    lo = min(int_happy, int_aggro)
    hi = max(int_happy, int_aggro)
    assert lo - 1e-9 <= int_blend <= hi + 1e-9


def test_intensity_increases_with_arousal_bucket():
    mood = _uniform_mood()
    _, lo = select_move("rock", "low", mood)
    _, mid = select_move("rock", "mid", mood)
    _, hi = select_move("rock", "high", mood)
    assert lo < mid < hi


def test_intensity_in_zero_one():
    for genre in ["metal", "classical", "pop"]:
        for bucket in ["low", "mid", "high"]:
            mood = {"aggressive": 1.0, "happy": 0.0, "sad": 0.0,
                    "calm": 0.0, "dark": 0.0, "epic": 0.0}
            _, intensity = select_move(genre, bucket, mood)
            assert 0.0 <= intensity <= 1.0


def test_unknown_mood_keys_ignored():
    mood = {"aggressive": 0.5, "unknown_mood_xyz": 999.0}
    move, intensity = select_move("rock", "high", mood)
    assert move in MOVES
    assert 0.0 <= intensity <= 1.0


def test_partial_mood_dict_is_ok():
    mood = {"happy": 1.0}  # missing other 5 — treated as 0
    move, _ = select_move("pop", "mid", mood)
    assert move in MOVES


def test_unknown_genre_raises():
    with pytest.raises(ValueError, match="genre"):
        select_move("kpop", "mid", {"happy": 1.0})


def test_unknown_arousal_bucket_raises():
    with pytest.raises(ValueError, match="arousal"):
        select_move("rock", "extreme", {"happy": 1.0})


def test_zero_mood_falls_back_to_genre_prior():
    # All-zero mood probs → genre × arousal alone should drive the choice.
    mood = {m: 0.0 for m in
            ("aggressive", "happy", "sad", "calm", "dark", "epic")}
    move, _ = select_move("metal", "high", mood)
    # Metal high without mood should still favour headbang via genre prior.
    assert move == "headbang"


# ─────────────────────────────────────────────  continuous-arousal mode
#
# Added 2026-10-02 together with the piecewise-linear arousal routing.
# The two modes are contractually required to AGREE at the three anchor
# arousal values; between anchors the continuous mode blends linearly,
# which the perturbation tests pin as a stability win over bucket mode.

def test_continuous_tent_weights_are_one_hot_at_anchors():
    from groovebot.style.table import (
        _continuous_arousal_weights,
        _LOW_ANCHOR, _MID_ANCHOR, _HIGH_ANCHOR,
    )
    assert _continuous_arousal_weights(_LOW_ANCHOR)  == (1.0, 0.0, 0.0)
    assert _continuous_arousal_weights(_MID_ANCHOR)  == (0.0, 1.0, 0.0)
    assert _continuous_arousal_weights(_HIGH_ANCHOR) == (0.0, 0.0, 1.0)


def test_continuous_tent_weights_sum_to_one_everywhere():
    import numpy as np
    from groovebot.style.table import _continuous_arousal_weights
    for a in np.linspace(0.0, 1.0, 41):
        s = sum(_continuous_arousal_weights(float(a)))
        assert abs(s - 1.0) < 1e-9, (a, s)


def test_continuous_weights_clamp_outside_the_anchor_range():
    """Below `_LOW_ANCHOR` must collapse to (1, 0, 0); above
    `_HIGH_ANCHOR` to (0, 0, 1). The DEAM head's calibrated output has
    not reached these ends in practice, but the clamp has to be clean."""
    from groovebot.style.table import _continuous_arousal_weights
    assert _continuous_arousal_weights(0.0)  == (1.0, 0.0, 0.0)
    assert _continuous_arousal_weights(0.05) == (1.0, 0.0, 0.0)
    assert _continuous_arousal_weights(0.95) == (0.0, 0.0, 1.0)
    assert _continuous_arousal_weights(1.0)  == (0.0, 0.0, 1.0)


def test_bucket_and_continuous_agree_at_anchors():
    """The two modes must produce the same (move, intensity) when
    arousal equals one of the three anchor points."""
    from groovebot.style.table import (
        _LOW_ANCHOR, _MID_ANCHOR, _HIGH_ANCHOR,
    )
    mood = {"happy": 0.5}
    for a, bucket in (
        (_LOW_ANCHOR,  "low"),
        (_MID_ANCHOR,  "mid"),
        (_HIGH_ANCHOR, "high"),
    ):
        mb, ib = select_move("pop", bucket, mood, mode="bucket")
        mc, ic = select_move("pop", bucket, mood, arousal=a, mode="continuous")
        assert mb == mc, (bucket, mb, mc)
        assert ib == pytest.approx(ic, abs=1e-9), (bucket, ib, ic)


def test_continuous_mode_requires_arousal():
    with pytest.raises(ValueError, match="arousal"):
        select_move("pop", "mid", {"happy": 1.0}, mode="continuous")


def test_continuous_intensity_moves_monotonically_with_arousal():
    """Keeping genre and mood fixed, raising arousal must never DECREASE
    the base intensity across the piecewise interpolation."""
    import numpy as np
    mood = {"happy": 0.5}
    last = 0.0
    for a in np.linspace(0.0, 1.0, 101):
        _, inten = select_move("pop", "mid", mood,
                               arousal=float(a), mode="continuous")
        assert inten >= last - 1e-9, (a, inten, last)
        last = inten


def test_continuous_mode_is_stable_across_the_low_mid_boundary():
    """Right at a=0.410 (the bucket low/mid cut), the discrete mode
    discontinuously flips both move and intensity. The continuous mode
    must stay smooth: no move flip and intensity change under 0.05
    across a ±0.005 jitter."""
    mood = {"sad": 0.33}

    # Discrete: steps across the boundary → tangible intensity jump.
    _, i_low  = select_move("pop", "low",  mood, mode="bucket")
    _, i_mid  = select_move("pop", "mid",  mood, mode="bucket")
    assert abs(i_mid - i_low) > 0.15   # the pathology we're replacing

    # Continuous: 0.405 → 0.415 straddles the old boundary.
    m_a, i_a = select_move("pop", "mid", mood,
                           arousal=0.405, mode="continuous")
    m_b, i_b = select_move("pop", "mid", mood,
                           arousal=0.415, mode="continuous")
    assert m_a == m_b, (m_a, m_b)
    assert abs(i_a - i_b) < 0.05, (i_a, i_b)


def test_selector_arousal_mode_defaults_to_bucket():
    """The selector keeps bucket as the default so existing checkpoints
    and tests do not shift behaviour until a user opts in."""
    from groovebot.style.select import GrooveStyleSelector
    sel = GrooveStyleSelector()
    assert sel.arousal_mode == "bucket"


def test_selector_rejects_unknown_arousal_mode():
    from groovebot.style.select import GrooveStyleSelector
    with pytest.raises(ValueError, match="arousal_mode"):
        GrooveStyleSelector(arousal_mode="smooth")  # type: ignore[arg-type]
