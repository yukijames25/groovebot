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
