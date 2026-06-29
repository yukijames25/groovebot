"""Tests for groovebot.style.mood_from_va (V/A → mood mapping)."""
from __future__ import annotations

import pytest

from groovebot.style.model import MOODS
from groovebot.style.mood_from_va import (
    DEFAULT_QUADRANT_PROTOTYPES,
    DRAFT_AUX_PROTOTYPES,
    PROTOTYPES_WITH_AUX,
    MoodPrototype,
    dominant_mood_from_va,
    mood_probs_from_va,
    quadrant_label,
)


# ---------------------------------------------------------------------- shape

def test_default_probs_sum_to_one_and_cover_all_moods():
    probs = mood_probs_from_va(0.5, 0.5)
    assert set(probs.keys()) == set(MOODS)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_aux_disabled_by_default_means_epic_and_dark_are_zero():
    probs = mood_probs_from_va(0.5, 0.5)
    assert probs["epic"] == 0.0
    assert probs["dark"] == 0.0


def test_aux_opt_in_populates_epic_and_dark():
    probs = mood_probs_from_va(0.5, 0.5, prototypes=PROTOTYPES_WITH_AUX)
    assert probs["epic"] > 0.0
    assert probs["dark"] > 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- 4 quadrants

@pytest.mark.parametrize(
    "valence,arousal,expected",
    [
        (1.0, 1.0, "happy"),
        (0.0, 1.0, "aggressive"),
        (1.0, 0.0, "calm"),
        (0.0, 0.0, "sad"),
    ],
)
def test_quadrant_corners_argmax_to_their_mood(valence, arousal, expected):
    probs = mood_probs_from_va(arousal, valence)
    assert max(probs, key=probs.get) == expected
    assert probs[expected] > 0.4  # corner gets dominant mass


def test_center_point_is_roughly_uniform_over_quadrants():
    """A query at the calibrated V/A center should not strongly prefer
    any one of the 4 quadrants (within ~5pp of uniform). The center is
    re-mapped to (0.5, 0.5) by the calibration loader, so testing with
    `recenter=False` at (0.5, 0.5) hits the same geometric point."""
    probs = mood_probs_from_va(0.5, 0.5, recenter=False)
    quad = ["happy", "aggressive", "calm", "sad"]
    for m in quad:
        assert probs[m] == pytest.approx(0.25, abs=0.05)


def test_clamps_out_of_range_inputs():
    """Inputs outside 0..1 are clamped; the function does not raise."""
    a_low = mood_probs_from_va(-5.0, -5.0)
    a_high = mood_probs_from_va(9.0, 9.0)
    assert max(a_low, key=a_low.get) == "sad"
    assert max(a_high, key=a_high.get) == "happy"


# ---------------------------------------------------------- soft membership

def test_soft_membership_is_continuous_along_arousal_axis():
    """Moving from low to high arousal at fixed positive valence:
    `calm` falls monotonically; `happy` rises monotonically."""
    valence = 1.0
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    calm = [mood_probs_from_va(a, valence)["calm"] for a in grid]
    happy = [mood_probs_from_va(a, valence)["happy"] for a in grid]
    assert calm == sorted(calm, reverse=True)
    assert happy == sorted(happy)


def test_soft_membership_is_continuous_along_valence_axis():
    """At high arousal, sweeping valence: aggressive falls, happy
    rises."""
    arousal = 1.0
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    aggressive = [
        mood_probs_from_va(arousal, v)["aggressive"] for v in grid
    ]
    happy = [mood_probs_from_va(arousal, v)["happy"] for v in grid]
    assert aggressive == sorted(aggressive, reverse=True)
    assert happy == sorted(happy)


def test_sigma_controls_sharpness():
    """Smaller sigma -> sharper assignment (corner mass closer to 1).
    Larger sigma -> flatter (corner mass closer to uniform)."""
    sharp = mood_probs_from_va(1.0, 1.0, sigma=0.1)["happy"]
    flat = mood_probs_from_va(1.0, 1.0, sigma=2.0)["happy"]
    assert sharp > flat
    assert sharp > 0.9
    assert flat < 0.5


# ------------------------------------------------------------ aux behaviour

def test_aux_weight_under_one_does_not_displace_corner_winners():
    """At a quadrant corner, the matching quadrant mood must still win
    when epic/dark are enabled (weight=0.5 protects against the
    auxiliary classes stealing the corner)."""
    probs = mood_probs_from_va(1.0, 1.0, prototypes=PROTOTYPES_WITH_AUX)
    assert max(probs, key=probs.get) == "happy"


def test_custom_prototypes_take_effect():
    """A user-supplied prototype map is respected (not silently
    overridden by the default)."""
    custom = {
        "happy": MoodPrototype(valence=0.0, arousal=0.0),  # swap with sad
        "sad":   MoodPrototype(valence=1.0, arousal=1.0),
    }
    probs = mood_probs_from_va(1.0, 1.0, prototypes=custom)
    assert max(probs, key=probs.get) == "sad"


def test_empty_prototypes_returns_uniform():
    probs = mood_probs_from_va(0.5, 0.5, prototypes={})
    expected = 1.0 / len(MOODS)
    for m in MOODS:
        assert probs[m] == pytest.approx(expected)


def test_zero_weight_prototype_is_inert():
    custom = {m: MoodPrototype(0.5, 0.5, weight=0.0) for m in MOODS}
    probs = mood_probs_from_va(0.5, 0.5, prototypes=custom)
    # All zero weights -> uniform fallback
    expected = 1.0 / len(MOODS)
    for m in MOODS:
        assert probs[m] == pytest.approx(expected)


def test_unknown_mood_keys_are_ignored_silently():
    custom = {
        "happy": MoodPrototype(1.0, 1.0),
        "not_a_real_mood": MoodPrototype(0.0, 0.0),
    }
    probs = mood_probs_from_va(1.0, 1.0, prototypes=custom)
    assert "not_a_real_mood" not in probs
    assert max(probs, key=probs.get) == "happy"


# -------------------------------------------------------------- helpers

def test_dominant_mood_matches_argmax_of_probs():
    for v, a in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.3, 0.7)]:
        probs = mood_probs_from_va(a, v)
        assert dominant_mood_from_va(a, v) == max(probs, key=probs.get)


@pytest.mark.parametrize(
    "valence,arousal,expected",
    [
        (0.9, 0.9, "happy"),
        (0.1, 0.9, "aggressive"),
        (0.9, 0.1, "calm"),
        (0.1, 0.1, "sad"),
        (0.5, 0.5, "happy"),  # boundary -> positive half
    ],
)
def test_quadrant_label_buckets(valence, arousal, expected):
    assert quadrant_label(arousal, valence) == expected


def test_default_quadrant_prototypes_cover_each_quadrant_once():
    """Sanity check: the 4 default prototypes really sit on the 4
    corners (catches accidental edits to coordinates)."""
    corners = {(p.valence, p.arousal) for p in DEFAULT_QUADRANT_PROTOTYPES.values()}
    assert corners == {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}


def test_draft_aux_prototypes_have_underweight_priors():
    for proto in DRAFT_AUX_PROTOTYPES.values():
        assert 0.0 < proto.weight < 1.0
