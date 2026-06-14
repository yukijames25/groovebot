"""groovebot.style.table — Yuki's nori (groove style) lookup table.

Inputs:
  - genre (str)         : GTZAN vocabulary token (`GENRES`)
  - arousal_bucket (str): "low" / "mid" / "high"
  - mood_probs (dict)   : mood -> probability, softmax over the 6 moods.

Output:
  - move (str)      : one of the upper-body-feasible classes in `MOVES`.
  - intensity (float): 0..1 scalar driving "how big / how fast" downstream.

Design choice (per Yuki's spec): mood enters as a soft distribution, not
argmax. Each mood contributes a preferred-move distribution; we sum them
weighted by their probability and multiply by the genre × arousal bias.

Upper-body-feasible move vocabulary (10-DOF torso/arms, no legs):
  - headbang       : neck pitch full range, fast (metal-grade)
  - bob_nod        : neck pitch small range, medium tempo
  - sway           : torso roll left-right, medium tempo (ballads)
  - rock           : torso pitch forward-back, slow
  - fist_pump      : arms up, beat-synced shoulder pitch
  - clap           : both arms forward, beat-synced elbow
  - penlight_wave  : one arm raised, side-to-side wave (idol-style)
  - quiet_listen   : near-pose, tiny breath-like motion only
"""
from __future__ import annotations
from typing import Mapping

from groovebot.style.model import GENRES, MOODS


MOVES = (
    "headbang", "bob_nod", "sway", "rock",
    "fist_pump", "clap", "penlight_wave", "quiet_listen",
)


# Per-genre prior over moves. Rows ideally sum to ~1 (treated as a weight,
# not strictly a distribution — small over/under is fine).
_GENRE_FAMILY: dict[str, dict[str, float]] = {
    "blues":     {"sway": 0.4, "bob_nod": 0.3, "rock": 0.2, "quiet_listen": 0.1},
    "classical": {"quiet_listen": 0.6, "sway": 0.4},
    "country":   {"sway": 0.4, "bob_nod": 0.4, "clap": 0.2},
    "disco":     {"clap": 0.3, "bob_nod": 0.3, "penlight_wave": 0.2, "fist_pump": 0.2},
    "hiphop":    {"bob_nod": 0.5, "fist_pump": 0.2, "headbang": 0.15, "sway": 0.15},
    "jazz":      {"sway": 0.4, "bob_nod": 0.3, "quiet_listen": 0.3},
    "metal":     {"headbang": 0.6, "fist_pump": 0.25, "bob_nod": 0.15},
    "pop":       {"clap": 0.3, "penlight_wave": 0.25, "bob_nod": 0.25, "sway": 0.2},
    "reggae":    {"sway": 0.5, "bob_nod": 0.3, "rock": 0.2},
    "rock":      {"headbang": 0.4, "fist_pump": 0.3, "bob_nod": 0.3},
}

# Arousal nudge: add to genre prior. Low arousal up-weights still moves,
# high arousal up-weights big moves.
_AROUSAL_NUDGE: dict[str, dict[str, float]] = {
    "low":  {"quiet_listen": +0.4, "sway": +0.2, "headbang": -0.4, "fist_pump": -0.3},
    "mid":  {},
    "high": {"quiet_listen": -0.4, "sway": -0.2, "headbang": +0.3, "fist_pump": +0.3, "clap": +0.1},
}

# Mood preference: distribution over moves per mood. Weight contribution
# is `mood_prob * value`. Multiple moods compose linearly.
_MOOD_MOVE_PREF: dict[str, dict[str, float]] = {
    "aggressive":   {"headbang": 0.6, "fist_pump": 0.3, "bob_nod": 0.1},
    "happy":        {"clap": 0.4, "penlight_wave": 0.3, "bob_nod": 0.2, "sway": 0.1},
    "sad":          {"sway": 0.5, "quiet_listen": 0.4, "rock": 0.1},
    "calm":         {"quiet_listen": 0.6, "sway": 0.4},
    "dark":         {"bob_nod": 0.4, "headbang": 0.3, "rock": 0.3},
    "epic":         {"penlight_wave": 0.4, "fist_pump": 0.4, "headbang": 0.2},
}

# Base intensity per arousal bucket. Mood then nudges within ±~30%.
_AROUSAL_INTENSITY: dict[str, float] = {"low": 0.30, "mid": 0.60, "high": 0.90}

# Mood -> intensity multiplier additive term. Sum is multiplied into the
# base intensity (clamped to [0, 1]).
_MOOD_INTENSITY_NUDGE: dict[str, float] = {
    "aggressive": +0.30,
    "epic":       +0.20,
    "happy":      +0.10,
    "dark":        0.0,
    "sad":        -0.20,
    "calm":       -0.30,
}


def select_move(
    genre: str,
    arousal_bucket: str,
    mood_probs: Mapping[str, float],
) -> tuple[str, float]:
    """Return (move, intensity) for the given style descriptors.

    `mood_probs` is a partial mapping; missing moods are treated as 0.
    Probabilities are not required to sum to 1, but we encourage softmax
    outputs from the model.

    The move is `argmax` over the combined bias dictionary. Ties resolve
    by `MOVES` order, which is deterministic.
    """
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre!r}; expected one of {GENRES}")
    if arousal_bucket not in _AROUSAL_INTENSITY:
        raise ValueError(
            f"unknown arousal bucket {arousal_bucket!r}; expected low/mid/high"
        )

    family = _GENRE_FAMILY.get(genre, {})
    nudge = _AROUSAL_NUDGE.get(arousal_bucket, {})
    genre_bias = {m: max(0.0, family.get(m, 0.0) + nudge.get(m, 0.0)) for m in MOVES}

    mood_bias: dict[str, float] = {m: 0.0 for m in MOVES}
    for mood, p in mood_probs.items():
        if mood not in MOODS:
            continue
        pref = _MOOD_MOVE_PREF.get(mood, {})
        for move, w in pref.items():
            mood_bias[move] += float(p) * float(w)

    # Combine: small floor (0.05) on the genre bias so a strong mood can
    # still surface a move the genre never lists.
    combined = {
        m: (genre_bias[m] + 0.05) * (1.0 + mood_bias[m])
        for m in MOVES
    }
    # argmax in MOVES order (deterministic tie-break)
    move = max(MOVES, key=lambda m: combined[m])

    base = _AROUSAL_INTENSITY[arousal_bucket]
    intensity_nudge = sum(
        float(mood_probs.get(m, 0.0)) * _MOOD_INTENSITY_NUDGE.get(m, 0.0)
        for m in MOODS
    )
    intensity = max(0.0, min(1.0, base * (1.0 + intensity_nudge)))
    return move, float(intensity)
