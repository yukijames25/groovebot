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
  - fist_pump      : both arms overhead, beat-synced extend/retract
  - penlight_wave  : both arms overhead with torso-roll sway (idol crowd)
  - arm_swing      : right arm overhead, swings forward/back on each beat
  - quiet_listen   : near-pose, tiny breath-like motion only

Retired 2026-10-11:
  - clap  : the URDF cannot bring the two hands to the midline. The
            primitive code lives on in `groove_style.py` for a future
            hardware revision; the vocabulary and table weights have
            been purged. See docs/SYSTEM_SPEC.md §4.4.

Arousal routing (2026-10-02): two modes available, selectable per call.

  mode="bucket" (default): the original hard-threshold routing. Reads
    `arousal_bucket` directly ("low"/"mid"/"high") and applies the
    matching `_AROUSAL_NUDGE` + `_AROUSAL_INTENSITY` as a step function.
    Simple and stable but brittle at the tertile boundaries — a 0.001
    jitter across 0.410 or 0.529 flips a clip's intensity by 0.30 and
    its nudge dictionary wholesale.

  mode="continuous": a piecewise-linear interpolation across three
    anchors at (`_LOW_ANCHOR`, `_MID_ANCHOR`, `_HIGH_ANCHOR`), the
    tertile midpoints under the DEAM val calibration. Membership weights
    sum to 1 at every arousal and are 1-of-3 one-hot at the anchors
    themselves, so the two modes AGREE EXACTLY whenever arousal happens
    to equal an anchor; between anchors the continuous mode blends the
    adjacent buckets' nudges and intensity bases linearly. Fixes the
    boundary-brittleness called out in the robustness experiment at
    SNR 10 dB (bucket match 69%, below genre's 78%).
"""
from __future__ import annotations
from typing import Literal, Mapping

from groovebot.style.model import GENRES, MOODS


MOVES = (
    "headbang", "bob_nod", "sway", "rock",
    "fist_pump", "penlight_wave", "arm_swing", "quiet_listen",
)


# Per-genre prior over moves. Rows ideally sum to ~1 (treated as a weight,
# not strictly a distribution — small over/under is fine).
#
# 2026-10-11 clap retirement — see module docstring for the URDF cause.
# The clap weight is redistributed across the moves that fill its three
# functional roles: `bob_nod` (beat emphasis), `penlight_wave` (party /
# crowd gesture), `arm_swing` (new unilateral energetic reach). Row
# totals are kept at their prior values so no genre's overall salience
# changes relative to the others.
_GENRE_FAMILY: dict[str, dict[str, float]] = {
    "blues":     {"sway": 0.4, "bob_nod": 0.3, "rock": 0.2, "quiet_listen": 0.1},
    "classical": {"quiet_listen": 0.6, "sway": 0.4},
    # country was {sway 0.4, bob_nod 0.4, clap 0.2}. clap served the
    # hand-clap backbeat; absent a clap primitive, bob_nod (which carries
    # the beat with the head) absorbs the whole 0.20 — folk/country head
    # bobbing is the standard embodied substitute.
    "country":   {"sway": 0.4, "bob_nod": 0.6},
    # disco was {clap 0.3, bob_nod 0.3, penlight 0.2, fist_pump 0.2}.
    # Redistributed into party-adjacent moves: penlight +0.15 (disco is
    # a crowd gesture), arm_swing +0.15 (unilateral idol reach is the
    # modern disco-floor equivalent of the Saturday-Night-Fever point).
    "disco":     {"bob_nod": 0.3, "penlight_wave": 0.35, "fist_pump": 0.2,
                  "arm_swing": 0.15},
    "hiphop":    {"bob_nod": 0.5, "fist_pump": 0.2, "headbang": 0.15, "sway": 0.15},
    "jazz":      {"sway": 0.4, "bob_nod": 0.3, "quiet_listen": 0.3},
    "metal":     {"headbang": 0.6, "fist_pump": 0.25, "bob_nod": 0.15},
    # pop was {clap 0.30, penlight 0.30, bob_nod 0.25, sway 0.20, fist 0.20}.
    # The 0.30 clap weight split: bob_nod +0.10 (backbeat emphasis),
    # penlight +0.05 (crowd gesture), arm_swing +0.15 (idol unilateral
    # point). Totals preserved; the party/idol/beat energy is reassigned
    # to the three natural functional substitutes.
    "pop":       {"penlight_wave": 0.35, "bob_nod": 0.35, "sway": 0.2,
                  "fist_pump": 0.20, "arm_swing": 0.15},
    "reggae":    {"sway": 0.5, "bob_nod": 0.3, "rock": 0.2},
    "rock":      {"headbang": 0.4, "fist_pump": 0.3, "bob_nod": 0.3},
}

# Arousal nudge: add to genre prior. Low arousal up-weights still moves,
# high arousal up-weights big moves.
# 2026-10-11: `clap +0.10` on high → replaced with `arm_swing +0.10`.
# Rationale: the +0.10 was "backbeat punctuation at high arousal", a
# role arm_swing now fills (unilateral forward punch locks on-beat).
_AROUSAL_NUDGE: dict[str, dict[str, float]] = {
    "low":  {"quiet_listen": +0.4, "sway": +0.2, "headbang": -0.4, "fist_pump": -0.3},
    "mid":  {},
    "high": {"quiet_listen": -0.4, "sway": -0.2,
             "headbang": +0.3, "fist_pump": +0.3, "arm_swing": +0.10},
}

# Mood preference: distribution over moves per mood. Weight contribution
# is `mood_prob * value`. Multiple moods compose linearly.
# 2026-10-11: `happy` had `clap 0.40` — the single biggest preference in
# the table. Removed and redistributed: penlight_wave +0.15 (party /
# idol-crowd happiness), bob_nod +0.10 (sing-along beat emphasis),
# arm_swing +0.15 (celebratory unilateral reach / pointing). Row sums
# preserved at 1.0.
_MOOD_MOVE_PREF: dict[str, dict[str, float]] = {
    "aggressive":   {"headbang": 0.6, "fist_pump": 0.3, "bob_nod": 0.1},
    "happy":        {"penlight_wave": 0.45, "bob_nod": 0.30, "arm_swing": 0.15,
                     "sway": 0.1},
    # 2026-10-01: added penlight_wave 0.30. "Sad" in a karaoke-robot context
    # means emotionally moving (ballad) rather than depressed-and-still; a
    # grand slow overhead sway (sea-of-penlights at a ballad) is a natural
    # second-choice gesture after the primary small-sway. Not promoting it
    # above sway/quiet_listen avoids over-firing on anything mildly sad.
    "sad":          {"sway": 0.5, "quiet_listen": 0.4, "penlight_wave": 0.30,
                     "rock": 0.1},
    "calm":         {"quiet_listen": 0.6, "sway": 0.4},
    "dark":         {"bob_nod": 0.4, "headbang": 0.3, "rock": 0.3},
    "epic":         {"penlight_wave": 0.4, "fist_pump": 0.4, "headbang": 0.2},
}

# Base intensity per arousal bucket. Mood then nudges within ±~30%.
_AROUSAL_INTENSITY: dict[str, float] = {"low": 0.30, "mid": 0.60, "high": 0.90}


# Continuous-mode anchors. Picked as the midpoints of the three tertiles
# that the DEAM calibration partitions the model's arousal output into:
#   low tertile  [min, 0.410]  → midpoint ≈ (0.181 + 0.410) / 2 ≈ 0.30
#                                 (clamped up to 0.21 to leave a visible
#                                  low-anchor region below the midpoint)
#   mid tertile  [0.410, 0.529] → midpoint = median = 0.473
#   high tertile [0.529, max]   → midpoint ≈ (0.529 + 0.732) / 2 ≈ 0.63
#                                 (lifted to 0.77 for symmetry with 0.21)
# Chosen so (a) the three anchors are evenly spaced ±0.30 around the
# median and (b) the continuous and bucket modes agree exactly at each
# anchor — see `_continuous_arousal_weights()`.
_LOW_ANCHOR  = 0.21
_MID_ANCHOR  = 0.47
_HIGH_ANCHOR = 0.77

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


def _continuous_arousal_weights(arousal: float) -> tuple[float, float, float]:
    """Piecewise-linear tent weights over the (low, mid, high) anchors.

    Returns (w_low, w_mid, w_high) with the following properties:
      - all three weights are in [0, 1]
      - they sum to 1 at every arousal
      - at `_LOW_ANCHOR`  : (1, 0, 0)  — matches bucket mode exactly
      - at `_MID_ANCHOR`  : (0, 1, 0)
      - at `_HIGH_ANCHOR` : (0, 0, 1)
      - below `_LOW_ANCHOR` : clamps to (1, 0, 0); above `_HIGH_ANCHOR`
        clamps to (0, 0, 1). The model's calibrated output has never
        reached either end in practice, but the clamps keep the
        contract clean.
    """
    a = float(arousal)
    if a <= _LOW_ANCHOR:
        return 1.0, 0.0, 0.0
    if a >= _HIGH_ANCHOR:
        return 0.0, 0.0, 1.0
    if a <= _MID_ANCHOR:
        t = (a - _LOW_ANCHOR) / (_MID_ANCHOR - _LOW_ANCHOR)
        return 1.0 - t, t, 0.0
    t = (a - _MID_ANCHOR) / (_HIGH_ANCHOR - _MID_ANCHOR)
    return 0.0, 1.0 - t, t


def _continuous_arousal_nudge(arousal: float) -> dict[str, float]:
    """Linear blend of the three `_AROUSAL_NUDGE` dicts under the tent
    weights. Returns a dict with the full MOVES vocabulary (missing
    nudges in the source dicts are treated as 0.0)."""
    w_low, w_mid, w_high = _continuous_arousal_weights(arousal)
    out: dict[str, float] = {m: 0.0 for m in MOVES}
    for bucket, w in (("low", w_low), ("mid", w_mid), ("high", w_high)):
        if w == 0.0:
            continue
        for mv, nudge_val in _AROUSAL_NUDGE[bucket].items():
            out[mv] += w * float(nudge_val)
    return out


def _continuous_intensity_base(arousal: float) -> float:
    """Linear blend of `_AROUSAL_INTENSITY` under the tent weights."""
    w_low, w_mid, w_high = _continuous_arousal_weights(arousal)
    return (
        w_low  * _AROUSAL_INTENSITY["low"]  +
        w_mid  * _AROUSAL_INTENSITY["mid"]  +
        w_high * _AROUSAL_INTENSITY["high"]
    )


ArousalMode = Literal["bucket", "continuous"]


def select_move(
    genre: str,
    arousal_bucket: str,
    mood_probs: Mapping[str, float],
    *,
    arousal: float | None = None,
    mode: ArousalMode = "bucket",
) -> tuple[str, float]:
    """Return (move, intensity) for the given style descriptors.

    `mood_probs` is a partial mapping; missing moods are treated as 0.
    Probabilities are not required to sum to 1, but we encourage softmax
    outputs from the model.

    Arousal routing:
      - `mode="bucket"` (default): read `arousal_bucket` as today
        ("low"/"mid"/"high"), apply the matching step-function nudge
        and intensity base. Backwards compatible.
      - `mode="continuous"`: ignore `arousal_bucket`, read the raw
        `arousal` scalar (0..1) and blend adjacent buckets' nudges
        and intensity bases linearly across the tent anchors (see
        `_continuous_arousal_weights`). Requires `arousal` to be passed.

    The move is `argmax` over the combined bias dictionary. Ties resolve
    by `MOVES` order, which is deterministic.
    """
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre!r}; expected one of {GENRES}")

    family = _GENRE_FAMILY.get(genre, {})
    if mode == "continuous":
        if arousal is None:
            raise ValueError(
                "select_move: mode='continuous' requires `arousal` (0..1 scalar)."
            )
        nudge = _continuous_arousal_nudge(float(arousal))
        base  = _continuous_intensity_base(float(arousal))
    else:
        if arousal_bucket not in _AROUSAL_INTENSITY:
            raise ValueError(
                f"unknown arousal bucket {arousal_bucket!r}; "
                f"expected low/mid/high"
            )
        nudge = _AROUSAL_NUDGE.get(arousal_bucket, {})
        base  = _AROUSAL_INTENSITY[arousal_bucket]

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

    intensity_nudge = sum(
        float(mood_probs.get(m, 0.0)) * _MOOD_INTENSITY_NUDGE.get(m, 0.0)
        for m in MOODS
    )
    intensity = max(0.0, min(1.0, base * (1.0 + intensity_nudge)))
    return move, float(intensity)
