"""groovebot.style.mood_from_va — derive mood probs from (arousal, valence).

Circumplex affect (Russell 1980): mood lives in a 2D plane spanned by
arousal (low → high) and valence (negative → positive). The 4 quadrants
of the 0..1 V/A square correspond cleanly to 4 of our 6 mood classes:

    happy      = (V=1, A=1)    high arousal, positive valence
    aggressive = (V=0, A=1)    high arousal, negative valence
    calm       = (V=1, A=0)    low  arousal, positive valence
    sad        = (V=0, A=0)    low  arousal, negative valence

`epic` and `dark` are NOT pure V/A coordinates:

- **epic**: high arousal, but valence is ambiguous (triumphant=positive,
  ominous-grand=negative). The MTG-trained v3 head confirmed this: epic
  rarely confuses with aggressive (0%) despite sharing the high-arousal
  half — the discriminating axis is "grand / cinematic," orthogonal to
  pure V/A.
- **dark**: low-mid arousal + low valence. Overlaps with `sad`. The v3
  head separates them more by timbre (dark = brassy/horror, sad =
  acoustic/intimate) than by V/A.

Default config: only the 4 quadrant prototypes are populated. `epic` and
`dark` get probability 0. To opt them in, pass `PROTOTYPES_WITH_AUX` (or
your own dict) as `prototypes`; the auxiliary entries are drafts and the
intended pattern is for the user to edit them.

Membership is **soft** (Gaussian on squared distance to the prototype),
NOT argmax. The output dict sums to 1.0 over `MOODS` and is shaped to
feed straight into `table.select_move`, which already treats mood as a
soft distribution. The companion `dominant_mood_from_va` is a logging
convenience.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping

from groovebot.style.model import MOODS


@dataclass(frozen=True)
class MoodPrototype:
    """A point in the V/A plane that pulls one mood class toward it.

    `weight` is a soft prior in (0, 1]. Use it to under-weight auxiliary
    classes (epic/dark) that share V/A territory with the 4 quadrants
    so they do not steal mass from the cleaner classes.
    """
    valence: float
    arousal: float
    weight: float = 1.0


# Four clean V/A quadrants. These four moods read off the circumplex
# directly and are the only ones populated by default.
DEFAULT_QUADRANT_PROTOTYPES: dict[str, MoodPrototype] = {
    "happy":      MoodPrototype(valence=1.0, arousal=1.0),
    "aggressive": MoodPrototype(valence=0.0, arousal=1.0),
    "calm":       MoodPrototype(valence=1.0, arousal=0.0),
    "sad":        MoodPrototype(valence=0.0, arousal=0.0),
}


# Drafts for epic / dark. Neither sits on a single V/A point, so the
# coordinates here are a starting point — the intended workflow is for
# the user to tune them against listening tests. Weight 0.5 keeps them
# from stealing mass from happy/aggressive/sad. Disabled by default; opt
# in by passing `PROTOTYPES_WITH_AUX` (or your own mix) as `prototypes`.
DRAFT_AUX_PROTOTYPES: dict[str, MoodPrototype] = {
    # epic: high arousal, mildly positive valence (triumphant trailer
    # bias). Underweighted so it does not eat happy or aggressive mass
    # at the corners.
    "epic": MoodPrototype(valence=0.55, arousal=0.95, weight=0.5),
    # dark: low-mid arousal, low valence. Underweighted so genuine sad
    # at A≈0 keeps its mass.
    "dark": MoodPrototype(valence=0.10, arousal=0.30, weight=0.5),
}


PROTOTYPES_WITH_AUX: dict[str, MoodPrototype] = {
    **DEFAULT_QUADRANT_PROTOTYPES,
    **DRAFT_AUX_PROTOTYPES,
}


# Softmax-style temperature on squared distance to each prototype.
# Smaller sigma → sharper assignment; larger → flatter. 0.45 was tuned
# so a query at the (0.5, 0.5) center spreads roughly uniformly over the
# 4 quadrants (~0.25 each) and a query at a corner gives ~0.5+ mass to
# the matching mood. The thresholds in `arousal_bucket()` (0.33 / 0.66)
# imply the same scale, so 0.45 keeps V/A and bucket geometry coherent.
DEFAULT_SIGMA = 0.45


def mood_probs_from_va(
    arousal: float,
    valence: float,
    *,
    prototypes: Mapping[str, MoodPrototype] = DEFAULT_QUADRANT_PROTOTYPES,
    sigma: float = DEFAULT_SIGMA,
) -> dict[str, float]:
    """Soft probability over `MOODS` from a (V, A) point.

    Both inputs are 0..1 (DEAM-calibrated by `sam_to_unit`). Output
    sums to 1.0; mood classes absent from `prototypes` get probability
    0. When `prototypes` is empty or all weights are 0, returns uniform
    (degenerate but safe).
    """
    a = float(max(0.0, min(1.0, arousal)))
    v = float(max(0.0, min(1.0, valence)))
    s = max(float(sigma), 1e-6)

    raw: dict[str, float] = {m: 0.0 for m in MOODS}
    for mood, proto in prototypes.items():
        if mood not in raw:
            continue  # silently ignore unknown moods
        d2 = (v - proto.valence) ** 2 + (a - proto.arousal) ** 2
        raw[mood] = max(0.0, float(proto.weight)) * math.exp(-d2 / (2.0 * s * s))

    total = sum(raw.values())
    if total <= 0.0:
        return {m: 1.0 / len(MOODS) for m in MOODS}
    return {m: raw[m] / total for m in MOODS}


def dominant_mood_from_va(
    arousal: float,
    valence: float,
    **kw,
) -> str:
    """Argmax convenience over `mood_probs_from_va`. Use the soft
    distribution for downstream selection; this is for logging only."""
    probs = mood_probs_from_va(arousal, valence, **kw)
    return max(probs, key=probs.get)


def quadrant_label(arousal: float, valence: float) -> str:
    """The 4-quadrant label a V/A point falls in, ignoring soft
    membership. Used by the comparison report and by tests. Boundary
    (V=0.5 or A=0.5) ties resolve toward the positive half so the
    function is total."""
    high_a = float(arousal) >= 0.5
    pos_v = float(valence) >= 0.5
    if high_a and pos_v:
        return "happy"
    if high_a and not pos_v:
        return "aggressive"
    if not high_a and pos_v:
        return "calm"
    return "sad"


__all__ = [
    "DEFAULT_QUADRANT_PROTOTYPES",
    "DEFAULT_SIGMA",
    "DRAFT_AUX_PROTOTYPES",
    "MoodPrototype",
    "PROTOTYPES_WITH_AUX",
    "dominant_mood_from_va",
    "mood_probs_from_va",
    "quadrant_label",
]
