"""
groove.py — turns a GrooveContext into a JointCommand.

This is the *placeholder* groove generator: simple, hand-authored, beat-synced
rules. It exists so the whole pipeline runs end-to-end today (M1), and it is
the precise seam you will swap out later:

    M1 (now)  : RuleGrooveGenerator (these rules)         <- you are here
    M2 (req.) : same rules, but the ctx's arousal/energy come from the singer
    M3 (goal) : replace this class with the trained generative model
                (VQ-VAE groove codebook conditioned on beat + arousal + voice
                 embedding, trained on vocal-separated AIST++)

The contract is `GrooveGenerator.generate(ctx) -> JointCommand` (spec §5.2);
swapping M1 -> M3 touches only this file.
"""
from __future__ import annotations
import math

from .types import GrooveContext, JointCommand


def _compute_targets(beat_pos: float, energy: float) -> dict[str, float]:
    """Mirror-type groove: louder/higher energy -> bigger, faster motion.

    Pure function over (beat_pos, energy) so both the new generate(ctx) entry
    point and the legacy GrooveController.compute(...) shim can share it.
    """
    e = max(0.0, min(1.0, energy))
    b = 2.0 * math.pi * beat_pos            # one cycle per beat
    sway = 2.0 * math.pi * beat_pos / 2.0   # one sway per 2 beats

    neck_pitch = -0.30 * e * (0.5 - 0.5 * math.cos(b))
    neck_yaw = 0.25 * e * math.sin(2.0 * math.pi * beat_pos / 4.0)

    torso_roll = 0.32 * e * math.sin(sway)
    torso_pitch = 0.10 * e * (0.5 - 0.5 * math.cos(b))

    raise_base = -0.5 - 0.7 * e
    pump = 0.35 * e * math.sin(b)
    elbow = 0.7 + 0.6 * e * (0.5 + 0.5 * math.sin(b))

    return {
        "neck_pitch": neck_pitch,
        "neck_yaw": neck_yaw,
        "torso_roll": torso_roll,
        "torso_pitch": torso_pitch,
        "l_shoulder_pitch": raise_base + pump,
        "l_shoulder_roll": 0.6 + 0.5 * e,
        "l_elbow": elbow,
        "r_shoulder_pitch": raise_base - pump,
        "r_shoulder_roll": -(0.6 + 0.5 * e),
        "r_elbow": elbow,
    }


class RuleGrooveGenerator:
    """M1 hand-authored generator. Conforms to spec §5.2 GrooveGenerator."""

    def generate(self, ctx: GrooveContext) -> JointCommand:
        return JointCommand(targets=_compute_targets(ctx.beat_pos, ctx.energy))


class GrooveController:
    """Legacy M1 shim. Kept so callers using the older (beat_pos, energy) entry
    point keep working until they migrate to RuleGrooveGenerator.generate(ctx).
    """

    def compute(self, beat_pos: float, energy: float) -> dict[str, float]:
        return _compute_targets(beat_pos, energy)
