"""NFR-4: every generator output, after clamping, must be inside the URDF
joint range. Sample many (beat_pos, energy, arousal) combinations to make this
a property test rather than a single-frame check.
"""
from __future__ import annotations
import math
import random

from groovebot.backend import JOINT_NAMES
from groovebot.groove import RuleGrooveGenerator
from groovebot.limits import clamp_command, load_joint_limits
from groovebot.types import GrooveContext

from .conftest import URDF_PATH


def _ctx(beat_pos: float, energy: float, arousal: float | None = None) -> GrooveContext:
    return GrooveContext(
        beat_pos=beat_pos,
        downbeat=False,
        tempo=120.0,
        arousal=energy if arousal is None else arousal,
        valence=0.0,
        energy=energy,
    )


def test_urdf_limits_cover_all_driven_joints():
    limits = load_joint_limits(URDF_PATH, joint_names=JOINT_NAMES)
    assert set(limits.keys()) == set(JOINT_NAMES)
    for name, (lo, hi) in limits.items():
        assert lo < hi, f"{name} has degenerate range [{lo}, {hi}]"


def test_generator_output_within_limits_after_clamp():
    limits = load_joint_limits(URDF_PATH, joint_names=JOINT_NAMES)
    gen = RuleGrooveGenerator()

    rng = random.Random(0)
    # Cover phase + energy product fully, plus the over-range/extrapolated
    # corner that exercises clamp (sweep energy beyond 1.0 and below 0.0).
    for _ in range(2000):
        beat_pos = rng.uniform(0.0, 64.0)
        energy = rng.uniform(-0.5, 1.5)
        cmd = gen.generate(_ctx(beat_pos, energy))
        clamped = clamp_command(cmd, limits)
        for name, value in clamped.targets.items():
            lo, hi = limits[name]
            assert lo - 1e-9 <= value <= hi + 1e-9, (
                f"{name}={value} outside [{lo},{hi}] at beat={beat_pos:.3f} energy={energy:.3f}")


def test_clamp_is_idempotent():
    """Clamping twice should not move a command that is already in range."""
    limits = load_joint_limits(URDF_PATH, joint_names=JOINT_NAMES)
    gen = RuleGrooveGenerator()
    cmd = gen.generate(_ctx(2.3, 0.7))
    once = clamp_command(cmd, limits)
    twice = clamp_command(once, limits)
    assert once.targets == twice.targets


def test_clamp_pushes_out_of_range_values_to_boundary():
    limits = {"neck_pitch": (-0.7, 0.7)}
    from groovebot.types import JointCommand
    out = clamp_command(JointCommand(targets={"neck_pitch": 5.0}), limits)
    assert out.targets["neck_pitch"] == 0.7
    out = clamp_command(JointCommand(targets={"neck_pitch": -3.0}), limits)
    assert out.targets["neck_pitch"] == -0.7
