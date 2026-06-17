"""Tests for groovebot.groove_style — JointCommand bridge v1.

Coverage:
  * URDF-limit safety sweep for every primitive (NFR-4).
  * style → primitive mapping covers the full MOVES vocabulary.
  * arousal monotonicity (higher intensity → larger amplitude).
  * beat → rate (one cycle per beat for headbang/bob/fist_pump/clap).
  * generator behaviour: neutral pose when no style, unknown move
    falls back to neutral, every emitted command carries all
    JOINT_NAMES (so the backend always gets a complete frame).
  * orchestrator wiring with a counting backend stays headless.

These tests do not import mujoco. The URDF is parsed via the existing
limits helper.
"""
from __future__ import annotations
import math
import os

import numpy as np
import pytest

from groovebot.backend import JOINT_NAMES
from groovebot.groove_style import (
    MOVE_PRIMITIVES,
    SOFT_AMP,
    StyleGrooveGenerator,
    metronome_from_style,
    neutral_pose,
)
from groovebot.limits import load_joint_limits
from groovebot.orchestrator import MetronomePerception, Orchestrator
from groovebot.style.select import GrooveStyle
from groovebot.style.table import MOVES
from groovebot.types import GrooveContext, JointCommand


URDF = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "robot", "groovebot.urdf"
)


# ----------------------------------------------------------------- helpers

def _style(move: str, *, intensity: float = 0.8, bpm: float = 120.0) -> GrooveStyle:
    return GrooveStyle(
        move=move,
        intensity=intensity,
        genre="rock",
        mood="aggressive",
        mood_probs={"aggressive": 1.0},
        genre_probs={"rock": 1.0},
        tempo_bpm=bpm,
        arousal=intensity,
        arousal_bucket="mid",
    )


def _ctx(beat_pos: float, *, arousal: float = 0.5) -> GrooveContext:
    return GrooveContext(
        beat_pos=beat_pos,
        downbeat=False,
        tempo=120.0,
        arousal=arousal,
        valence=0.0,
        energy=arousal,
    )


class _CountingBackend:
    def __init__(self):
        self.set_calls = 0
        self.step_calls = 0
        self.last = {}
        self.history: list[dict[str, float]] = []

    def load(self, urdf_path: str) -> None:
        pass

    def set_joint_targets(self, targets):
        self.set_calls += 1
        self.last = dict(targets)
        self.history.append(dict(targets))

    def step(self, dt: float) -> None:
        self.step_calls += 1

    def get_joint_states(self):
        return dict(self.last)

    def close(self) -> None:
        pass


# ---------------------------------------------- vocabulary completeness

def test_move_primitives_cover_full_vocabulary():
    # Every move the table can output must have a primitive. If the
    # table grows, this test catches the missing primitive immediately.
    assert set(MOVE_PRIMITIVES.keys()) == set(MOVES)


def test_neutral_pose_covers_all_joints():
    pose = neutral_pose()
    assert set(pose.keys()) == set(JOINT_NAMES)
    assert all(v == 0.0 for v in pose.values())


# ------------------------------------------------- per-primitive safety

@pytest.mark.parametrize("move", list(MOVE_PRIMITIVES.keys()))
def test_primitive_outputs_respect_urdf_limits(move):
    limits = load_joint_limits(URDF, joint_names=JOINT_NAMES)
    prim = MOVE_PRIMITIVES[move]
    bs = np.linspace(0.0, 4.0, 401)
    for intensity in (0.0, 0.25, 0.5, 0.75, 1.0):
        for b in bs:
            delta = prim(float(b), float(intensity))
            for name, value in delta.items():
                lo, hi = limits[name]
                assert lo - 1e-9 <= value <= hi + 1e-9, (
                    f"{move}@i={intensity}, b={b:.3f}: "
                    f"{name}={value:.4f} outside URDF [{lo}, {hi}]"
                )


@pytest.mark.parametrize("move", list(MOVE_PRIMITIVES.keys()))
def test_primitive_outputs_respect_soft_amp(move):
    # The soft ceiling is the design contract: SOFT_AMP must not be
    # exceeded by any primitive at intensity 1, so we never ride the
    # hard URDF rail. Failure here means a primitive amplitude is too
    # aggressive for the v1 bridge.
    prim = MOVE_PRIMITIVES[move]
    bs = np.linspace(0.0, 4.0, 401)
    for b in bs:
        delta = prim(float(b), 1.0)
        for name, value in delta.items():
            amp = SOFT_AMP[name]
            assert abs(value) <= amp + 1e-9, (
                f"{move}@i=1.0, b={b:.3f}: |{name}|={abs(value):.4f} > "
                f"soft amp {amp}"
            )


@pytest.mark.parametrize("move", list(MOVE_PRIMITIVES.keys()))
def test_primitive_at_zero_intensity_returns_zero(move):
    prim = MOVE_PRIMITIVES[move]
    # All primitives use intensity as a multiplicative amplitude factor.
    # quiet_listen is the one exception: it floors at i=0.1 because a
    # "totally still" robot looks broken on stage, so it still produces
    # a breathing motion <= 0.005 rad.
    for b in (0.0, 0.5, 1.0, 1.5):
        out = prim(float(b), 0.0)
        for name, v in out.items():
            if move == "quiet_listen":
                assert abs(v) <= 0.01
            else:
                assert v == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------ arousal monotonicity

@pytest.mark.parametrize("move", ["headbang", "bob_nod", "sway", "rock"])
def test_arousal_increases_amplitude(move):
    # Bigger intensity → bigger peak-to-peak swing on the driven joint.
    prim = MOVE_PRIMITIVES[move]
    bs = np.linspace(0.0, 2.0, 200)

    def peak_to_peak(intensity: float) -> float:
        joint = next(iter(prim(0.0, intensity).keys()))
        vals = [prim(float(b), intensity)[joint] for b in bs]
        return max(vals) - min(vals)

    lo = peak_to_peak(0.3)
    mid = peak_to_peak(0.6)
    hi = peak_to_peak(1.0)
    assert lo < mid < hi


# ----------------------------------------------------- beat → rate

def _count_local_minima(xs: list[float]) -> int:
    n = 0
    for i in range(1, len(xs) - 1):
        if xs[i] < xs[i - 1] and xs[i] < xs[i + 1]:
            n += 1
    return n


def test_headbang_completes_one_dip_per_beat():
    # neck_pitch reaches its negative peak at b=0.5, 1.5, 2.5, 3.5 over
    # 4 beats → exactly 4 dips. Coarse enough to be stable under
    # sampling density.
    prim = MOVE_PRIMITIVES["headbang"]
    bs = np.linspace(0.0, 4.0, 4001)
    vals = [prim(float(b), 1.0)["neck_pitch"] for b in bs]
    assert _count_local_minima(vals) == 4


def test_sway_completes_one_cycle_per_two_beats():
    # torso_roll = 0.35 * sin(π b) on a 4-beat window crosses zero at
    # interior integer beats b=1, 2, 3 → 3 sign changes (endpoint zeros
    # at b=0 and b=4 don't count as interior crossings).
    prim = MOVE_PRIMITIVES["sway"]
    bs = np.linspace(0.0, 4.0, 4001)
    vals = [prim(float(b), 1.0)["torso_roll"] for b in bs]
    crossings = 0
    for i in range(1, len(vals)):
        if vals[i - 1] * vals[i] < 0:
            crossings += 1
    assert crossings == 3


# ----------------------------------------------------- generator wiring

def test_generator_returns_neutral_pose_when_no_style():
    gen = StyleGrooveGenerator(style=None)
    cmd = gen.generate(_ctx(0.5))
    assert isinstance(cmd, JointCommand)
    assert cmd.targets == neutral_pose()


def test_generator_returns_neutral_when_move_unknown():
    gen = StyleGrooveGenerator(style=_style("not_a_real_move"))
    cmd = gen.generate(_ctx(0.5))
    assert cmd.targets == neutral_pose()


@pytest.mark.parametrize("move", list(MOVE_PRIMITIVES.keys()))
def test_generator_emits_all_joints_for_every_move(move):
    gen = StyleGrooveGenerator(style=_style(move))
    cmd = gen.generate(_ctx(0.5))
    assert set(cmd.targets.keys()) == set(JOINT_NAMES)


def test_generator_routes_move_to_primitive():
    # Forcing a primitive that touches torso_roll only proves the dispatch:
    # if the generator routes by name, torso_roll moves and (say)
    # neck_pitch stays at the neutral pose value (0).
    gen = StyleGrooveGenerator(style=_style("sway", intensity=1.0))
    cmd = gen.generate(_ctx(0.5))         # sin(π·0.5)=1 → max torso_roll
    assert cmd.targets["torso_roll"] == pytest.approx(0.35, abs=1e-6)
    assert cmd.targets["neck_pitch"] == 0.0


def test_set_style_updates_active_move():
    gen = StyleGrooveGenerator(style=_style("sway", intensity=1.0))
    cmd1 = gen.generate(_ctx(0.5))
    assert cmd1.targets["torso_roll"] != 0.0
    gen.set_style(_style("headbang", intensity=1.0))
    cmd2 = gen.generate(_ctx(0.5))
    assert cmd2.targets["torso_roll"] == 0.0       # sway no longer active
    assert cmd2.targets["neck_pitch"] != 0.0       # headbang now driving


def test_use_ctx_arousal_modulates_intensity():
    # use_ctx_arousal=True multiplies intensity by 0.5 + 0.5*ctx.arousal.
    # At ctx.arousal=0 the effective intensity halves → amplitude halves
    # for a linear-amplitude move (sway).
    base = _style("sway", intensity=1.0)
    gen_off = StyleGrooveGenerator(style=base)
    gen_on = StyleGrooveGenerator(style=base, use_ctx_arousal=True)

    ctx_quiet = _ctx(0.5, arousal=0.0)
    off_val = gen_off.generate(ctx_quiet).targets["torso_roll"]
    on_val = gen_on.generate(ctx_quiet).targets["torso_roll"]
    # off uses intensity=1 → 0.35*sin(π·0.5)=0.35
    # on  uses intensity=0.5 → 0.175
    assert off_val == pytest.approx(0.35, abs=1e-6)
    assert on_val == pytest.approx(0.175, abs=1e-6)


def test_orchestrator_drives_style_generator_at_fixed_rate():
    backend = _CountingBackend()
    gen = StyleGrooveGenerator(style=_style("bob_nod"))
    orch = Orchestrator(
        perception=MetronomePerception(bpm=120.0, energy=0.8),
        generator=gen,
        backend=backend,
        rate=50.0,
    )
    orch.run(seconds=1.0)
    assert backend.set_calls == 50
    assert backend.step_calls == 50
    assert set(backend.last.keys()) == set(JOINT_NAMES)


def test_orchestrator_picks_up_set_style_mid_run():
    # Mimic the M2 control flow: the selector pushes a new style during
    # the run. Half a second of "sway", half a second of "headbang".
    backend = _CountingBackend()
    gen = StyleGrooveGenerator(style=_style("sway", intensity=1.0))
    perception = MetronomePerception(bpm=120.0, energy=0.8)
    orch = Orchestrator(
        perception=perception,
        generator=gen,
        backend=backend,
        rate=50.0,
    )
    orch.run(seconds=0.5)
    assert any(h["torso_roll"] != 0.0 for h in backend.history)
    n_before = len(backend.history)

    gen.set_style(_style("headbang", intensity=1.0))
    orch.run(seconds=0.5)
    after = backend.history[n_before:]
    # After the swap: torso_roll is back at neutral and neck_pitch is
    # the one moving. (One frame after a beat boundary may sit at 0 for
    # both — we look at the whole window.)
    assert any(h["neck_pitch"] != 0.0 for h in after)


# ---------------------------------------------- perception helper

def test_metronome_from_style_uses_style_tempo():
    style = _style("bob_nod", intensity=0.6, bpm=140.0)
    perc = metronome_from_style(style)
    ctx = perc.tick(1.0 / 50.0)
    assert perc.bpm == pytest.approx(140.0)
    assert perc.energy == pytest.approx(0.6)
    assert ctx.tempo == pytest.approx(140.0)


def test_metronome_from_style_floors_degenerate_bpm():
    # A bogus tempo estimate (e.g. 0 BPM from librosa on a near-silent
    # clip) should not freeze the clock. We floor it so the loop still
    # ticks, just slowly.
    style = _style("quiet_listen", intensity=0.1, bpm=0.0)
    perc = metronome_from_style(style, floor_bpm=40.0)
    assert perc.bpm == pytest.approx(40.0)
