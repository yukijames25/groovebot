"""Body-agnosticism regression test (spec §10.1):
the same brain must drive both MuJoCo and PyBullet without crashing.

PyBullet is not in this project's hard dependency set (it is referenced by
backend.py for users who want dynamics), so the PyBullet half skips cleanly
when pybullet isn't installed.
"""
from __future__ import annotations
import pytest

from groovebot.backend import JOINT_NAMES, MujocoBackend, PyBulletBackend
from groovebot.groove import RuleGrooveGenerator
from groovebot.limits import make_clamp
from groovebot.orchestrator import MetronomePerception, Orchestrator

from .conftest import URDF_PATH


def _run(backend, seconds: float = 0.5, rate: float = 50.0):
    backend.load(URDF_PATH)
    orch = Orchestrator(
        perception=MetronomePerception(bpm=120.0, energy=0.8),
        generator=RuleGrooveGenerator(),
        backend=backend,
        rate=rate,
        clamp=make_clamp(URDF_PATH, joint_names=JOINT_NAMES),
    )
    orch.run(seconds)
    states = backend.get_joint_states()
    backend.close()
    return states


def test_mujoco_smoke():
    pytest.importorskip("mujoco")
    states = _run(MujocoBackend())
    assert set(states.keys()) == set(JOINT_NAMES)
    for name, value in states.items():
        assert isinstance(value, float)


def test_pybullet_smoke():
    pytest.importorskip("pybullet")
    states = _run(PyBulletBackend(gui=False))
    assert set(states.keys()) == set(JOINT_NAMES)
    for name, value in states.items():
        assert isinstance(value, float)


def test_brain_is_body_agnostic():
    """The brain object itself never touches a simulator import."""
    import sys
    # Drop sim modules so we can prove the generator works without them.
    for mod in ("mujoco", "pybullet"):
        sys.modules.pop(mod, None)
    from groovebot.types import GrooveContext
    from groovebot.groove import RuleGrooveGenerator
    gen = RuleGrooveGenerator()
    cmd = gen.generate(GrooveContext(beat_pos=1.0, downbeat=True, tempo=120.0,
                                     arousal=0.8, valence=0.0, energy=0.8))
    assert set(cmd.targets.keys()) == set(JOINT_NAMES)
