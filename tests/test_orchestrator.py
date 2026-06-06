"""NFR-7 + §6 sanity: orchestrator enforces the 30-50 Hz band and the loop
runs the expected number of perception/generate/backend ticks.
"""
from __future__ import annotations
import pytest

from groovebot.backend import JOINT_NAMES
from groovebot.groove import RuleGrooveGenerator
from groovebot.orchestrator import MetronomePerception, Orchestrator
from groovebot.types import GrooveContext, JointCommand


class _CountingBackend:
    def __init__(self):
        self.set_calls = 0
        self.step_calls = 0
        self.last = {}

    def load(self, urdf_path: str) -> None:
        pass

    def set_joint_targets(self, targets):
        self.set_calls += 1
        self.last = dict(targets)

    def step(self, dt: float) -> None:
        self.step_calls += 1

    def get_joint_states(self):
        return dict(self.last)

    def close(self) -> None:
        pass


def test_rate_must_be_in_30_to_50_hz():
    with pytest.raises(ValueError):
        Orchestrator(MetronomePerception(), RuleGrooveGenerator(),
                     _CountingBackend(), rate=10.0)
    with pytest.raises(ValueError):
        Orchestrator(MetronomePerception(), RuleGrooveGenerator(),
                     _CountingBackend(), rate=100.0)


def test_loop_drives_backend_at_fixed_rate():
    backend = _CountingBackend()
    orch = Orchestrator(MetronomePerception(bpm=120.0, energy=0.8),
                        RuleGrooveGenerator(), backend, rate=50.0)
    orch.run(seconds=1.0)
    assert backend.set_calls == 50
    assert backend.step_calls == 50
    assert set(backend.last.keys()) == set(JOINT_NAMES)


def test_metronome_advances_beat_pos():
    perc = MetronomePerception(bpm=120.0, energy=0.8)
    # 1s at 120 BPM = 2 beats
    for _ in range(50):
        ctx = perc.tick(1.0 / 50.0)
    assert isinstance(ctx, GrooveContext)
    assert ctx.beat_pos == pytest.approx(2.0, abs=1e-9)
    assert ctx.tempo == 120.0
