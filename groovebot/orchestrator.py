"""
orchestrator.py — the fixed-rate control loop (spec §6.1, NFR-7).

Drives `Perception -> GrooveGenerator -> RobotBackend` at a fixed 30-50 Hz. The
control loop is intentionally decoupled from perception: perception is "slow
and variable" (beat tracker, arousal head), the loop is "fast and steady".
For M1 the perception is just a metronome + constant energy stub, but it is
behind the same `Perception` protocol that the real M2 trackers will conform to,
so swapping it in later touches no other file.

NFR-4 (clamp every command to URDF joint limits) is wired in here via an
optional `clamp` callable; see groovebot.limits for the helper used by demo
and tests.
"""
from __future__ import annotations
import time
from typing import Callable, Optional, Protocol

from .types import GrooveContext, JointCommand


class Perception(Protocol):
    """Single tick of perception. Returns the latest GrooveContext.

    M1: metronome + constants. M2: live BeatTracker + ArousalEstimator. The
    control loop sees nothing of that change — only this interface.
    """

    def tick(self, dt: float) -> GrooveContext: ...


class GrooveGenerator(Protocol):
    def generate(self, ctx: GrooveContext) -> JointCommand: ...


class RobotBackend(Protocol):
    def load(self, urdf_path: str) -> None: ...
    def set_joint_targets(self, targets: dict[str, float]) -> None: ...
    def step(self, dt: float) -> None: ...
    def get_joint_states(self) -> dict[str, float]: ...
    def close(self) -> None: ...


class MetronomePerception:
    """M1 stub: synthetic beat clock + constant energy/arousal/valence.

    This is what the orchestrator drives in demos and tests until the real
    BeatTracker / ArousalEstimator land in M2.
    """

    def __init__(self, bpm: float = 120.0, energy: float = 0.8,
                 arousal: float | None = None, valence: float = 0.0,
                 beats_per_bar: int = 4):
        self.bpm = bpm
        self.energy = energy
        self.arousal = energy if arousal is None else arousal
        self.valence = valence
        self.beats_per_bar = beats_per_bar
        self._beat_pos = 0.0
        self._last_beat_int = -1

    def tick(self, dt: float) -> GrooveContext:
        self._beat_pos += dt * self.bpm / 60.0
        beat_int = int(self._beat_pos)
        downbeat = beat_int != self._last_beat_int and (beat_int % self.beats_per_bar == 0)
        self._last_beat_int = beat_int
        return GrooveContext(
            beat_pos=self._beat_pos,
            downbeat=downbeat,
            tempo=self.bpm,
            arousal=self.arousal,
            valence=self.valence,
            energy=self.energy,
        )


class Orchestrator:
    """Fixed-rate control loop: perception -> generator -> (clamp) -> backend.

    `rate` must be 30..50 Hz per NFR-7. `clamp` is optional so this module can
    be tested in isolation; the demo wires in the URDF clamp from
    groovebot.limits.
    """

    def __init__(
        self,
        perception: Perception,
        generator: GrooveGenerator,
        backend: RobotBackend,
        rate: float = 50.0,
        clamp: Optional[Callable[[JointCommand], JointCommand]] = None,
        realtime: bool = False,
    ):
        if not (30.0 <= rate <= 50.0):
            raise ValueError(f"control rate must be 30..50 Hz (NFR-7), got {rate}")
        self.perception = perception
        self.generator = generator
        self.backend = backend
        self.rate = rate
        self.clamp = clamp
        self.realtime = realtime

    def run(self, seconds: float) -> None:
        dt = 1.0 / self.rate
        steps = int(seconds * self.rate)
        for _ in range(steps):
            ctx = self.perception.tick(dt)
            cmd = self.generator.generate(ctx)
            if self.clamp is not None:
                cmd = self.clamp(cmd)
            self.backend.set_joint_targets(cmd.targets)
            self.backend.step(dt)
            if self.realtime:
                time.sleep(dt)
