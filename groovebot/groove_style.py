"""
groove_style.py — JointCommand bridge v1 (style-conditioned groove → joints).

Pipeline (M2-shape, ports & adapters):

    GrooveStyleSelector.select(audio_window) -> GrooveStyle
                                                    │  (slow, 5-10 s window)
                                                    ▼
                                  StyleGrooveGenerator.set_style(style)
                                                    │
                       Orchestrator (30-50 Hz) ─► generate(ctx) ─► JointCommand

The selector runs on a slow audio window. The generator runs at the control
rate. They are decoupled by `set_style()`: the selector pushes a new style
when it wants (in M2 this will be every few seconds), the generator picks
it up at the next tick boundary. `ctx.beat_pos` (from the perception's
tempo clock — `MetronomePerception` today, `ReferenceAligner` in M2)
drives the per-tick phase.

Spec hooks (docs/SYSTEM_SPEC.md):
- §5.1 GrooveContext / JointCommand contract is unchanged.
- §5.2 GrooveGenerator.generate(ctx) -> JointCommand is the only entry.
- §14 v1 bridge section explains the design and honest limits.
- NFR-4: primitive amplitudes are sized so output stays inside the URDF
  joint range for intensity ∈ [0, 1]. The orchestrator's clamp is still
  wired downstream as the hard safety guard.
"""
from __future__ import annotations
import math
from typing import Callable, Mapping

from .backend import JOINT_NAMES
from .orchestrator import MetronomePerception
from .types import GrooveContext, JointCommand
from .style.select import GrooveStyle


# Per-joint soft amplitude ceiling. URDF (robot/groovebot.urdf) is the
# hard limit — these sit a comfortable margin inside so commands don't
# ride the rail under intensity=1.0. The downstream `clamp_command`
# (NFR-4) is still the authority on safety; this is the design ceiling.
SOFT_AMP = {
    "neck_pitch": 0.55,           # URDF ±0.7
    "neck_yaw": 0.50,             # URDF ±1.2
    "torso_roll": 0.40,           # URDF ±0.45
    "torso_pitch": 0.40,          # URDF ±0.45
    "l_shoulder_pitch": 2.00,     # URDF ±2.2
    "r_shoulder_pitch": 2.00,
    "l_shoulder_roll": 2.40,      # URDF [-0.2, 2.6]
    "r_shoulder_roll": 2.40,      # URDF [-2.6, 0.2]
    "l_elbow": 2.30,              # URDF [0, 2.4]
    "r_elbow": 2.30,
}


def neutral_pose() -> dict[str, float]:
    """Rest pose: every driven joint at 0.0.

    The URDF default already has arms relaxed at sides at q=0, so this is
    a safe rest. Primitives only *replace* the joints they touch — the
    rest stay at neutral.
    """
    return {n: 0.0 for n in JOINT_NAMES}


# ----------------------------------------------------------- primitives
# Each primitive is (beat_pos, intensity) -> dict[str, float]. Only the
# joints the primitive drives appear; the generator fills the rest with
# the neutral pose so every emitted JointCommand is complete.
#
# Conventions:
#   - beat_pos is musical position in beats (same as ctx.beat_pos).
#     Integer b is the down/click of each beat; b=0 is the first beat.
#   - intensity ∈ [0, 1] scales amplitude.
#   - At intensity=1.0 every value stays inside the URDF limit.
#
# Beat-phase contract (v1.2, see docs/SYSTEM_SPEC.md §14 v1):
#
#   move          | category   | cycle  | peak / accent lands at
#   --------------|------------|--------|---------------------------------
#   headbang      | accent     | 1 beat | on every beat (b = 0,1,2,...)
#   bob_nod       | accent     | 1 beat | on every beat
#   fist_pump     | accent     | 1 beat | on every beat
#   clap          | accent     | 2 beat | on musical beats 2 & 4
#                 |            |        |   (b = 1,3,5,... in 0-indexed
#                 |            |        |    beat_pos = backbeat)
#   sway          | continuous | 2 beat | zero-cross on each beat;
#                 |            |        |   peak at half-beat (sin(πb))
#   rock          | continuous | 2 beat | same convention as sway
#   penlight_wave | continuous | 2 beat | same convention as sway
#   quiet_listen  | minimal    | 4 beat | breathing only, no accent
#
# All accent moves use cos(2π·b) (period 1 beat) phased so that the
# accent samples ON the beat. Clap uses cos(π·b) (period 2 beats) phased
# so that accents land on b=1,3,... — the backbeat. Continuous moves use
# sin(π·b) so they cross the center on every beat and reach their
# extreme between beats: this is the "in-the-pocket" sway feel and is a
# deliberate, consistent phase choice across sway/rock/penlight_wave.

Primitive = Callable[[float, float], dict[str, float]]


def _prim_headbang(b: float, i: float) -> dict[str, float]:
    """Fast forward nod, head down ON every beat (metal-style)."""
    amp = 0.55 * i
    nod = -amp * 0.5 * (1.0 + math.cos(2.0 * math.pi * b))
    return {"neck_pitch": nod}


def _prim_bob_nod(b: float, i: float) -> dict[str, float]:
    """Small head bob, peak nod ON every beat."""
    amp = 0.25 * i
    nod = -amp * 0.5 * (1.0 + math.cos(2.0 * math.pi * b))
    return {"neck_pitch": nod}


def _prim_sway(b: float, i: float) -> dict[str, float]:
    """Side-to-side torso sway, one full sway per 2 beats.

    Continuous oscillation: torso crosses center on each integer beat
    and hits its extreme at the half-beat. The beat is the carrier of
    motion direction, not an accent.
    """
    amp = 0.35 * i
    roll = amp * math.sin(math.pi * b)
    return {"torso_roll": roll}


def _prim_rock(b: float, i: float) -> dict[str, float]:
    """Forward-back torso rock, one full rock per 2 beats (continuous)."""
    amp = 0.30 * i
    pitch = amp * math.sin(math.pi * b)
    return {"torso_pitch": pitch}


def _prim_fist_pump(b: float, i: float) -> dict[str, float]:
    """Both arms raised, shoulder pitch pulses peak ON every beat.

    Every term carries the intensity factor so i=0 collapses to the
    neutral pose. The arms come up as i grows, and the pulse peaks on
    the beat.
    """
    pulse = 0.5 * (1.0 + math.cos(2.0 * math.pi * b))           # peak=1 at b=0,1,2,...
    sh_pitch = (-1.5 - 0.40 * pulse) * i                        # i=1: -1.9 on-beat, -1.5 off
    return {
        "l_shoulder_pitch": sh_pitch,
        "r_shoulder_pitch": sh_pitch,
        "l_shoulder_roll": 0.40 * i,
        "r_shoulder_roll": -0.40 * i,
        "l_elbow": (0.5 + 0.5 * pulse) * i,                     # i=1: 1.0 on-beat, 0.5 off
        "r_elbow": (0.5 + 0.5 * pulse) * i,
    }


def _prim_clap(b: float, i: float) -> dict[str, float]:
    """Hands forward, clap accents on musical beats 2 & 4 (the backbeat).

    Period is 2 beats: pulse=0 at b=0,2,4,... (musical beats 1,3, the
    "and" / rest) and pulse=1 at b=1,3,5,... (musical beats 2,4, the
    clap moment). Every term carries intensity so i=0 returns neutral.
    """
    pulse = 0.5 * (1.0 - math.cos(math.pi * b))                 # peak=1 at b=1,3,5,...
    sh_pitch = -0.6 * i
    l_roll = (1.0 + 0.6 * pulse) * i                            # i=1: 1.0 rest, 1.6 clap
    r_roll = -(1.0 + 0.6 * pulse) * i
    elbow = (1.2 + 0.6 * pulse) * i                             # i=1: 1.2 rest, 1.8 clap
    return {
        "l_shoulder_pitch": sh_pitch,
        "r_shoulder_pitch": sh_pitch,
        "l_shoulder_roll": l_roll,
        "r_shoulder_roll": r_roll,
        "l_elbow": elbow,
        "r_elbow": elbow,
    }


def _prim_penlight_wave(b: float, i: float) -> dict[str, float]:
    """One arm raised high, swaying with the bar (continuous, 2-beat cycle).

    Same phase convention as sway/rock: sin(π·b) crosses zero on each
    integer beat, hits the extreme between beats. Every term carries
    intensity → i=0 returns the neutral pose.
    """
    wave = math.sin(math.pi * b)                                # ±1, period 2 beats
    return {
        "l_shoulder_pitch": -1.6 * i,
        "l_shoulder_roll": (1.4 + 0.4 * wave) * i,              # i=1: [1.0, 1.8]
        "l_elbow": 0.6 * i,
        "neck_yaw": 0.40 * i * wave,
        "torso_roll": 0.20 * i * wave,
    }


def _prim_quiet_listen(b: float, i: float) -> dict[str, float]:
    """Near-pose breathing motion only. Capped tiny even at intensity 1."""
    amp = 0.05 * max(i, 0.1)
    nod = amp * math.sin(math.pi * b / 2.0)                     # very slow
    return {"neck_pitch": nod}


MOVE_PRIMITIVES: dict[str, Primitive] = {
    "headbang":      _prim_headbang,
    "bob_nod":       _prim_bob_nod,
    "sway":          _prim_sway,
    "rock":          _prim_rock,
    "fist_pump":     _prim_fist_pump,
    "clap":          _prim_clap,
    "penlight_wave": _prim_penlight_wave,
    "quiet_listen":  _prim_quiet_listen,
}


# ------------------------------------------------------ the generator

class StyleGrooveGenerator:
    """Style-conditioned rule generator.

    `set_style(style)` swaps the current `GrooveStyle` at any tick
    boundary. The selector runs on a slow window, so updates are
    infrequent compared to the 30-50 Hz control loop — there is no
    timing coupling between them.

    Parameters
    ----------
    style :
        Initial style; may be None, in which case `generate()` returns
        the neutral pose until `set_style()` is called.
    primitives :
        Override the default move → primitive map (mostly for tests).
    use_ctx_arousal :
        If True, multiply `style.intensity` by `0.5 + 0.5 * ctx.arousal`
        so that live ArousalEstimator output (M2) can modulate motion
        size on top of the table-set intensity. Off by default so v1 is
        deterministic from the selector alone.
    """

    def __init__(
        self,
        style: GrooveStyle | None = None,
        primitives: Mapping[str, Primitive] | None = None,
        use_ctx_arousal: bool = False,
    ):
        self._style = style
        self._primitives: dict[str, Primitive] = (
            dict(primitives) if primitives is not None else dict(MOVE_PRIMITIVES)
        )
        self.use_ctx_arousal = bool(use_ctx_arousal)

    def set_style(self, style: GrooveStyle) -> None:
        self._style = style

    @property
    def style(self) -> GrooveStyle | None:
        return self._style

    def generate(self, ctx: GrooveContext) -> JointCommand:
        targets = neutral_pose()
        style = self._style
        if style is None:
            return JointCommand(targets=targets)

        intensity = float(style.intensity)
        if self.use_ctx_arousal:
            intensity *= 0.5 + 0.5 * float(ctx.arousal)
        intensity = max(0.0, min(1.0, intensity))

        prim = self._primitives.get(style.move)
        if prim is None:                       # unknown move → stay neutral
            return JointCommand(targets=targets)

        targets.update(prim(float(ctx.beat_pos), intensity))
        return JointCommand(targets=targets)


# ------------------------------------------- perception helper for v1

def metronome_from_style(
    style: GrooveStyle,
    *,
    beats_per_bar: int = 4,
    floor_bpm: float = 40.0,
) -> MetronomePerception:
    """Build a `MetronomePerception` whose tempo comes from `style.tempo_bpm`.

    The v1 bridge uses the metronome as a stand-in beat source until M2
    lands the real `ReferenceAligner`. We bound the BPM by `floor_bpm`
    so that a degenerate tempo estimate (e.g. 0) does not freeze the
    clock — the orchestrator still ticks, the perception just runs slow.

    Energy/arousal are forwarded from the style so that the orchestrator's
    GrooveContext carries a coherent affect snapshot for any downstream
    modulator (e.g. `StyleGrooveGenerator(use_ctx_arousal=True)`).
    """
    bpm = max(float(floor_bpm), float(style.tempo_bpm))
    return MetronomePerception(
        bpm=bpm,
        energy=float(style.intensity),
        arousal=float(style.arousal),
        valence=0.0,
        beats_per_bar=beats_per_bar,
    )
