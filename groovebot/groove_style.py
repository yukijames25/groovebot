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
    """Both fists overhead, pulsing upward ON every beat.

    Rewrite (2026-10-01): the earlier version left the fists below
    shoulder height (hand z = 27.5 cm vs shoulder 31 cm, head top 42 cm)
    because it used shoulder_pitch to swing the arms forward instead of
    using shoulder_roll to lift them. Measured by
    `tools/verify_primitive_pose.py` → forward-kinematics world coords.

    Design (verified by `tools/verify_primitive_pose.py` across the
    intensity sweep 0.3 / 0.5 / 0.7 / 1.0 so hand reach is checked at
    the operating points, not only at the ceiling):

      Base pose — NOT scaled by intensity, held at ALL i ≥ 0 so even
      low-energy picks still read as "arms overhead":
        - shoulder_roll ±2.15 (max |roll| < SOFT_AMP 2.40,
          well inside the URDF ±2.60): the overhead abduction that
          puts both elbows above the head.
        - shoulder_pitch base −0.10: slight forward tilt so the fists
          are visible from the front.
        - elbow 1.80 at rest-of-cycle: deeply bent, fists near the
          head (bicep-up pose).

      Oscillation — SCALED by intensity (rides on `pulse`):
        - Elbow extends from 1.80 toward 0 on the beat. At i=1 the
          extension is full (1.80 → 0.00); at i=0.5 the elbow only
          opens to 0.90; at i=0 the elbow stays at 1.80 (frozen
          overhead-bent). The hand's forward-and-up reach scales with
          i while the overhead posture is maintained.
        - A small forward pitch pulse (−0.30·pulse·i) adds a thrust
          component so narration's "primary joint peak" picks the
          beat correctly for accent-category labelling.

      Right side mirrors (negated roll). Phase contract: accent peak
      ON each beat via (1+cos 2πb)/2 (1-beat cycle).

    Rewrite 2026-10-14: the previous version multiplied shoulder_roll
    by intensity, so at i=0.5 the roll was only 1.075 — arms barely
    half-raised. The verifier's intensity sweep now catches that: a
    pre-rewrite row showed `fist_pump` at i=0.5 reached z=20.3 cm,
    −21.7 cm BELOW head top. Decoupling pins the overhead pose
    independent of i.
    """
    pulse = 0.5 * (1.0 + math.cos(2.0 * math.pi * b))           # peak=1 at b=0,1,2,...
    # Base pose joints (NOT scaled by i) — fists already overhead at i=0.
    # The base elbow 1.00 is moderately bent so the pulse's -1.00
    # extension can still read as a thrust; a tighter base (e.g. 1.80)
    # left the fist 6 cm below head top at low intensity.
    base_shoulder_pitch =  -0.10
    base_shoulder_roll_mag = 2.25
    base_elbow            =  1.00
    # Oscillation terms (scaled by i)
    pitch_offset = -0.30 * pulse * i                            # i=1: [-0.30, 0.00]
    elbow_offset = -1.00 * pulse * i                            # i=1: [-1.00, 0.00]
    return {
        "l_shoulder_pitch": base_shoulder_pitch + pitch_offset,
        "r_shoulder_pitch": base_shoulder_pitch + pitch_offset,
        "l_shoulder_roll":  base_shoulder_roll_mag,              # constant
        "r_shoulder_roll": -base_shoulder_roll_mag,              # constant
        "l_elbow": base_elbow + elbow_offset,
        "r_elbow": base_elbow + elbow_offset,
    }


def _prim_clap(b: float, i: float) -> dict[str, float]:
    """Both hands meet in front of the chest on musical beats 2 & 4.

    RETIRED (2026-10-11, NOT WIRED INTO MOVE_PRIMITIVES): the current
    URDF cannot physically realise a two-handed clap at the body
    midline. The l_shoulder_roll range is [-0.2, +2.6] — essentially no
    adduction — and the elbow axis is Y, so the forearm hinges in the
    sagittal plane, not across the chest. The best this primitive can
    achieve is a 20 cm minimum hand separation in front of the body at
    peak pulse: hands come near each other but never meet. Measured by
    `tools/verify_primitive_pose.py`.

    The function is kept so that a future URDF revision — adding a
    shoulder-yaw / internal-rotation axis, or widening l_shoulder_roll
    to a negative range so the arm can cross the midline — can revive
    it by re-registering under `MOVE_PRIMITIVES["clap"]`. See
    `docs/SYSTEM_SPEC.md §4.4` for the hardware-design implication.

    Phase contract (as last authored): 2-beat period with the accent at
    b=1,3,… (musical beats 2 & 4 = backbeat), via (1−cos πb)/2.
    """
    pulse = 0.5 * (1.0 - math.cos(math.pi * b))                 # peak=1 at b=1,3,5,...
    sh_pitch = -1.50 * i                                        # arms horizontal forward
    roll_mag = (0.55 - 0.55 * pulse) * i                        # 0.55 rest -> 0.00 at clap
    elbow    = (1.10 + 1.10 * pulse) * i                        # 1.10 rest -> 2.20 at clap
    return {
        "l_shoulder_pitch": sh_pitch,
        "r_shoulder_pitch": sh_pitch,
        "l_shoulder_roll":   roll_mag,
        "r_shoulder_roll":  -roll_mag,
        "l_elbow": elbow,
        "r_elbow": elbow,
    }


def _prim_penlight_wave(b: float, i: float) -> dict[str, float]:
    """Both arms overhead, swaying left-right together (continuous, 2 beats).

    Rewrite (2026-10-01): the earlier version drove only the LEFT arm and
    peaked at z = 30 cm (below the shoulder joint), so the name
    "両手を頭上で大きくゆっくり振る" was aspirational, not literal. The
    rewrite drives both arms symmetrically with shoulder_roll near the
    URDF ceiling, so the hands sit above head-top; the "wave" is a
    torso-and-neck lateral sway (sin πb) that moves the overhead-held
    arms together from side to side.

    Measured by `tools/verify_primitive_pose.py` across the intensity
    sweep 0.3 / 0.5 / 0.7 / 1.0:
      - L and R hand max z ≈ 51 cm at EVERY intensity (head top 42,
        shoulder 31). The arms are overhead regardless of i.
      - Sway amplitude scales with i: torso_roll 0.30·wave·i and
        neck_yaw 0.40·wave·i → bigger sway at higher i, no sway at i=0.
    Phase contract preserved: continuous sin(πb) means zero at integer
    beats, extrema at the half-beat — same convention as sway/rock.

    Rewrite 2026-10-14 (base-pose decoupling): previously multiplied
    shoulder_roll by i, so at jp_ballad's intensity 0.51 the roll was
    only 1.02 rad and the hands sat ~22 cm — well below head. The
    verifier's intensity sweep showed penlight_wave hands at z=21.6 cm
    (−20.4 vs head top) at i=0.5 under the old formula. The decoupling
    pins the overhead posture independent of i; only the sway amplitude
    scales now.
    """
    wave = math.sin(math.pi * b)                                # ±1, period 2 beats
    return {
        # Base pose (NOT scaled by i): both arms overhead and abducted,
        # elbow nearly straight, slight forward tilt for front visibility.
        # Roll 2.10 (up from 2.00) lifts the i=0 hand above head top; the
        # oscillation contributions only add to that.
        "l_shoulder_pitch": -0.10,
        "r_shoulder_pitch": -0.10,
        "l_shoulder_roll":   2.10,
        "r_shoulder_roll":  -2.10,
        "l_elbow": 0.20,
        "r_elbow": 0.20,
        # Oscillation (SCALED by i): lateral sway via torso + neck.
        "torso_roll": 0.30 * wave * i,
        "neck_yaw":   0.40 * wave * i,
    }


def _prim_quiet_listen(b: float, i: float) -> dict[str, float]:
    """Near-pose breathing motion only. Capped tiny even at intensity 1."""
    amp = 0.05 * max(i, 0.1)
    nod = amp * math.sin(math.pi * b / 2.0)                     # very slow
    return {"neck_pitch": nod}


def _prim_arm_swing(b: float, i: float) -> dict[str, float]:
    """Right arm raised overhead, swinging in the sagittal plane on each beat.

    Added 2026-10-11 to carve out the "one-arm overhead point/wave" slot
    that `fist_pump` (bilateral vertical extend) does not fill. Kinematic
    chain idea: `r_shoulder_pitch` is applied BEFORE `r_shoulder_roll`
    in the URDF, so a small pitch oscillation at rest-of-cycle (arm
    hanging) becomes a FORWARD/BACKWARD swing once the roll lifts the
    arm overhead. The verifier confirms the sweep is dominant in the
    x-axis with minimal lateral (y) or vertical (z) travel — the "振る
    方向は前後" requirement from the task.

    Design (verified by `tools/verify_primitive_pose.py` across the
    intensity sweep 0.3 / 0.5 / 0.7 / 1.0):

      Base pose — NOT scaled by i:
        - r_shoulder_roll = −2.20 (overhead-right abduction, inside URDF
          [-2.60, +0.20] and under SOFT_AMP 2.40)
        - r_elbow = 0.05 (nearly straight)
        - Left arm untouched (hangs neutral) — unilateral by design.

      Oscillation — SCALED by i:
        - r_shoulder_pitch rides on `pulse`: it is 0 at i=0, and at i=1
          swings from +0.30 (rest, lean back) to −0.60 (peak, thrust
          forward). Scaled by i so lower intensities give proportionally
          smaller forward/back swings while the arm stays overhead.
        - `neck_pitch` and `torso_pitch` add forward-lean / nod on the
          beat, both scaled by i so a quiet pick stays still.

    Phase contract: accent category, 1-beat cycle, peak ON every beat
    via (1+cos 2πb)/2 — same as headbang / bob_nod / fist_pump.

    Rewrite 2026-10-14 (base-pose decoupling): previously multiplied
    r_shoulder_roll and r_elbow by i, so at i=0.5 the arm was only
    half-raised. The verifier's intensity sweep showed arm_swing hands
    at z=19.9 cm (−22.1 vs head top) at i=0.5 under the old formula.
    The decoupling pins the overhead posture independent of i; only
    the pitch swing and head/torso lean scale now.
    """
    pulse = 0.5 * (1.0 + math.cos(2.0 * math.pi * b))           # peak=1 at b=0,1,2,...
    return {
        # Oscillation (SCALED by i): pitch swing between rest (+0.30·i)
        # and forward thrust (−0.60·i).
        "r_shoulder_pitch": (0.30 - 0.90 * pulse) * i,
        # Base pose (NOT scaled by i): overhead-right, straight forearm.
        "r_shoulder_roll":  -2.20,
        "r_elbow":           0.05,
        # Head and body nod/lean on the beat (SCALED).
        "neck_pitch":  -0.20 * pulse * i,
        "torso_pitch": -0.15 * pulse * i,
    }


MOVE_PRIMITIVES: dict[str, Primitive] = {
    "headbang":      _prim_headbang,
    "bob_nod":       _prim_bob_nod,
    "sway":          _prim_sway,
    "rock":          _prim_rock,
    "fist_pump":     _prim_fist_pump,
    # "clap" is retired as of 2026-10-11: the URDF cannot bring the two
    # hands to the body midline (l_shoulder_roll has no negative range,
    # the elbow axis is Y). `_prim_clap()` is kept in the module so a
    # future URDF revision can re-register it, but it is deliberately
    # NOT in this dict so no selector path can surface it.
    "penlight_wave": _prim_penlight_wave,
    "arm_swing":     _prim_arm_swing,
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
