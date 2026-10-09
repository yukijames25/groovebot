"""Forward-kinematics invariants for the three arm-driving primitives.

The pure-math tests in `test_groove_style.py` check joint-angle shape
(phase, symmetry, amplitude direction). This file checks what those
joint angles actually PRODUCE in world coordinates — did the rewrite
really put the hands overhead for fist_pump, overhead for penlight_wave,
and converging in front for clap, or did we just change numbers?

Lives in its own file so `test_groove_style.py` can keep its "no mujoco
import" guarantee. Each test here calls into
`tools.verify_primitive_pose.measure_primitive`, so the numeric contract
is held in exactly one place: the tool the showcase workflow uses.
"""
from __future__ import annotations

import pytest


# ─── fist_pump: both fists reach overhead on the beat ─────────────────────

def test_fist_pump_hands_reach_overhead():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("fist_pump", intensity=1.0, n_samples=48)
    # Both hands end up above the top of the head cube at the beat peak.
    assert s["hand_z_max_cm"]["L"] >= HEAD_TOP_CM, s["hand_z_max_cm"]
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM, s["hand_z_max_cm"]


@pytest.mark.parametrize("intensity", [0.3, 0.5, 0.7, 1.0])
def test_fist_pump_overhead_at_every_intensity(intensity):
    """The whole point of the 2026-10-14 base/oscillation decoupling:
    the arms are overhead regardless of intensity. Previously, roll
    was multiplied by i, so fist_pump at i=0.5 landed the hands at
    z=20.3 cm, below shoulder. The verifier's intensity sweep catches
    that; this test fixes the contract."""
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("fist_pump", intensity=intensity, n_samples=48)
    assert s["hand_z_max_cm"]["L"] >= HEAD_TOP_CM, (
        intensity, s["hand_z_max_cm"])
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM, (
        intensity, s["hand_z_max_cm"])


def test_fist_pump_drives_both_arms_symmetrically():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("fist_pump", intensity=1.0, n_samples=48)
    # No sway means both arms track the same vertical path through the
    # cycle; the earlier version (single-side) failed this trivially.
    assert s["symmetry_cm"] < 0.5
    # The reach of each arm is the same (no left-vs-right bias).
    assert s["hand_z_max_cm"]["L"] == pytest.approx(
        s["hand_z_max_cm"]["R"], abs=0.5)


def test_fist_pump_stays_inside_urdf_range():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("fist_pump", intensity=1.0, n_samples=48)
    assert s["n_urdf_violations"] == 0, s["urdf_violations"][:3]


# ─── penlight_wave: overhead + bilateral + lateral sway ───────────────────

def test_penlight_wave_both_hands_overhead():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("penlight_wave", intensity=1.0, n_samples=48)
    # Both hands reach above head top (the "両手を頭上で" part).
    assert s["hand_z_max_cm"]["L"] >= HEAD_TOP_CM
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM


@pytest.mark.parametrize("intensity", [0.3, 0.5, 0.7, 1.0])
def test_penlight_wave_overhead_at_every_intensity(intensity):
    """The jp_ballad pathology (intensity 0.51 showing arms half-raised)
    is the raison d'etre of the base/oscillation decoupling. The overhead
    pose must hold at EVERY intensity the showcase could pick."""
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("penlight_wave", intensity=intensity, n_samples=48)
    assert s["hand_z_max_cm"]["L"] >= HEAD_TOP_CM, (
        intensity, s["hand_z_max_cm"])
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM, (
        intensity, s["hand_z_max_cm"])


def test_penlight_wave_both_arms_reach_equal_height():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("penlight_wave", intensity=1.0, n_samples=48)
    # Each arm's max z should match the other's: the rewrite drives
    # them symmetrically. (Instantaneous Δz is nonzero because the body
    # sways; peak reach must be equal.)
    assert s["hand_z_max_cm"]["L"] == pytest.approx(
        s["hand_z_max_cm"]["R"], abs=0.5)


def test_penlight_wave_produces_lateral_sway():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("penlight_wave", intensity=1.0, n_samples=48)
    # The sway makes one arm go up as the other goes down, so the
    # cycle-mean |L_z - R_z| is clearly nonzero. (If this is 0, the
    # body isn't rocking and the hands are static overhead.)
    assert s["symmetry_cm"] > 5.0


def test_penlight_wave_stays_inside_urdf_range():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("penlight_wave", intensity=1.0, n_samples=48)
    assert s["n_urdf_violations"] == 0, s["urdf_violations"][:3]


# ─── arm_swing: right arm overhead, swing is forward (x), not up or lateral

def test_arm_swing_right_hand_reaches_overhead():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("arm_swing", intensity=1.0, n_samples=48)
    # Right hand peaks above head top ("overhead" in the task spec).
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM, s["hand_z_max_cm"]


@pytest.mark.parametrize("intensity", [0.3, 0.5, 0.7, 1.0])
def test_arm_swing_right_hand_overhead_at_every_intensity(intensity):
    """arm_swing's shoulder_roll is a base-pose constant (not scaled by
    i), so the right hand is overhead at ANY intensity. This is the
    strictest form of the decoupling guarantee — the swing amplitude
    is the only thing that changes with i."""
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive, HEAD_TOP_CM

    s = measure_primitive("arm_swing", intensity=intensity, n_samples=48)
    assert s["hand_z_max_cm"]["R"] >= HEAD_TOP_CM, (
        intensity, s["hand_z_max_cm"])


def test_arm_swing_is_unilateral():
    """Left arm is deliberately NOT driven; its travel should be only
    the tiny amount induced by the torso-pitch lean on the beat."""
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("arm_swing", intensity=1.0, n_samples=48)
    L = s["hand_travel_p2p_cm"]["L"]
    R = s["hand_travel_p2p_cm"]["R"]
    # Right arm moves measurably; left arm barely moves.
    assert R["x"] > 10.0, R
    assert L["x"] < 2.0 and L["y"] < 2.0 and L["z"] < 2.0, L


def test_arm_swing_dominant_axis_is_forward():
    """Task requirement: 「振る方向は前後（shoulder_pitch の振動）であること。
    上下や左右ではない」. We assert the x-axis p2p travel of the right
    hand strictly dominates both y and z travel."""
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("arm_swing", intensity=1.0, n_samples=48)
    R = s["hand_travel_p2p_cm"]["R"]
    assert R["x"] > 2.0 * R["z"], (
        f"forward travel {R['x']:.1f} cm should dominate vertical "
        f"{R['z']:.1f} cm by at least 2x")
    assert R["x"] > 2.0 * R["y"], (
        f"forward travel {R['x']:.1f} cm should dominate lateral "
        f"{R['y']:.1f} cm by at least 2x")


def test_arm_swing_stays_inside_urdf_range():
    pytest.importorskip("mujoco")
    from tools.verify_primitive_pose import measure_primitive

    s = measure_primitive("arm_swing", intensity=1.0, n_samples=48)
    assert s["n_urdf_violations"] == 0, s["urdf_violations"][:3]
