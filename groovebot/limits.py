"""
limits.py — URDF-derived joint limits + clamp helper (NFR-4).

Spec NFR-4: every JointCommand the brain produces must be clamped to the URDF
joint range before it is sent to the body. The URDF is the single source of
truth, so we read it directly rather than hardcoding numbers in Python.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Iterable

from .types import JointCommand


JointLimits = dict[str, tuple[float, float]]


def load_joint_limits(urdf_path: str,
                      joint_names: Iterable[str] | None = None) -> JointLimits:
    """Parse a URDF and return {joint_name: (lower, upper)} for revolute /
    prismatic joints. If `joint_names` is given, only those joints are returned
    and a KeyError is raised if any is missing — useful for guarding against
    URDF/JOINT_NAMES drift.
    """
    root = ET.parse(urdf_path).getroot()
    limits: JointLimits = {}
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype not in ("revolute", "prismatic"):
            continue
        name = joint.get("name")
        limit_el = joint.find("limit")
        if name is None or limit_el is None:
            continue
        lower = float(limit_el.get("lower", "0"))
        upper = float(limit_el.get("upper", "0"))
        limits[name] = (lower, upper)

    if joint_names is not None:
        wanted = list(joint_names)
        missing = [n for n in wanted if n not in limits]
        if missing:
            raise KeyError(f"URDF {urdf_path} missing limits for: {missing}")
        limits = {n: limits[n] for n in wanted}

    return limits


def clamp_command(cmd: JointCommand, limits: JointLimits) -> JointCommand:
    """Return a new JointCommand whose targets are all within `limits`.

    Joints not present in `limits` pass through unchanged (so this helper can
    be used with a subset of joints, e.g. when wiring up partial bodies). The
    caller is responsible for ensuring `limits` covers what the body needs.
    """
    clamped: dict[str, float] = {}
    for name, value in cmd.targets.items():
        if name in limits:
            lo, hi = limits[name]
            if value < lo:
                value = lo
            elif value > hi:
                value = hi
        clamped[name] = value
    return JointCommand(targets=clamped)


def make_clamp(urdf_path: str,
               joint_names: Iterable[str] | None = None):
    """Convenience: load limits once and return a `(cmd) -> cmd` callable for
    plugging into Orchestrator(clamp=...).
    """
    limits = load_joint_limits(urdf_path, joint_names=joint_names)
    def _clamp(cmd: JointCommand) -> JointCommand:
        return clamp_command(cmd, limits)
    return _clamp
