"""tools/verify_primitive_pose.py — forward-kinematics check for MOVE_PRIMITIVES.

For each primitive in `groovebot.groove_style.MOVE_PRIMITIVES`, this tool
plays the move through MuJoCo's forward kinematics and reports where the
hands actually end up in world coordinates — so we can tell whether a
move named "fist_pump" actually sends the hands overhead, whether "clap"
brings the hands together in front of the body, and so on.

What it computes (per primitive, per intensity, over one full cycle):

  * `hand_z_max_cm`   : highest point each hand reaches
  * `vs_shoulder_cm`  : hand max height minus shoulder joint height (31 cm)
  * `vs_head_top_cm`  : hand max height minus head-top height (42 cm)
  * `symmetry_cm`     : |L hand z − R hand z| averaged across the cycle
                        (0 = perfectly symmetric left-right driving)
  * `min_hand_sep_cm` : smallest left-to-right hand distance over the cycle
                        (small = hands meet, e.g. clap)
  * `max_forward_cm`  : furthest hand x-position (positive = in front of torso)
  * `urdf_violation`  : any joint outside its URDF range on any sampled pose

The script does NOT exercise the full control loop — it drives the pose
directly from the primitive and reads world coordinates via `mj_forward`.
Pure measurement, no shaping, no clamping (so this catches primitives
that would rely on the orchestrator's safety clamp to stay legal).

Usage:
    python -m tools.verify_primitive_pose                       # all 8 moves
    python -m tools.verify_primitive_pose fist_pump             # one move
    python -m tools.verify_primitive_pose --intensity 0.5
    python -m tools.verify_primitive_pose --json out.json       # machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── geometry constants (hand-measured from robot/groovebot.urdf) ─────────
#
# Shoulder joint origin in base_link frame is at z = 0.31 m (torso_link
# origin 0.11 + l_shoulder_pitch offset 0.20).
# Head cube centre at z = 0.37, half-height 0.05 → top ≈ 0.42 m.
# Forearm link cylinder: length 0.12 m, origin xyz="0 0 -0.06" → tip at
# z = −0.12 m in the link's local frame.
SHOULDER_Z_CM = 31.0
HEAD_TOP_CM   = 42.0
HAND_IN_FOREARM_FRAME = np.array([0.0, 0.0, -0.12])


# ─── pose evaluation ──────────────────────────────────────────────────────

def _load_model():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(_REPO_ROOT / "robot" / "groovebot.urdf"))
    d = mujoco.MjData(m)
    return mujoco, m, d


def _joint_addrs(mujoco, m):
    from groovebot.backend import JOINT_NAMES
    addrs: dict[str, int] = {}
    for n in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError(f"joint {n!r} not in URDF")
        addrs[n] = int(m.jnt_qposadr[jid])
    return addrs


def _joint_ranges(mujoco, m):
    """Return {joint_name: (lower, upper)} from the URDF."""
    from groovebot.backend import JOINT_NAMES
    out: dict[str, tuple[float, float]] = {}
    for n in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = m.jnt_range[jid]
        out[n] = (float(lo), float(hi))
    return out


def _hand_world(mujoco, m, d, side: str) -> np.ndarray:
    """World position of the hand tip for 'l' or 'r'."""
    body = f"{side}_forearm_link"
    bid = mujoco.mj_name2id(mujoco, m, mujoco.mjtObj.mjOBJ_BODY, body) \
          if False else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    pos = np.array(d.xpos[bid])
    R = np.array(d.xmat[bid]).reshape(3, 3)
    return pos + R @ HAND_IN_FOREARM_FRAME


# ─── per-primitive measurement ────────────────────────────────────────────

def measure_primitive(
    name: str,
    *,
    intensity: float = 1.0,
    n_samples: int = 64,
    cycle_beats: float = 2.0,
) -> dict:
    """Sweep one primitive across one cycle and return summary stats.

    `cycle_beats = 2.0` covers the longest standard cycle (clap, sway,
    rock, penlight). 1-beat accent moves repeat twice inside this window.
    `quiet_listen` is 4 beats; we double the span for it below.
    """
    from groovebot.groove_style import MOVE_PRIMITIVES, neutral_pose

    if name not in MOVE_PRIMITIVES:
        raise ValueError(f"unknown primitive {name!r}; "
                         f"choose from {sorted(MOVE_PRIMITIVES)}")

    mujoco, m, d = _load_model()
    qadr = _joint_addrs(mujoco, m)
    ranges = _joint_ranges(mujoco, m)
    prim = MOVE_PRIMITIVES[name]

    if name == "quiet_listen":
        cycle_beats = 4.0

    L_pts: list[np.ndarray] = []
    R_pts: list[np.ndarray] = []
    violations: list[tuple[float, str, float, tuple[float, float]]] = []

    for k in range(n_samples):
        b = cycle_beats * k / n_samples

        d.qpos[:] = 0.0
        pose = neutral_pose()
        delta = prim(b, intensity)
        pose.update(delta)

        # Check URDF range before applying — this catches primitives
        # whose raw output would exceed the hard limit even before the
        # orchestrator's clamp runs.
        for jn, v in pose.items():
            lo, hi = ranges[jn]
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                violations.append((b, jn, v, (lo, hi)))
            d.qpos[qadr[jn]] = max(lo, min(hi, v))  # clamp for FK only

        mujoco.mj_forward(m, d)
        L_pts.append(_hand_world(mujoco, m, d, "l"))
        R_pts.append(_hand_world(mujoco, m, d, "r"))

    L = np.stack(L_pts) * 100.0   # metres → centimetres
    R = np.stack(R_pts) * 100.0

    sep = np.linalg.norm(L - R, axis=1)
    symmetry = float(np.mean(np.abs(L[:, 2] - R[:, 2])))

    # Per-axis peak-to-peak hand travel. Used to characterise WHICH
    # direction a swinging move actually swings in — forward (x), lateral
    # (y), or vertical (z). Added 2026-10-11 when arm_swing was added:
    # the task requires the swing to be in x (not y or z), and we want
    # a direct number to assert on.
    def _p2p(a: np.ndarray) -> dict[str, float]:
        return {
            "x": float(a[:, 0].max() - a[:, 0].min()),
            "y": float(a[:, 1].max() - a[:, 1].min()),
            "z": float(a[:, 2].max() - a[:, 2].min()),
        }

    return {
        "name":              name,
        "intensity":         float(intensity),
        "n_samples":         n_samples,
        "cycle_beats":       cycle_beats,
        "hand_z_max_cm":     {"L": float(L[:, 2].max()), "R": float(R[:, 2].max())},
        "hand_z_min_cm":     {"L": float(L[:, 2].min()), "R": float(R[:, 2].min())},
        "vs_shoulder_cm":    {"L": float(L[:, 2].max() - SHOULDER_Z_CM),
                              "R": float(R[:, 2].max() - SHOULDER_Z_CM)},
        "vs_head_top_cm":    {"L": float(L[:, 2].max() - HEAD_TOP_CM),
                              "R": float(R[:, 2].max() - HEAD_TOP_CM)},
        "symmetry_cm":       symmetry,
        "min_hand_sep_cm":   float(sep.min()),
        "max_hand_sep_cm":   float(sep.max()),
        "max_forward_cm":    float(max(L[:, 0].max(), R[:, 0].max())),
        "min_forward_cm":    float(min(L[:, 0].min(), R[:, 0].min())),
        "hand_travel_p2p_cm": {"L": _p2p(L), "R": _p2p(R)},
        "urdf_violations":   [
            {"beat": b, "joint": j, "value": v, "range": [lo, hi]}
            for b, j, v, (lo, hi) in violations
        ],
        "n_urdf_violations": len(violations),
    }


# ─── pretty print ─────────────────────────────────────────────────────────

def _mark_vs_head(dz: float) -> str:
    """Short badge for 'hand max vs head top'."""
    if dz >= 0.0:   return "overhead"
    if dz >= -5.0:  return "near head"
    if dz >= -15.0: return "below head"
    return "well below"


def _mark_meeting(sep: float) -> str:
    """Short badge for 'how close do the hands get'."""
    if sep <= 10.0:  return "touching"
    if sep <= 20.0:  return "close"
    return "apart"


def print_table(results: list[dict]) -> None:
    print()
    print("=" * 118)
    print(f"MOVE_PRIMITIVES hand-tip verification  "
          f"(shoulder joint z = {SHOULDER_Z_CM:.0f} cm, head top ≈ {HEAD_TOP_CM:.0f} cm)")
    print("=" * 118)
    print(f"  {'primitive':<16} "
          f"{'L z_max':>8} {'R z_max':>8} "
          f"{'vs shoulder (L/R)':>18} "
          f"{'vs head (L/R)':>16} "
          f"{'sym Δz':>8} "
          f"{'min sep':>8} "
          f"{'fwd max':>8}  "
          f"{'verdict':<24}")
    print("  " + "-" * 112)

    for r in results:
        lz = r["hand_z_max_cm"]["L"]
        rz = r["hand_z_max_cm"]["R"]
        ls = r["vs_shoulder_cm"]["L"]
        rs = r["vs_shoulder_cm"]["R"]
        lh = r["vs_head_top_cm"]["L"]
        rh = r["vs_head_top_cm"]["R"]
        sym = r["symmetry_cm"]
        sep = r["min_hand_sep_cm"]
        fwd = r["max_forward_cm"]
        verdict = f"{_mark_vs_head(max(lh, rh))} · {_mark_meeting(sep)}"
        if r["n_urdf_violations"] > 0:
            verdict = f"URDF x{r['n_urdf_violations']} · " + verdict
        print(f"  {r['name']:<16} "
              f"{lz:>7.1f}  {rz:>7.1f}  "
              f"{ls:>+7.1f}/{rs:>+6.1f}  "
              f"{lh:>+6.1f}/{rh:>+6.1f}  "
              f"{sym:>7.2f}  "
              f"{sep:>7.1f}  "
              f"{fwd:>+7.1f}   "
              f"{verdict:<24}")

    print("=" * 118)
    print("Columns:  z (cm) is world height from base plate.")
    print("  'vs shoulder' / 'vs head' are the hands' max z minus 31 cm / 42 cm.")
    print("  'sym Δz' is mean |L_z - R_z|; 0.0 cm = perfectly symmetric driving.")
    print("  'min sep' = closest L-to-R hand distance across the cycle.")
    print("  'fwd max' is the furthest-forward x; positive is in front of the torso.")
    print("  Any URDF range violation is flagged in the verdict column.")
    print()


# ─── entry point ──────────────────────────────────────────────────────────

def print_intensity_sweep(sweeps: dict[str, list[dict]]) -> None:
    """One row per primitive, one column per intensity, each cell shows
    the right-hand (or dominant-hand) z_max in cm and the Δ vs head top.

    Highlights where a move only meets the "overhead" requirement at a
    specific intensity — the pathology the task was built to detect
    (penlight_wave was reported overhead at i=1.0 but drops below head
    at i=0.51, which is where the showcase actually drives it).
    """
    first = next(iter(sweeps.values()))
    intensities = [r["intensity"] for r in first]

    print()
    print("=" * 118)
    print(f"Hand reach vs intensity "
          f"(right-hand z_max cm / Δ to head top {HEAD_TOP_CM:.0f} cm; "
          f"L→left-hand z_max in brackets when driven)")
    print("=" * 118)

    header = "  primitive       " + "   ".join(
        f"i={i:4.2f}" for i in intensities
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, series in sweeps.items():
        cells = []
        for r in series:
            Rz = r["hand_z_max_cm"]["R"]
            Lz = r["hand_z_max_cm"]["L"]
            dh = r["vs_head_top_cm"]["R"]
            # pick whichever hand goes higher at this intensity
            z = max(Rz, Lz)
            dh_eff = max(dh, r["vs_head_top_cm"]["L"])
            mark = "↑" if dh_eff >= 0 else " "
            cells.append(f"{z:5.1f}/{dh_eff:+5.1f}{mark}")
        print(f"  {name:<16}" + "  ".join(cells))

    print()
    print("  '↑' marks intensities where at least one hand reaches above")
    print("  head top (42 cm). A row that only shows '↑' at i=1.0 means")
    print("  the move's overhead requirement is intensity-dependent — the")
    print("  base-pose/oscillation decoupling was supposed to fix that.")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("primitives", nargs="*",
                    help="One or more primitive names. Default: all 8.")
    ap.add_argument("--intensity", type=float, default=None,
                    help="Legacy single-intensity flag. If omitted, "
                         "runs the full sweep from --intensities.")
    ap.add_argument("--intensities", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 1.0],
                    help="Sweep of intensities to measure at (default: "
                         "0.3 0.5 0.7 1.0).")
    ap.add_argument("--samples", type=int, default=64,
                    help="Beat positions sampled per cycle (default 64).")
    ap.add_argument("--json", default=None,
                    help="Also write machine-readable results here.")
    args = ap.parse_args(argv)

    from groovebot.groove_style import MOVE_PRIMITIVES
    names = args.primitives or list(MOVE_PRIMITIVES.keys())
    unknown = [n for n in names if n not in MOVE_PRIMITIVES]
    if unknown:
        print(f"unknown primitive(s): {unknown}; "
              f"choose from {sorted(MOVE_PRIMITIVES)}", file=sys.stderr)
        return 2

    # Backward-compat: --intensity N runs a single-intensity table (as the
    # tool did before the sweep landed). Default behaviour is the sweep.
    if args.intensity is not None:
        results = [
            measure_primitive(n, intensity=args.intensity,
                              n_samples=args.samples)
            for n in names
        ]
        print_table(results)
        if args.json:
            Path(args.json).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(f"[verify_primitive_pose] json → {args.json}", flush=True)
        return 0

    # Multi-intensity sweep.
    sweeps: dict[str, list[dict]] = {}
    for n in names:
        sweeps[n] = [
            measure_primitive(n, intensity=float(i),
                              n_samples=args.samples)
            for i in args.intensities
        ]

    # Full detail table at the top intensity for a quick orientation…
    top_intensity = max(args.intensities)
    print_table([
        measure_primitive(n, intensity=top_intensity,
                          n_samples=args.samples)
        for n in names
    ])
    # …then the sweep that is the point of this tool.
    print_intensity_sweep(sweeps)

    if args.json:
        Path(args.json).write_text(
            json.dumps({
                "intensities": list(args.intensities),
                "sweeps": sweeps,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"[verify_primitive_pose] json → {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
