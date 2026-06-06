"""
demo_groove.py — minimal end-to-end loop (M1).

    Perception (stub: metronome + constant energy)
        -> RuleGrooveGenerator
        -> clamp to URDF limits (NFR-4)
        -> RobotBackend (MuJoCo / PyBullet)

The Perception stub is the spec §6 seam M2 will replace with a live
BeatTracker + ArousalEstimator. The RuleGrooveGenerator is the §5.2 seam M3
will replace with the trained model. Neither replacement touches this file.

Run:
    python demo_groove.py --bpm 120 --energy 0.8 --seconds 8
    python demo_groove.py --backend pybullet --gui          # dynamics + window
"""
from __future__ import annotations
import argparse, os

from groovebot.backend import MujocoBackend, PyBulletBackend, JOINT_NAMES
from groovebot.groove import RuleGrooveGenerator
from groovebot.limits import make_clamp
from groovebot.orchestrator import MetronomePerception, Orchestrator

URDF = os.path.join(os.path.dirname(__file__), "robot", "groovebot.urdf")


def make_backend(kind: str, gui: bool):
    if kind == "mujoco":
        return MujocoBackend()
    if kind == "pybullet":
        return PyBulletBackend(gui=gui)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mujoco", "pybullet"], default="mujoco")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--energy", type=float, default=0.8)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=50.0, help="control rate (Hz, 30-50)")
    args = ap.parse_args()

    backend = make_backend(args.backend, args.gui)
    backend.load(URDF)

    perception = MetronomePerception(bpm=args.bpm, energy=args.energy)
    generator = RuleGrooveGenerator()
    clamp = make_clamp(URDF, joint_names=JOINT_NAMES)

    orch = Orchestrator(
        perception=perception,
        generator=generator,
        backend=backend,
        rate=args.rate,
        clamp=clamp,
        realtime=args.gui,
    )
    orch.run(args.seconds)

    print("final pose:", {k: round(v, 3) for k, v in backend.get_joint_states().items()})
    backend.close()


if __name__ == "__main__":
    main()
