"""
backend.py — the seam between the "brain" (perception -> groove) and the "body".

The brain only ever talks to RobotBackend. It does NOT import mujoco/pybullet/
ROS. To run on a different body, implement RobotBackend once for that body and
inject it. Nothing else changes.

    brain  ->  RobotBackend (interface)  ->  { MuJoCo | PyBullet | real servos }

Joint targets are always a dict {joint_name: angle_in_radians}, keyed by the
joint names defined in robot/groovebot.urdf.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


# The 10 driven joints, in URDF order. Real servos will map 1:1 to these names.
JOINT_NAMES = [
    "torso_roll", "torso_pitch", "neck_yaw", "neck_pitch",
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow",
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow",
]


@runtime_checkable
class RobotBackend(Protocol):
    def load(self, urdf_path: str) -> None: ...
    def set_joint_targets(self, targets: dict[str, float]) -> None: ...
    def step(self, dt: float) -> None: ...
    def get_joint_states(self) -> dict[str, float]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# MuJoCo backend — kinematic playback (default for visualising choreography).
# No actuators needed: we drive joint angles directly and let MuJoCo compute
# forward kinematics. Great for "see the groove". Swap to actuator/PD control
# later when you care about real dynamics.
# --------------------------------------------------------------------------- #
class MujocoBackend:
    def __init__(self, smoothing: float = 0.35):
        self.smoothing = smoothing  # 0..1 low-pass toward target (servo-like lag)
        self._m = None
        self._d = None
        self._qadr: dict[str, int] = {}
        self._target: dict[str, float] = {}

    def load(self, urdf_path: str) -> None:
        import mujoco
        self._m = mujoco.MjModel.from_xml_path(urdf_path)
        self._d = mujoco.MjData(self._m)
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint '{name}' not found in {urdf_path}")
            self._qadr[name] = int(self._m.jnt_qposadr[jid])
            self._target[name] = 0.0
        mujoco.mj_forward(self._m, self._d)

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        self._target.update(targets)

    def step(self, dt: float) -> None:
        import mujoco
        a = self.smoothing
        for name, adr in self._qadr.items():
            cur = self._d.qpos[adr]
            self._d.qpos[adr] = cur + a * (self._target[name] - cur)
        mujoco.mj_forward(self._m, self._d)

    def get_joint_states(self) -> dict[str, float]:
        return {n: float(self._d.qpos[a]) for n, a in self._qadr.items()}

    def render_frame(self, width: int = 640, height: int = 480):
        """Return an RGB numpy frame (needs a GL backend; optional)."""
        import mujoco
        renderer = mujoco.Renderer(self._m, height=height, width=width)
        renderer.update_scene(self._d, camera=-1)
        return renderer.render()

    def close(self) -> None:
        self._m = self._d = None


# --------------------------------------------------------------------------- #
# PyBullet backend — real position control with dynamics (fixed base).
# Use when you want physically plausible motion / contact / servo PD behaviour.
# --------------------------------------------------------------------------- #
class PyBulletBackend:
    def __init__(self, gui: bool = False):
        self.gui = gui
        self._p = None
        self._uid = None
        self._jidx: dict[str, int] = {}

    def load(self, urdf_path: str) -> None:
        import pybullet as p
        import pybullet_data
        self._p = p
        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self._uid = p.loadURDF(urdf_path, useFixedBase=True)
        for j in range(p.getNumJoints(self._uid)):
            name = p.getJointInfo(self._uid, j)[1].decode()
            if name in JOINT_NAMES:
                self._jidx[name] = j

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        p = self._p
        for name, ang in targets.items():
            if name in self._jidx:
                p.setJointMotorControl2(self._uid, self._jidx[name],
                                        p.POSITION_CONTROL, targetPosition=ang,
                                        force=8.0)

    def step(self, dt: float) -> None:
        self._p.setTimeStep(dt)
        self._p.stepSimulation()

    def get_joint_states(self) -> dict[str, float]:
        return {n: self._p.getJointState(self._uid, j)[0] for n, j in self._jidx.items()}

    def close(self) -> None:
        if self._p is not None:
            self._p.disconnect()


# --------------------------------------------------------------------------- #
# RealServoBackend — fill in after lab assignment.
# Map each JOINT_NAME to a servo channel; convert radians -> servo command.
# Because the brain only uses the RobotBackend interface, NOTHING else changes.
# --------------------------------------------------------------------------- #
class RealServoBackend:
    def __init__(self, port: str):
        self.port = port
        raise NotImplementedError(
            "Implement after building the robot: open serial bus, map "
            "JOINT_NAMES -> servo IDs, send angle commands at the control rate."
        )
