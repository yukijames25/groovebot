# GrooveBot

A cappella / humming → a robot that grooves *with* you, matching your tempo and
tension. Built software-first: everything below runs with **no hardware** and on
**free compute**. After lab assignment, you build the 10-DOF upper body to match
`robot/groovebot.urdf` and swap in the real-servo backend — the brain code does
not change.

## Architecture — brain vs body

```
 voice ─► [beat tracker] ─► beat_pos ┐
         [arousal model] ─► energy   ├─► GrooveController ─► RobotBackend ─► body
         [voice embed]   ─► (M3)     ┘                         (interface)   │
                                                                             ├ MuJoCo  (now, visualise)
                                                                             ├ PyBullet(dynamics)
                                                                             └ RealServo(after build)
```

The brain only ever calls `RobotBackend`. It never imports a simulator. To run
on any body — MuJoCo, PyBullet, Isaac (via a thin adapter), or real servos —
implement the interface once and inject it. `robot/groovebot.urdf` is the single
source of truth ("the contract") that both the sim and the physical robot obey.

## Roadmap

- **M0** Perception reality check. Run an open-source *singing* beat tracker
  (mjhydri / SingNet) on your own a cappella + humming. Find where it breaks.
  Pure audio — no robot.
- **M1** *(this scaffold)* End-to-end loop with a metronome + hand-authored
  groove. Proves the whole pipeline runs. `python demo_groove.py`.
- **M2** *(required)* Replace the constant `energy` with arousal estimated from
  the singer's voice (energy envelope + pitch). Replace the metronome with the
  live beat tracker from M0.
- **M3** *(goal)* Replace `GrooveController` with a trained generative model: a
  VQ-VAE groove codebook, sequenced by a transformer conditioned on
  beat-phase + arousal + voice embedding, trained on vocal-separated AIST++.
  See `train/PIPELINE.md`.

## The three plug-in seams

All three live behind stable interfaces (see `docs/SYSTEM_SPEC.md` §5.2), so
you can land them one at a time:
- beat / arousal source → `Perception` protocol (`groovebot/orchestrator.py`).
  M1 ships `MetronomePerception` (stub); M2 replaces it with a live
  `BeatTracker` + `ArousalEstimator`.
- groove model → `GrooveGenerator` protocol (`groovebot/groove.py`). M1 ships
  `RuleGrooveGenerator`; M3 swaps in the trained model. The orchestrator and
  demo do not change.

## Free, serious stack (what the SOTA papers actually use)

- **PyTorch** — modelling
- **AIST++** — paired music↔3D-dance data (you only need SMPL motion + audio,
  not the huge multi-view video, so the footprint is small)
- **Demucs** — separate vocals from full tracks → the vocal inherits the song's
  beat labels (the trick that makes a cappella beat data possible)
- **WavLM / HuBERT** (HuggingFace) — self-supervised voice front-end
- **Kaggle Notebooks** — ~30 free GPU h/week (T4/P100 16 GB); data stays in the
  cloud, so your PC's disk/VRAM is irrelevant for training. Lab GPUs later for
  full-scale runs.

Note: the M3 core is **supervised generative modelling, not reinforcement
learning** — so no heavy RL sim or GPU farm is required.

## Run (no hardware needed)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, allow it for this user once:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

pip install -r requirements.txt
# Optional: dynamics + a window (PyBullet is not in requirements.txt by default)
# pip install pybullet

python demo_groove.py --backend mujoco --bpm 120 --energy 0.85 --seconds 8
python demo_groove.py --backend pybullet --gui      # only after pip install pybullet
pytest
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo_groove.py --backend mujoco --bpm 120 --energy 0.85 --seconds 8
pytest
```

## Layout

```
robot/groovebot.urdf      the body contract (10 DOF upper body)
groovebot/
  backend.py              RobotBackend interface + MuJoCo / PyBullet / RealServo
  types.py                GrooveContext / JointCommand (spec §5.1)
  groove.py               RuleGrooveGenerator — hand-authored groove (M1)
  orchestrator.py         fixed 30-50 Hz control loop + MetronomePerception stub
  limits.py               URDF-derived joint limits + clamp helper (NFR-4)
demo_groove.py            end-to-end loop driven by the orchestrator
tests/                    pytest: limit property + body-agnostic smoke + loop
docs/SYSTEM_SPEC.md       the spec (the canonical reference)
train/PIPELINE.md         the B-3 training pipeline (AIST++ → codebook → robot)
```
