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

- **M0** *(this iteration)* Perception reality check. Run BeatNet on your own
  a cappella + humming, score with F/CMLt/AMLt, see where it breaks. Pure
  audio — no robot. See the **M0 — beat-tracking reality check** section
  below for the harness and recording protocol.
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

## M0 — beat-tracking reality check (a cappella + humming)

Goal: run a real singing beat tracker on your own a cappella and humming, see
where it falls over, and have numbers to point to. The tracker is
[BeatNet](https://github.com/mjhydri/BeatNet) (online / particle-filter mode,
NFR-2 causal). The harness is `tools/eval_beat.py`.

**Recording protocol (click-synced ground truth)**

1. Put on earphones. Play a metronome through them at a known BPM.
2. Record (mic + earphone audio routed only to your ears, not into the mic) a
   short take of: a cappella → humming → a cappella, ~30 s each. Save the WAV.
3. The metronome BPM is the ground truth. The harness builds the click grid
   from `--bpm`.

Store WAVs under `data/` (git-ignored).

**Where it runs**

- **Local** (Windows laptop): only the harness's synthetic-WAV smoke tests run
  here. BeatNet itself is heavy on Windows (pulls torch + madmom), so we do not
  install it locally — `tests/test_beat_tracker.py::*_real_beatnet_*` skips.
- **Colab / Kaggle**: BeatNet runs here. Install with
  `!pip install -r requirements-experiments.txt`, then call the same
  `tools.eval_beat` CLI.

**Running**

```bash
# Locally — synthetic smoke (no BeatNet needed):
python -m tools.eval_beat synth --out data/click_120.wav --bpm 120 --seconds 8 --with-vocal

# Anywhere BeatNet is installed (Colab/Kaggle, or after pip install):
python -m tools.eval_beat eval --wav data/singing_120.wav --bpm 120 \
       --beats-per-bar 4 --out data/singing_120.png
python -m tools.eval_beat eval --wav data/humming_120.wav --bpm 120 \
       --beats-per-bar 4 --out data/humming_120.png --json > data/humming_120.json
```

**Results table — fill in once real recordings exist**

| Track | BPM | F | CMLt | AMLt | RT-factor |
|---|---:|---:|---:|---:|---:|
| a-cappella_120bpm.wav | 120 | _ | _ | _ | _ |
| humming_120bpm.wav    | 120 | _ | _ | _ | _ |
| a-cappella_90bpm.wav  |  90 | _ | _ | _ | _ |
| humming_90bpm.wav     |  90 | _ | _ | _ | _ |

Read the metrics as:
- **F** — proximity F-measure (70 ms window). Counts how many detected beats
  land near a click.
- **CMLt** — Correct-Metrical-Level, tempo-locked. Drops to 0 if the tracker is
  on a wrong tempo grid.
- **AMLt** — Allowed-Metrical-Level. Forgiving of tempo halving / doubling. The
  CMLt vs AMLt gap is the "octave error" signal you care about for humming.
- **RT-factor** = process_sec / audio_sec. ≤ 1.0 means realtime-capable.

### Public-data evaluation (no recordings needed; SOTA-comparable)

The harness also accepts an annotation file via `--beats <FILE>` (one beat
time in seconds per line). Combined with vocal-separated audio from a
public, beat-annotated dataset, this lets us run the same evaluation against
SOTA-compatible ground truth before any home recordings exist (spec §10.2).

**Recommended public datasets**

- *Beat-annotated pop / rock* (primary — directly comparable with SOTA):
  - **GTZAN-Rhythm** (Marchand & Peeters 2015) — 1000 tracks, 10 genres
  - **Ballroom** (Gouyon et al. 2006) — 698 tracks, stable tempo
  - **Hainsworth** (Hainsworth & Macleod 2004) — 222 difficult tracks
  - **Isophonics / Beatles** (Mauch et al. 2009) — beats + chords + structure
  - **RWC Popular** (Goto et al. 2002, AIST annotations) — 100 J-Pop tracks
- *Real a cappella* (control — no Demucs needed):
  - **Dagstuhl ChoirSet** (Rosenzweig et al. 2020)
  - **Choral Singing Dataset** (Cuesta et al. 2018)
- *Humming corpora* (auxiliary — melody-only GT; for tempo-only / has-beat
  analysis, not F-measure against beats):
  - **MIR-QBSH** (Jang & Lee 2008), **MTG-QBH** (Salamon et al. 2013)

**Pipeline (Colab/Kaggle — Demucs and BeatNet live there)**

```bash
# (1) Convert the dataset's annotation to our --beats format.
#     Runs anywhere; no heavy deps.
python -m tools.prep_dataset convert-ann --dataset ballroom \
       --input  raw/Ballroom/annotations/Albums-AnaBelen_Veneo-11.beats \
       --output data/beats/AnaBelen_Veneo-11.beats

# (2) Separate vocals with Demucs. Colab/Kaggle only.
#     !pip install -r requirements-experiments.txt
python -m tools.prep_dataset separate \
       --wav     raw/Ballroom/audio/AnaBelen_Veneo-11.wav \
       --out-dir data/vocal

# (3) Evaluate the same way as for click-synced recordings, but with --beats:
python -m tools.eval_beat eval \
       --wav   data/vocal/AnaBelen_Veneo-11.wav \
       --beats data/beats/AnaBelen_Veneo-11.beats \
       --out   data/eval/AnaBelen_Veneo-11.png
```

Locally, only **(1)** runs — the parser is pure Python (covered by
`tests/test_prep_dataset.py`). Steps **(2)** and **(3)** raise a clear
"installed on Colab/Kaggle" error when Demucs / BeatNet are missing.

### Turnkey Colab notebook — GTZAN-Rhythm sweep

`notebooks/m0_gtzan_eval.ipynb` runs the whole pipeline (clone, install,
fetch GTZAN audio + Marchand–Peeters annotations, Demucs vocal separation,
BeatNet evaluation, per-genre table + CSV) on a fresh Colab GPU runtime.

**Open in Colab:**

> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yukijames25/groovebot/blob/master/notebooks/m0_gtzan_eval.ipynb)

Or paste this URL into a browser:
`https://colab.research.google.com/github/yukijames25/groovebot/blob/master/notebooks/m0_gtzan_eval.ipynb`

The notebook:
- evaluates only the **vocal-heavy** GTZAN genres
  (`pop`, `rock`, `hiphop`, `reggae`, `blues`, `country`, `disco`); classical /
  jazz / metal are excluded.
- uses BeatNet's `online` (particle-filter) mode only (NFR-2).
- has a single user knob, `PER_GENRE_LIMIT` (default 5 → 35 tracks total;
  ~20–30 min on a T4).
- prefers the `TempoBeatDownbeat/gtzan_mini` GitHub mirror for audio and
  falls back to the Kaggle GTZAN mirror (requires `kaggle.json`).
- writes `data/m0_work/{beats,vocal,eval}/` + per-track and per-genre CSVs.

The actual loop / aggregation lives in `experiments/run_gtzan_eval.py` and
is covered by `tests/test_run_gtzan_eval.py` (no Demucs / BeatNet / torch
needed locally).

**Editing the notebook locally:** the notebook is generated from
`tools/_build_m0_notebook.py`. Edit that script, then run
`python -m tools._build_m0_notebook` to regenerate
`notebooks/m0_gtzan_eval.ipynb`. Structural and CLI-flag consistency are
checked by `tests/test_notebook.py`.

## Layout

```
robot/groovebot.urdf          the body contract (10 DOF upper body)
groovebot/
  backend.py                  RobotBackend interface + MuJoCo / PyBullet / RealServo
  types.py                    GrooveContext / JointCommand (spec §5.1)
  groove.py                   RuleGrooveGenerator — hand-authored groove (M1)
  orchestrator.py             fixed 30-50 Hz control loop + MetronomePerception stub
  limits.py                   URDF-derived joint limits + clamp helper (NFR-4)
  perception/
    beat_tracker.py           BeatTrackerPerception wrapping BeatNet (M0; spec §5.2)
tools/
  eval_beat.py                M0 evaluation CLI (--bpm click GT or --beats annotation; F/CMLt/AMLt + RT-factor + PNG)
  prep_dataset.py             public-dataset prep: annotation -> .beats; Demucs vocal separation (Colab/Kaggle)
  _build_m0_notebook.py       regenerates notebooks/m0_gtzan_eval.ipynb (source of truth)
experiments/
  run_gtzan_eval.py           Colab-side engine: select/convert/separate/evaluate/aggregate
notebooks/
  m0_gtzan_eval.ipynb         turnkey Colab notebook for the GTZAN sweep
demo_groove.py                end-to-end loop driven by the orchestrator
tests/                        pytest: limits, body-agnostic, orchestrator, eval, tracker
docs/SYSTEM_SPEC.md           the spec (the canonical reference)
train/PIPELINE.md             the B-3 training pipeline (AIST++ → codebook → robot)
requirements.txt              core deps (local: mujoco, mir_eval, soundfile, matplotlib, pytest)
requirements-experiments.txt  BeatNet stack (Colab/Kaggle only)
```
