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

- **~~M0~~** *(shelved — see `docs/SYSTEM_SPEC.md` §14.3)* Blind online beat
  tracking on a cappella + humming (BeatNet). Kept as fallback code only;
  not the primary perception path anymore.
- **M0'** *(this iteration)* Reference-alignment feasibility — pick a known
  song, time-stretch its audio (synth_warp), run offline DTW, score how
  well the reference beat grid is recovered. Local, CPU, librosa-only.
  See the **M0' — alignment feasibility check** section below.
- **M1** *(done)* End-to-end loop with a metronome + hand-authored groove.
  `python demo_groove.py`.
- **M2** *(required)* Wire an online `ReferenceAligner` (online DTW /
  score following) into the orchestrator's perception thread; add arousal
  estimation + screen/face feedback.
- **M3** *(goal)* Trained generative groove: a VQ-VAE groove codebook
  sequenced by a transformer conditioned on beat-phase + downbeat +
  **song structure** + arousal + voice embedding, trained on
  vocal-separated AIST++. See `train/PIPELINE.md`.

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

## M0' — alignment feasibility check (local, CPU, librosa-only)

> Primary perception path per the post-pivot spec (§9.x and §14). See the
> shelved M0 section below for the older blind-tracking flow we kept around
> as a fallback.

Goal: prove that **offline DTW alignment can recover a reference song's beat
grid from a warped query**. We synthesise the query by time-stretching the
reference at known rates, so we have ground truth without recording anything.

**What runs:**

- `groovebot/align/features.py` — chroma or pyin-pitch features (12 × T).
- `groovebot/align/dtw_align.py` — `OfflineDTWAligner` (librosa DTW) + a
  `map_reference_beats` helper that pulls reference beats through the warp
  path onto the query timeline.
- `tools/synth_warp.py` — applies time-stretch rates to a wav and outputs
  the warped audio + warped beat annotations.
- `experiments/run_m0p_align.py` — runs the full sweep and writes per-track
  / per-rate / overall CSVs plus per-track overlay PNGs.

Scoring reuses the same `mir_eval` harness as the blind path
(`tools/eval_beat.py::score_beats`), so M0' numbers and M0 numbers (when
revived) sit in the same table.

**Data prep**

1. Pick a small set of references (≥10 s each so `mir_eval.beat.trim_beats`
   leaves a meaningful number of beats). The repository's GTZAN-Rhythm
   convenience scripts (`tools/prep_dataset.py`, `experiments/run_gtzan_eval.py`)
   already discover and pair audio + `.beats` annotations.
2. For full-mix audio it sometimes helps DTW to operate on a vocal stem.
   Run Demucs once (Colab/Kaggle, since we don't install it locally) and
   drop the resulting `vocals.wav` next to its `.beats`. For an initial
   smoke pass, plain full-mix audio also works.
3. Put everything under `data/m0p_refs/` (git-ignored) as
   `<stem>.wav` + `<stem>.beats` neighbours.

**Install + run (local, Windows / mac / Linux)**

```powershell
# Local profile only needs librosa (no torch, no madmom, no Demucs).
pip install librosa

python -m experiments.run_m0p_align `
       --root    data/m0p_refs `
       --out-dir data/m0p_work `
       --feature chroma `
       --rates 0.9 0.95 1.0 1.05 1.1
```

Outputs in `data/m0p_work/`:

```
m0p_per_track.csv     one row per (track, rate): F / CMLt / AMLt / RT-factor
m0p_per_rate.csv      means per stretch rate
m0p_overall.csv       overall means
<stem>_r<rate>.png    warp path + query waveform with GT vs. recovered beats
```

Read the metrics the same way as the M0 reality check: F (70 ms window),
CMLt (tempo-locked), AMLt (tempo-doubling forgiven), RT-factor
(process_sec / audio_sec; offline DTW will not be ≤ 1.0, that's a tracker
property — online alignment lands in M2).

**Caveat (Tier 1)**: because the query comes from time-stretching the
reference itself, this checks the alignment *mechanism* and tempo
robustness only — not the harder cross-performer case. Tier 2 (below)
is the real test.

## M0' Tier 2 — real renditions (singing & humming)

Tier 2 swaps the self-warped query for **a different performance** of the
same song — typically you (the user) singing or humming along while the
original plays through earphones. That gives real timbre / micro-timing
variation, and exposes humming's worst case (no harmonic stack, so chroma
is noisy and pyin pitch contour has to carry the alignment).

**Input layout** (one directory per song):

```
data/m0p_t2/<song>/
  original.wav            full-mix reference (Demucs vocal stem is derived here)
  original.beats          reference beat times (one per line, seconds)
  rendition_sing*.wav     (any number) singing renditions  -> chroma DTW
  rendition_hum*.wav      (any number) humming renditions  -> pyin-pitch DTW
  rendition_*.wav         (any number) generic; defaults to chroma
  gt.beats                ground-truth beat times for the renditions
```

The filename is what routes to the right feature: `hum` (or `humming`)
anywhere in the basename picks the pyin melody path; anything else uses
chroma. `gt.beats` is shared by all renditions in the dir — when you
record along to the original, GT == `original.beats` and you can just
copy it.

**Data prep (recommended, copyright-clean)**

1. Pick 2-3 songs you have legal access to. Drop the full-mix `original.wav`
   under `data/m0p_t2/<song>/` (gitignored — never commit audio).
2. Drop `original.beats`. For GTZAN-Rhythm you already have the annotations;
   for anything else use the metronome / hand-tap / `tools/prep_dataset.py`.
3. Put on earphones, play the original, record yourself singing / humming
   on a separate channel. Save as `rendition_sing*.wav` and / or
   `rendition_hum*.wav`. Because you sing *to* the original, the original's
   beat grid is the rendition's ground truth — copy `original.beats` to
   `gt.beats`.
4. Alternative: licensed karaoke / cover-research datasets where the GT
   beats come from the dataset.

> Original recordings stay under `data/` (already in `.gitignore`). Do not
> commit, push, or upload them to public services — they may be cached
> even after deletion.

**Install + run**

```powershell
# Local profile (librosa). Demucs is on the experiments profile and runs
# in Colab/Kaggle; the runner lazy-imports it via tools.prep_dataset and
# skips songs whose build_reference() fails.
pip install librosa

python -m experiments.run_m0p_t2 `
       --root    data/m0p_t2 `
       --out-dir data/m0p_t2_work
```

Outputs in `data/m0p_t2_work/`:

```
m0p_t2_per_rendition.csv  one row per rendition: F / CMLt / AMLt / RT-factor
m0p_t2_per_kind.csv       means by feature kind (chroma vs pitch)
m0p_t2_per_song.csv       means by song
m0p_t2_overall.csv        overall means
<song>_<rendition>.png    query waveform + GT vs recovered beats + warp path
```

Scoring is the **same `tools/eval_beat.py::score_beats` mir_eval harness**
as Tier 1 and the shelved blind path, so Tier 1 / Tier 2 / blind numbers
all sit in the same table.

## M0' Tier 2 — DAMP route (no recording, real amateurs at scale)

A no-recording variant of Tier 2 that scales to thousands of independent
amateur performances using the **DAMP-VSEP** or **DAMP-S-AG** corpora
(Smule Research Data License). The datasets ship pre-separated stems
(vocal / backing / mixture), so **Demucs is not needed** here. See spec
§9.x DAMP for the full rationale and limits.

For each arrangement (the same song sung by many singers, all timed to a
shared backing track):

- **beats**         librosa.beat.beat_track on the backing — reliable
                    because the backing is instrumental and offline (this
                    is not the shelved blind vocal beat path).
- **chroma ref**    librosa chroma_cqt on the backing.
- **melody ref**    either a single **designated** rendition's pyin F0,
                    or a **consensus** built from a frame-wise nanmedian
                    of every rendition's F0, leave-one-out per query.

Every query rendition is scored on **both paths** with the same scorer:

- **chroma path**   query vocal chroma -> backing chroma.
- **pitch path**    query F0 chroma -> melody reference.

**License + data handling**

DAMP datasets require a request to Smule and are released under their
Research Data License: **non-commercial, no redistribution**. Treat them
exactly like the rest of `data/`:

- Local-only. Already covered by `.gitignore` (whole `data/` tree).
- **Never commit, push, or upload** DAMP audio anywhere. Public caches
  may keep copies after deletion.
- Don't redistribute derived artifacts (separated stems, F0 contours,
  even spectrograms) outside the project's local working tree.

**Normalized input layout** (one arrangement per subdirectory):

```
data/m0p_t2_damp/<arrangement_id>/
  backing.wav
  vocal_<rendition_id>.wav     (one or more)
```

For DAMP-VSEP, treat each (or each filtered) segment as an arrangement
with one rendition. For DAMP-S-AG, every singer-take becomes a rendition
of the same arrangement (`amazing_grace`). The raw DAMP file naming
varies between releases; if yours doesn't match, write a one-off script
to copy or symlink into the layout above. `tools/ingest_damp.py list
--root data/m0p_t2_damp` prints a quick sanity listing.

**Run**

```powershell
pip install librosa

# Designated melody reference (first rendition by id).
python -m experiments.run_m0p_t2_damp `
       --root    data/m0p_t2_damp `
       --out-dir data/m0p_t2_damp_work

# Consensus melody (leave-one-out median across renditions).
python -m experiments.run_m0p_t2_damp `
       --root    data/m0p_t2_damp `
       --out-dir data/m0p_t2_damp_work `
       --melody-mode consensus
```

### DAMP-S-AG variant — MIDI reference (no backing audio / ffmpeg)

DAMP-S-AG (Sing! Amazing Grace, 17,582 renditions of one song) ships an
`amazing_grace.midi` next to the audio. We use that MIDI directly as the
reference, sidestepping the need to decode the M4A backing track —
`pretty_midi` gives us beats, a one-hot pitch chroma melody, and a
column-L2 chroma template, all on the same `(12, T)` shape as our chroma
features. ffmpeg is not needed; renditions are libsndfile-readable
(`.m4a` extension, OGG/VORBIS payload).

**Data prep**

`tools/ingest_damp damp-s-ag` stream-extracts a subset directly from the
raw `amazing_grace.tar.gz` into the normalized arrangement layout. The
source tarball is **not modified**.

```powershell
pip install librosa pretty_midi

# Extract the first 100 renditions (in archive order), plus the MIDI.
python -m tools.ingest_damp damp-s-ag `
       --tarball data/amazing_grace.tar.gz `
       --out     data/m0p_t2_damp `
       --max-n   100

# Optional filters before --max-n is applied:
#   --headphones-only        (TSV column `headphones == 1`)
#   --country US             (TSV column `country`)
```

Writes:

```
data/m0p_t2_damp/amazing_grace/
  reference.midi              (from amazing_grace.midi in tar)
  vocal_<perf_id>.m4a × N     (OGG/VORBIS payload; readable by soundfile)
```

**Run**

```powershell
python -m experiments.run_m0p_t2_damp `
       --root             data/m0p_t2_damp `
       --out-dir          data/m0p_t2_damp_work `
       --reference-source midi `
       --vocal-glob       "vocal_*.m4a"
```

The pitch path (query F0 vs MIDI melody) is the intended primary read
for humming behaviour; the chroma path (query chroma vs MIDI chroma
template) is reported alongside for comparison. MIDI mode scores every
rendition (no designated / consensus selection).

**Specific limits for DAMP-S-AG**

- Only one song. DAMP-S-AG is a single-arrangement deep dive; for
  arrangement coverage you still want DAMP-VSEP.
- *Amazing Grace* is a **slow, rubato-leaning hymn**. Renditions vary in
  tempo and phrasing; the reference grid still comes from the same
  shared backing track all singers heard, so the warp DTW has to absorb
  is genuine cross-performer variation.
- All other DAMP-S-AG limits from the section above (timing fixed to
  backing, singing as a humming proxy) still apply.

Outputs in `data/m0p_t2_damp_work/`:

```
m0p_t2_damp_per_path.csv          one row per (rendition, path): F / CMLt / AMLt / RT
m0p_t2_damp_per_kind.csv          means by feature kind (chroma vs pitch)
m0p_t2_damp_per_arrangement.csv   means by arrangement_id
m0p_t2_damp_overall.csv           overall means
<arr>__<rendition>__<kind>.png    query vocal + GT vs recovered beats + warp path
```

**Limits to read the numbers with** (also in spec §9.x DAMP):

- DAMP renditions are sung *to* the backing, so timing is largely fixed
  — easier than truly free-tempo independent performances. This matches
  our target use case ("pick a song, sing/hum along") but means the
  numbers will be optimistic for a free-tempo regime.
- The pitch path uses **real singing as a humming proxy**. It exercises
  the F0-only DTW path on real performer F0 (not self-derived), but may
  be optimistic for true humming where pitch is often less stable. A
  separate confirmation pass on a QBH corpus (e.g. MIR-QBSH) with a
  melody-alignment metric is optional follow-up.

## ~~M0~~ — beat-tracking reality check *(shelved fallback, blind path)*

> Shelved per `docs/SYSTEM_SPEC.md` §14.3 (madmom 0.16.1 doesn't build on
> Colab's Python 3.12 + numpy 2.x stack). The code stays as the fallback
> path for unknown songs / improvised humming; we re-enable the numbers
> once the dependency situation untangles.

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

## GrooveStyleSelector — startup style picker (v1, parallel track)

The alignment / beat work above decides **when** the robot moves. The
`groovebot.style` package decides **how** it moves — what kind of nori
(headbang vs sway vs penlight wave, big vs small) given the song's
character. This is a deliberately separate vertical slice from the
timing track: style decisions happen once at the startup window
(≈5–10 s of audio at the song's start), so DAMP availability and online
alignment progress do not block it.

See spec `docs/SYSTEM_SPEC.md` §14 module note for the contract.

### Design

```
audio (5-10 s startup window)
   ├── features.log_mel_spectrogram ─► StyleCNN ─► (genre_probs, mood_probs)
   ├── attributes.estimate_tempo    ─► tempo BPM
   └── attributes.estimate_arousal  ─► arousal 0..1 ─► arousal_bucket
                                                            │
                              table.select_move(genre, arousal_bucket, mood_probs)
                                                            │
                                                            ▼
                                                      GrooveStyle
                                                  (move + intensity text)
```

- **features.py** — log-mel spectrogram (librosa, n_mels=64, hop=512,
  resamples to 22050 Hz). Variable time dim; model handles via
  adaptive average pool.
- **model.py** — small 4-block CNN (channels 16→32→64→128, BN+ReLU+
  MaxPool each), `AdaptiveAvgPool2d(1)`, then a `ModuleDict` of heads.
  v1 ships `genre` (10-class, GTZAN vocab) and `mood` (6-class). Tempo
  and arousal are computed by heuristic for now; both can be promoted
  to learned heads later without changing the call sites in
  `select.py`.
- **attributes.py** — `estimate_tempo` (librosa.beat.beat_track) and
  `estimate_arousal` (geometric mean of normalised RMS loudness and
  onset density; the geometric form self-gates near-silent inputs
  whose RMS is tiny but whose onset peak picker fires on noise).
- **table.py** — Yuki's nori lookup: `(genre, arousal_bucket, mood) →
  (move, intensity)`. Mood enters as a soft probability distribution,
  not argmax: each mood contributes a preferred-move distribution
  weighted by its probability, summed, then multiplied by the genre×
  arousal bias. Argmax over the combined bias picks the move.
- **select.py** — `GrooveStyleSelector.select(audio, sr)` returns a
  `GrooveStyle` dataclass with `move`, `intensity`, `tempo_bpm`,
  `arousal`, and the full softmax probability dicts.

### Upper-body-feasible move vocabulary (10 DOF, no legs)

`headbang`, `bob_nod`, `sway`, `rock`, `fist_pump`, `clap`,
`penlight_wave`, `quiet_listen`. Move semantics are documented in
`groovebot/style/table.py`. v1 outputs text labels only; mapping these
to concrete `JointCommand` trajectories belongs to a later
`GrooveGenerator` revision once we know the labels are stable.

### Data

- **Genre** (real): GTZAN, 10 classes. `data/raw/gtzan_mini/genres/
  <genre>/<file>.wav` is enough to smoke-train the pipeline; the full
  GTZAN (~1 k files) is required for any meaningful accuracy.
- **Mood** (stub): no CC mood-tagged audio is wired up yet.
  `experiments/train_style.py` uses a deterministic
  genre → mood pseudo-map (`_STUB_MOOD`) so the multi-head training
  loop runs end-to-end. The mood val accuracy this produces is
  meaningless; it measures whether the network learned the
  deterministic map.
- **TODO**: replace `_STUB_MOOD` with a loader over the **MTG-Jamendo
  mood subset** (~14 k CC clips, autotag mood labels). Spec §14 module
  note also mentions FMA mood as an option.
- **Hard rule**: no copyrighted J-pop audio or video on disk or in
  commits. The whole style pipeline only ever ingests CC / research-
  licensed material (data sits under `data/`, which is gitignored).

### Run

Install (one extra; everything else is already on the M0' profile):

```bash
pip install torch  # CPU is fine for v1; small CNN trains in minutes
```

Smoke-train on `gtzan_mini`:

```bash
python -m experiments.train_style \
    --gtzan-root data/raw/gtzan_mini/genres \
    --out-dir data/style_smoke \
    --epochs 20 --batch-size 8
```

Outputs:
- `data/style_smoke/style_cnn.pt` — checkpoint
- `data/style_smoke/report.json` — train/val history, representative
  `GrooveStyle` labels per genre, tempo estimates

Inference from Python:

```python
import soundfile as sf
from groovebot.style import GrooveStyleSelector

selector = GrooveStyleSelector()                     # random weights
# selector.model.load_state_dict(torch.load("data/style_smoke/style_cnn.pt")["state_dict"])
audio, sr = sf.read("some_song.wav", dtype="float32", always_2d=False)
style = selector.select(audio[: 10 * sr], sr)
print(style.as_text())
# -> "headbang@0.95 (metal/aggressive, 145BPM, arousal=0.81/high)"
```

### Validation

`tests/test_style_*.py` cover (CPU-only, no real audio dataset):

- `features.py` shape, dtype, resampling, multichannel-to-mono
- `model.py` forward shapes per head, softmax sum, time-dim invariance
- `table.py` per-mood/genre/arousal expected moves, soft weighting
  (blend ≠ argmax), intensity in [0, 1], unknown-key handling
- `select.py` end-to-end on synthetic click tracks: tempo close to
  ground-truth BPM (with octave-error tolerance), arousal contrast
  between dense and near-silent inputs, `GrooveStyle` contract.

`pytest -q tests/test_style_*.py` — all green on CPU, no GPU required.

### Limits (v1, honest)

- **Mood head is not real.** Trained on a deterministic
  genre → mood map. Until MTG-Jamendo (or equivalent) is wired in, the
  mood softmax is just a one-hot regression on the genre prediction.
- **gtzan_mini is 10 files/genre.** Accuracy bounded by data, not
  model. Full GTZAN brings genre into the published-baseline range
  (~70–80%); v1's smoke run on the mini set lands around 30–40%.
- **Arousal heuristic is signal-level**, not perceptual. It cannot
  distinguish "energetic mellow ballad" from "loud noise"; for that
  we eventually want a learned head trained on AVEC-style arousal
  labels.
- **Tempo via `librosa.beat.beat_track`** is single-scalar and prone
  to octave errors (classical clip at 234 BPM is the smoke run
  catching this). The table keys on arousal bucket, not BPM, so this
  rarely changes the move selection; but downstream code that reads
  `style.tempo_bpm` should expect ±octave.
- **No JointCommand bridge yet.** Outputs are text labels; the
  `GrooveGenerator` still uses the M1 rule-based map. The two will
  meet once labels are stable.

### v2 — genre head, real numbers (full GTZAN, two splits)

v2 trains the genre head on the full GTZAN audio (1000 tracks) and
reports both a fault-filtered + artist-aware split (honest) and a naive
stratified split (optimistic baseline). The gap between the two is the
leakage bias the GTZAN faults induce. The mood head is left in the
network but `--mood-weight 0.0` (the new default) zeros its loss, so the
backbone is not pulled toward the deterministic genre→mood STUB during
v2 training.

**Data**

- Audio: Kaggle `andradaolteanu/gtzan-dataset-music-genre-classification`,
  unpacked to `data/raw/gtzan_full/Data/genres_original/`.
- Fault-filtered splits: `train_filtered.txt` / `valid_filtered.txt` /
  `test_filtered.txt` from
  [jongpillee/music_dataset_split](https://github.com/jongpillee/music_dataset_split)
  (`GTZAN_split/`), following Kereliuk 2015 / Sturm 2013. 443 / 197 / 290 =
  930 tracks; 70 removed as duplicates / mislabels / distortions; the
  split keeps artists disjoint across train/val/test as far as the
  known-artist coverage allows (GTZAN artist labels are partial —
  artist-non-overlap is best-effort).
- Probe drops: `sf.info` is called on every file at discovery. On the
  Kaggle mirror, `jazz/jazz.00054.wav` consistently fails (the famous
  Sturm-documented broken file); it is logged to
  `report.json["skipped"]`.

**Overfitting controls (small data, must-have)**

- random time crop (`--random-crop`): per-clip random window of the
  log-mel along the time axis, preventing position memorisation.
- SpecAugment (`--specaugment`, Park et al. 2019): random frequency
  and time masks zero out small bands of the mel.
- head dropout (`--dropout 0.3`): default in v2.
- val-acc early stopping (`--early-stopping-patience 8`): stop if val
  accuracy does not improve for N consecutive epochs.

**Run (two passes)**

```bash
# A — honest numbers
python -m experiments.train_style \
    --gtzan-root data/raw/gtzan_full/Data/genres_original \
    --splits-dir data/raw/gtzan_splits \
    --split-mode fault \
    --out-dir data/style_full_fault \
    --epochs 40 --batch-size 16 --dropout 0.3 \
    --specaugment --random-crop \
    --early-stopping-patience 8

# B — leakage baseline
python -m experiments.train_style \
    --gtzan-root data/raw/gtzan_full/Data/genres_original \
    --split-mode naive \
    --out-dir data/style_full_naive \
    --epochs 40 --batch-size 16 --dropout 0.3 \
    --specaugment --random-crop \
    --early-stopping-patience 8
```

Each pass writes:
- `style_cnn.pt` — best-val checkpoint
- `report.json` — full per-epoch history, best val acc + epoch, held-out
  test acc, train-val gap at the best epoch, confusion matrix (counts +
  row-normalised), representative `GrooveStyle` per test clip, source
  attribution for the split, skipped-file log

**Results — see `data/style_full_*/report.json`** (CPU runs, single
laptop; values reproducible with `--seed 0`, 40 max epochs, patience 8):

| split  | best val genre acc | test genre acc | train-val gap @ best |
|--------|--------------------|----------------|----------------------|
| fault  | 0.498 @ ep 12      | 0.431          | +0.222               |
| naive  | 0.620 @ ep 28      | 0.573          | +0.241               |

**Naive minus fault**: +12.2 pp val, +14.2 pp test. That delta is the
leakage bias caused by GTZAN's known duplicates / repeated artists. The
honest number for this model is the fault row; the naive row sets an
upper bound on how much "ladder we get for free" by ignoring the
faults.

Naive-split confusion is much more diagonal than fault — most rows
recognise their own class — because the test set contains tracks from
artists the model already saw in train.

Fault-split confusion (row-normalised, test set; rows = true, columns =
predicted, top 4 chars):

```
       blue  clas  coun  disc  hiph  jazz  meta   pop  regg  rock
 blue  0.03  0.03  0.03  0.00  0.00  0.52  0.00  0.00  0.00  0.39
 clas  0.03  0.97  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
 coun  0.00  0.10  0.17  0.00  0.00  0.00  0.00  0.43  0.00  0.30
 disc  0.00  0.00  0.07  0.03  0.14  0.00  0.03  0.07  0.31  0.34
 hiph  0.00  0.00  0.00  0.00  0.26  0.00  0.00  0.67  0.07  0.00
 jazz  0.04  0.07  0.04  0.00  0.00  0.22  0.00  0.63  0.00  0.00
 meta  0.00  0.00  0.00  0.00  0.00  0.00  0.78  0.04  0.04  0.15
  pop  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00  0.00
 regg  0.00  0.04  0.00  0.00  0.00  0.04  0.00  0.50  0.42  0.00
 rock  0.03  0.03  0.03  0.03  0.00  0.00  0.03  0.34  0.09  0.41
```

Notes from the confusion: classical (0.97) and pop (1.00) are extremes
on opposite axes — classical is genuinely well separated, pop is a
*sink* class that collects mis-confident predictions from
country/hiphop/jazz/reggae/rock. The model is too eager to predict pop.
Likely contributors: pop sits in the middle of the mel-statistics
distribution (mid tempo, mid spectral centroid), and v2's single-crop
+ constant-LR training doesn't push the head out of that local optimum.
A weighted cross-entropy or multi-crop test-time averaging would
plausibly help; both are deferred.

### Limits (v2, honest)

- **GTZAN published faults make every number slightly optimistic**, even
  under the fault-filtered split. Sturm 2013 documents repetitions,
  mislabels, and distortions; the jongpillee filter removes the worst
  but artist coverage is partial.
- **Mood head is still STUB.** v2 zeros its loss; its accuracy in the
  reports is uninformative. MTG-Jamendo / FMA mood wiring is the
  v3 task.
- **Single-window inference.** v2 trains on a single crop per epoch
  (random in train, center in val/test). Multi-crop averaging at
  inference would likely lift test acc 2–5 pp; deferred.
- **No fine-grained learning rate schedule.** v2 uses constant Adam
  `lr=1e-3`. A warmup + cosine schedule is the obvious next lever.

### v3 — transfer learning (frozen PANNs CNN14) + real mood head

v3 swaps the v2 small-from-scratch StyleCNN backbone for a **frozen,
pretrained PANNs CNN14** (Kong et al. 2020, trained on AudioSet) and
puts an MLP `StyleHead` on top of its 2048-d clip embedding. The
public selector signature does not change — `select(audio, sr) ->
GrooveStyle` still produces the same dataclass with the same fields.

Mood gets a real head too: `tools/ingest_mtg_moodtheme.py` joins the
MTG-Jamendo `autotagging_moodtheme` corpus to a 38-tag → 6-class
mapping (`groovebot/style/mood_mapping.py`), and
`experiments/train_mood_tl.py` trains the mood head with an
artist-non-overlap split. The v1/v2 deterministic genre→mood STUB is
retired.

**New deps (CPU, light beyond the v2 stack)**

- `panns-inference` (PyTorch) — pretrained CNN14 wrapper. Install:
  `pip install panns-inference`.
- `Cnn14_mAP=0.431.pth` (~340 MB) — download once to
  `data/raw/Cnn14_mAP=0.431.pth`:

  ```bash
  curl -fSL -o "data/raw/Cnn14_mAP=0.431.pth" \
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
  ```

  Windows note: `panns_inference` itself tries to fetch a small
  AudioSet labels CSV via `wget` at import time. Pre-place it:

  ```bash
  mkdir -p "$HOME/panns_data"
  curl -fsSL "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv" \
    -o "$HOME/panns_data/class_labels_indices.csv"
  ```

  Both files are gitignored under `data/raw/`.

**Selector API (unchanged contract, new constructor path)**

```python
from groovebot.style import GrooveStyleSelector

# v3 — frozen PANNs backbone + trained head
sel = GrooveStyleSelector.from_panns(
    "data/raw/Cnn14_mAP=0.431.pth",
    head_weights="data/style_v3_fault/style_head.pt",
)
# or equivalently:
# sel = GrooveStyleSelector(backbone=PannsBackbone(ckpt), head=StyleHead())

style = sel.select(audio, sr)         # → GrooveStyle, same dataclass
print(style.as_text())                # → "headbang@0.95 (metal/aggressive, ...)"
```

**v3 genre — fault-filtered split, GTZAN 930 tracks**

```bash
python -m experiments.train_genre_tl \
    --gtzan-root data/raw/gtzan_full/Data/genres_original \
    --splits-dir data/raw/gtzan_splits \
    --panns-ckpt "data/raw/Cnn14_mAP=0.431.pth" \
    --cache-dir data/style_emb_gtzan \
    --out-dir data/style_v3_fault \
    --split-mode fault \
    --epochs 40 --batch-size 32 --dropout 0.3 \
    --window-sec 10.0 --early-stopping-patience 10
```

Precompute takes ~5 min on CPU (PANNs forward is ~0.3 s per 10 s clip
after warm-up). Head training is sub-second per epoch on cached `.npy`
embeddings. Cached embeddings stay valid across re-runs.

**Genre results — v2 vs v3 (fault split, 930 tracks, same artist-non-overlap)**

| version | best val | test    | train-val gap @ best |
|---------|---------:|--------:|---------------------:|
| v2      | 0.498    | 0.431   | +0.222               |
| v3      | **0.761**| **0.817** | **+0.026**         |

v3 lifts test accuracy by **+38.6 pp** while shrinking the train-val
gap from +22 pp to +3 pp — the regularisation drama goes away because
the head is tiny and the embedding is frozen.

v3 fault-split confusion (row-normalised, test = 290 clips):

```
       blue  clas  coun  disc  hiph  jazz  meta   pop  regg  rock
 blue  0.71  0.00  0.07  0.00  0.00  0.03  0.00  0.00  0.10  0.10
 clas  0.00  1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
 coun  0.03  0.00  0.73  0.07  0.00  0.00  0.00  0.03  0.00  0.13
 disc  0.00  0.00  0.00  0.90  0.00  0.00  0.00  0.03  0.03  0.03
 hiph  0.00  0.00  0.04  0.04  0.85  0.00  0.00  0.04  0.04  0.00
 jazz  0.00  0.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00  0.00
 meta  0.00  0.00  0.00  0.04  0.00  0.00  0.78  0.00  0.00  0.18
  pop  0.00  0.00  0.00  0.17  0.03  0.00  0.00  0.77  0.00  0.03
 regg  0.00  0.00  0.08  0.08  0.12  0.00  0.00  0.00  0.73  0.00
 rock  0.03  0.00  0.03  0.03  0.03  0.00  0.09  0.03  0.03  0.72
```

The v2 "pop sink" disappears: pop is now correctly classified 77 % of
the time and other classes no longer collapse into it. Classical and
jazz reach 100 % — both are easy targets at 10 s because they have
strong distributional fingerprints AudioSet's pretraining captures.
Rock confuses with country (3 %), pop (3 %), reggae (3 %), metal (9 %),
which is roughly the human ear's experience.

**v3 mood — MTG-Jamendo `autotagging_moodtheme`**

1. Get the audio. The dataset is CC, but redistribution from us would be
   impolite — fetch it yourself. MTG's `download.py` has no
   archive-range flag (it always pulls all 100 tars), so for a bounded
   subset we curl individual tars by name. Each `audio-low` tar is
   ~517 MB / ~200 moodtheme clips. **6 archives ≈ 3.1 GB ≈ ~780 mood-
   kept clips after the tag mapping. 12 archives ≈ 6.2 GB. Stop before
   that — never pull all 100 (~52 GB).**

   ```bash
   mkdir -p data/raw/mtg_moodtheme
   cd data/raw/mtg_moodtheme

   # 1a. Tag TSV (~3 MB; needed for the manifest, independent of audio).
   curl -fSL -o autotagging_moodtheme.tsv \
       "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_moodtheme.tsv"

   # 1b. Bounded audio pull. Each $n is an archive 00..99.
   for n in 00 01 02 03 04 05; do
     fn="autotagging_moodtheme_audio-low-${n}.tar"
     curl -fSL --retry 3 -o "${fn}" \
       "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/autotagging_moodtheme/audio-low/${fn}"
     # SHA-256 checksums live in data/download/autotagging_moodtheme_audio-low_sha256_tars.txt
     # in the MTG repo — verify if you care.
     tar -xf "${fn}" && rm "${fn}"
   done
   ```

   The unpacked layout is `data/raw/mtg_moodtheme/<NN>/<track_id>.mp3`,
   matching the TSV's `PATH` column.

2. Build the manifest:

   ```bash
   python -m tools.ingest_mtg_moodtheme \
       --audio-root data/raw/mtg_moodtheme \
       --tsv data/raw/mtg_moodtheme/autotagging_moodtheme.tsv \
       --out-csv data/mtg_moodtheme_manifest.csv \
       --conflict-rule drop_on_disagreement
   ```

3. Train:

   ```bash
   python -m experiments.train_mood_tl \
       --manifest data/mtg_moodtheme_manifest.csv \
       --audio-root data/raw/mtg_moodtheme \
       --panns-ckpt "data/raw/Cnn14_mAP=0.431.pth" \
       --cache-dir data/style_emb_mtg \
       --out-dir data/style_v3_mood \
       --epochs 40 --batch-size 64 --dropout 0.3
   ```

A `--synthetic-stub` mode (class-conditional Gaussian embeddings,
`report.json["is_stub"] = true`) is retained so the training loop can
go end-to-end without the MTG download — useful for verifying the
loss / split / report machinery in CI.

**v3 mood — real numbers (6 archives, 780 mood-kept clips)**

The 6-archive slice (TSV dirs `00`-`05`) yields **780 kept clips
across all 6 classes** after the tag mapping drops theme-only and
ambiguous clips. Class balance:

| class      | clips | share |
|------------|------:|------:|
| aggressive |  105  |  13 % |
| happy      |  128  |  16 % |
| sad        |  125  |  16 % |
| calm       |  200  |  26 % |
| dark       |  130  |  17 % |
| epic       |   92  |  12 % |

Min/max ratio = 92 / 200 = 2.2× — moderately imbalanced; `calm` is the
largest class.

**Conflict rule comparison — the rule is moot for this corpus**

Both `drop_on_disagreement` and `first_match` manifests are
**byte-identical** (verified by md5). Running an audit on the full
18 486-clip moodtheme TSV gives the underlying reason:

| group                                         | count   | share  |
|-----------------------------------------------|--------:|-------:|
| no mood tag (theme / ambiguous only)          |  6 293  | 34.0 % |
| one mood tag                                  | 12 193  | 66.0 % |
| multiple mood tags, all in same class         |    0    |  0.0 % |
| multiple mood tags, spanning ≥ 2 v3 classes   |    0    |  0.0 % |

**Every** moodtheme clip in the entire corpus has at most one mood tag
under the 38-tag mapping. The two conflict rules therefore produce the
same training data for any subset; their accuracy gap is pure
stochastic noise from the training trajectory.

| conflict rule        | best val | test acc | train-val gap @ best |
|----------------------|---------:|---------:|---------------------:|
| drop_on_disagreement |    0.359 |   0.350  |             +0.059   |
| first_match          |    0.342 |   0.333  |             +0.109   |

(Chance for a 6-class head with this balance ≈ 0.20 if it always
predicts `calm`. The head sits ~1.5–1.8× over that majority baseline.)

The same-manifest, same-seed runs land within ±0.017 of each other —
the gap is the training-loop's stochastic ordering, not the mapping
rule. The numbers below are the `drop_on_disagreement` head (kept as
the default).

Mood-head confusion (`drop_on_disagreement`, row-normalised, test = 117
clips):

```
       aggr  happ   sad  calm  dark  epic   support
 aggr  0.12  0.12  0.00  0.44  0.31  0.00      16
 happ  0.25  0.31  0.06  0.31  0.06  0.00      16
  sad  0.00  0.00  0.14  0.62  0.05  0.19      21
 calm  0.03  0.10  0.13  0.70  0.03  0.00      30
 dark  0.14  0.14  0.10  0.43  0.19  0.00      21
 epic  0.08  0.00  0.08  0.39  0.00  0.46      13
```

Findings:

- **`calm` is the v3 mood sink.** Every other class collapses partly
  into it — `sad` 62 %, `aggressive` 44 %, `dark` 43 %, `epic` 39 %.
  Same shape as v2's "pop sink" on the genre head: the model bets on
  the largest class when uncertain. `calm` is the largest class (26 %
  of the corpus) so this is the high-prior safe call.
- **`calm` itself is 70 % correct** — the head learned the calm
  signature well, it just over-applies it.
- **`epic` is the cleanest minority class at 46 % correct** despite
  only 13 test clips and only 92 training clips. PANNs's "cinematic"
  features survive transfer.
- **`calm ↔ sad` is asymmetric**: `sad → calm` 62 %, `calm → sad`
  13 %. The low-arousal axis collapses sad into calm but not the
  other way — calm clips have a more distinct timbre fingerprint than
  sad ones.
- **`aggressive ↔ epic` is *not* the dominant confusion** the design
  draft expected: `aggressive → epic` 0 %, `epic → aggressive` 8 %.
  PANNs distinguishes these reasonably well. The `aggressive` errors
  go to `calm` (44 %) and `dark` (31 %) instead.
- The mood head is genuinely limited by data size at this scale — 92
  epic training clips is the floor we hit. Doubling the archive count
  (12 archives, ~6 GB) would roughly double per-class support and
  likely push the head past 0.40 test accuracy.

The artist-non-overlap split is exact (MTG ships per-track artist
IDs); `train_mood_tl.py` fills test, then val, then train, so an
artist's tracks never straddle the boundary.

**Tag mapping — 59 MTG tags → 6 v3 mood classes**

`groovebot/style/mood_mapping.py` ships an editable mapping:

- 38 mood tags routed to `{aggressive, happy, sad, calm, dark, epic}`.
- 18 theme tags (`advertising`, `christmas`, `game`, ...) dropped from
  training.
- 3 ambiguous tags (`cool`, `melodic`, `powerful`) dropped.
- Conflict rule defaults to `drop_on_disagreement` (cleaner data);
  `first_match` falls back to `MOODS` order when needed.

`tests/test_mood_mapping.py` guards the snapshot: every MTG tag must
be either mapped or dropped, every mapped class must live in `MOODS`,
and the canonical tag count is asserted (59) so a quiet MTG upstream
revision shouts at us.

### Limits (v3, honest)

- **PANNs CNN14 was pretrained on AudioSet** which contains some
  Jamendo / GTZAN distribution. The +38 pp jump is partly genuine
  feature transfer and partly that pretraining and target distributions
  overlap. It is still a fair feature-quality measure, but should not
  be cited as a from-scratch result.
- **Mood head is bounded by MTG corpus size + balance.** v3 ships a
  6-archive (~3.1 GB / 780 mood-kept clips) result; `calm` is over-
  represented and acts as the prediction sink. Doubling the archive
  count would raise per-class support and likely lift test acc above
  0.40, at a 6 GB data cost.
- **The conflict rule is moot for MTG.** No clip in the full 18 486-
  row moodtheme corpus has mood tags spanning ≥ 2 of our 6 classes
  under the current 38-tag mapping. `drop_on_disagreement` and
  `first_match` produce byte-identical manifests; their accuracy
  delta (~±0.02) is pure training-loop noise. If a future mapping
  splits e.g. `aggressive` into multiple buckets, conflicts would
  surface; today the rule is informational only.
- **MTG download is gated behind the upstream tool.** We deliberately do
  not redistribute their audio; v3 wires the manifest / training but
  does not ship a one-click data pull. Use the bounded `--from / --to`
  flags.
- **JointCommand bridge still deferred.** v3 outputs are still text
  labels via the same `GrooveStyle` dataclass; the path to actual
  joint trajectories (M3 generator) is the next milestone.

**v3 arousal/valence — DEAM static annotations**

1. Get the audio + annotations. Direct curl from the canonical mirror —
   no Kaggle auth needed. **1.25 GB audio + 4.5 MB annotations.**

   ```bash
   mkdir -p data/raw/deam
   cd data/raw/deam
   curl -fSL --retry 3 -o DEAM_audio.zip \
       'https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip'
   curl -fSL --retry 3 -o DEAM_Annotations.zip \
       'https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip'
   python -c "import zipfile; zipfile.ZipFile('DEAM_audio.zip').extractall('.'); zipfile.ZipFile('DEAM_Annotations.zip').extractall('.')"
   ```

   Layout: `data/raw/deam/MEMD_audio/<song_id>.mp3` (1802 files) +
   `data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_*.csv`.
   License: CC BY-NC; `data/` is in .gitignore so nothing is re-distributed.

2. Train the regression head (frozen PANNs + 2 x linear). Embedding
   precompute is ~12 min on CPU, head training is instantaneous.

   ```bash
   python -m experiments.train_arousal_tl \
       --static-csv 'data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv' \
       --static-csv 'data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv' \
       --audio-root data/raw/deam \
       --panns-ckpt 'data/raw/Cnn14_mAP=0.431.pth' \
       --cache-dir data/style_emb_deam \
       --out-dir data/style_v3_arousal \
       --epochs 50 --batch-size 64 --early-stopping-patience 15
   ```

**v3 arousal — real numbers (DEAM 1802 songs, song-disjoint 1262/270/270)**

| target  | val R²  | val RMSE | val r  | test R² | test RMSE | test r | literature |
|---------|--------:|---------:|-------:|--------:|----------:|-------:|-----------:|
| arousal |  0.549  |   0.901  |  0.742 |  0.522  |    0.886  |  0.723 | R²≈0.6     |
| valence |  0.398  |   0.898  |  0.653 |  0.451  |    0.875  |  0.675 | R²≈0.4     |

RMSE is on the DEAM SAM 1..9 scale. Early stopped at epoch 26 (patience 15).

**Heuristic vs learned — does the v2 heuristic hold up?**

The v2 `estimate_arousal()` (RMS x onset density, 0..1) was untested
against ground truth. The trainer measures both on the same 270 test
clips and reports the Pearson correlation:

| arousal source | Pearson r vs DEAM truth | explained variance |
|----------------|------------------------:|-------------------:|
| heuristic (v2) |                  0.422  |             ~18 %  |
| learned (v3)   |                  0.723  |             ~52 %  |

Verdict: the heuristic is directionally right but weak (it over-
estimates the corpus mean by +0.16 on the unit scale). The learned
head wins by +0.30 r / +34 pp explained variance. We keep the
heuristic as the default for the no-PANNs fast path and add
`make_panns_arousal_fn(backbone, head)` for the v3 wiring:

```python
from groovebot.style.backbone import PannsBackbone
from groovebot.style.model import StyleRegressionHead
from groovebot.style.select import GrooveStyleSelector, make_panns_arousal_fn
import torch

backbone = PannsBackbone('data/raw/Cnn14_mAP=0.431.pth')
head = StyleRegressionHead()
ck = torch.load('data/style_v3_arousal/style_head_arousal.pt', map_location='cpu')
head.load_state_dict(ck['state_dict'])
arousal_fn = make_panns_arousal_fn(backbone, head)  # 0..1, DEAM-calibrated
selector = GrooveStyleSelector(arousal_fn=arousal_fn, ...)
```

`GrooveStyle` public output is unchanged.

### Limits (v3 arousal/valence)

- **Frozen-backbone ceiling.** R² = 0.52 (arousal) / 0.45 (valence)
  is the floor for "frozen PANNs + small MLP." Fine-tuning the
  backbone or moving to a Music Tagging Transformer would close the
  -0.08 gap to literature on arousal, but is out of scope here.
- **DEAM static only.** Song-level (one label per song) — dynamic
  (per-second) annotations are not used. The M2 online arousal
  estimator will need a different head (the dynamic CSVs are in the
  same archive).
- **Calibration is linear.** `sam_to_unit(v, 1, 9) = (v - 1) / 8`.
  The truth mean (4.76 SAM ≈ 0.47 unit) sits in the "mid" bucket
  of `arousal_bucket()`; the heuristic shifts it +0.16 too high.
- **AudioSet -> DEAM transfer.** PANNs CNN14 was pretrained on
  AudioSet; the overlap with DEAM is smaller than with MTG/GTZAN
  but not zero. Read the R² as a frozen-embedding feature-quality
  measure, not a from-scratch result.

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
    beat_tracker.py           BeatTrackerPerception wrapping BeatNet (fallback; spec §14.3)
  align/                      M0' offline reference alignment (local, librosa-only)
    features.py               chroma / pyin-pitch features -> (12, T) for DTW
    dtw_align.py              OfflineDTWAligner + map_reference_beats
    reference.py              ReferenceBundle + build_reference (Demucs lazy import) for Tier 2
    midi_ref.py               MidiReference + load_reference_from_midi (pretty_midi; DAMP-S-AG MIDI route)
  style/                      GrooveStyleSelector (startup style picker, parallel track)
    features.py               log-mel spectrogram (5-10 s startup window; v1/v2 path)
    model.py                  StyleCNN (v1/v2) + StyleHead + StyleRegressionHead (v3 arousal/valence)
    augment.py                random_time_crop + SpecAugment (train-only)
    backbone.py               PannsBackbone — frozen PANNs CNN14 + .npy cache (v3 transfer)
    mood_mapping.py           MTG-Jamendo 59 moodtheme tags → 6 v3 mood classes
    deam.py                   DEAM static-annotation loader + SAM<->unit calibrator (v3 arousal)
    attributes.py             tempo (librosa.beat) + arousal heuristic (RMS × onset density)
    table.py                  Yuki's nori table: (genre, arousal, mood probs) -> (move, intensity)
    select.py                 GrooveStyleSelector — pluggable arousal_fn + make_panns_arousal_fn
tools/
  eval_beat.py                evaluation CLI (--bpm click GT or --beats annotation; F/CMLt/AMLt + RT-factor + PNG). Scorer reused by M0'.
  synth_warp.py               apply time-stretch rates to (wav + .beats) -> warped (wav + .beats) for M0' Tier 1
  prep_dataset.py             public-dataset prep: annotation -> .beats; Demucs vocal separation (Colab/Kaggle)
  ingest_damp.py              DAMP-VSEP / DAMP-S-AG adapter: list (discovery) + damp-s-ag (stream tarball subset)
  gtzan_split.py              full GTZAN discovery (sf.info probe) + naive stratified + jongpillee fault-filtered splits
  ingest_mtg_moodtheme.py     MTG-Jamendo moodtheme TSV → (path, mood_class, artist_id) manifest (v3)
  _build_m0_notebook.py       regenerates notebooks/m0_gtzan_eval.ipynb (source of truth)
experiments/
  run_gtzan_eval.py           Colab-side engine: select/convert/separate/evaluate/aggregate (shelved blind path)
  run_m0p_align.py            M0' Tier 1 runner: synth_warp -> features -> DTW -> recovered beats -> score
  run_m0p_t2.py               M0' Tier 2 runner: build_reference (vocal+melody) -> DTW per rendition -> score
  run_m0p_t2_damp.py          M0' Tier 2 DAMP runner: backing -> beats/chroma; designated/consensus melody; chroma + pitch paths
  train_style.py              GrooveStyleSelector v1/v2: genre (GTZAN) + mood (stub) multi-head training; CPU
  train_genre_tl.py           v3 genre TL: PANNs CNN14 embeddings (GTZAN fault split) -> MLP head; CPU
  train_mood_tl.py            v3 mood TL: PANNs CNN14 embeddings (MTG-Jamendo or stub) -> MLP head; CPU
  train_arousal_tl.py         v3 arousal/valence TL: PANNs CNN14 embeddings (DEAM or stub) -> regression head + heuristic vs truth cross-check; CPU
notebooks/
  m0_gtzan_eval.ipynb         turnkey Colab notebook for the GTZAN sweep
demo_groove.py                end-to-end loop driven by the orchestrator
tests/                        pytest: limits, body-agnostic, orchestrator, eval, tracker
docs/SYSTEM_SPEC.md           the spec (the canonical reference)
train/PIPELINE.md             the B-3 training pipeline (AIST++ → codebook → robot)
requirements.txt              core deps (local: mujoco, mir_eval, soundfile, matplotlib, pytest)
requirements-experiments.txt  two profiles: local (librosa for M0' alignment) + Colab/Kaggle (BeatNet + Demucs, shelved)
```
