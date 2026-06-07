"""Regenerate `notebooks/m0_gtzan_eval.ipynb`.

The notebook is the deliverable; this script is its source of truth so future
edits stay structured. Run:

    python -m tools._build_m0_notebook

It rewrites the .ipynb file with deterministic content (sorted keys, fixed
metadata). The notebook is then committed alongside this script.
"""
from __future__ import annotations
import json
from pathlib import Path


REPO_URL = "https://github.com/yukijames25/groovebot.git"
REPO_NAME = "groovebot"                  # `git clone` directory name
NOTEBOOK_OUT = Path(__file__).resolve().parents[1] / "notebooks" / "m0_gtzan_eval.ipynb"


def md(text: str) -> dict:
    """Build a markdown cell. `text` may contain blank lines; we preserve them."""
    lines = text.splitlines(keepends=True)
    # Jupyter trims the trailing \n of the last line; that's fine.
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(src: str) -> dict:
    """Build a code cell. Always cleared (no outputs, no execution count)."""
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    # 1 — Title + summary
    cells.append(md(f"""\
# GrooveBot M0 — beat tracking on GTZAN-Rhythm (Colab)

End-to-end M0 evaluation on the **vocal-heavy** GTZAN-Rhythm subset
(pop/rock/hiphop/reggae/blues/country/disco; classical/jazz/metal excluded
because they are vocal-light or instrumental).

For each track we:

1. take the original full-mix audio,
2. convert the dataset's beat annotation to our `--beats` format
   (`tools.prep_dataset`),
3. separate the vocal stem with **Demucs**,
4. run **BeatNet** in `online` mode (causal, NFR-2) on the vocal stem,
5. score against the GTZAN ground truth with `mir_eval` — F / CMLt / AMLt —
   and record an RT-factor.

See `docs/SYSTEM_SPEC.md` §10.2 ("公開データ方式") and §14 for the wider
context. This notebook is a thin wrapper around
`experiments/run_gtzan_eval.py`; the heavy lifting (Demucs, BeatNet) is
imported lazily on first use.

**Data sources**

- Audio: [GTZAN](http://marsyas.info/downloads/datasets.html) (10 genres,
  1000 30-sec clips). Fetched via the `gtzan_mini` mirror first; falls back
  to a Kaggle mirror if needed.
- Annotations: [TempoBeatDownbeat/gtzan_tempo_beat](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat)
  (Marchand & Peeters 2015).
"""))

    # 2 — User action checklist
    cells.append(md("""\
## What you (human) need to do

1. **Runtime → Change runtime type → GPU.** Demucs and BeatNet will be
   painfully slow on CPU. The notebook still runs without one; expect long
   waits.
2. Adjust **`PER_GENRE_LIMIT`** in the config cell below if you want more or
   fewer tracks per genre. The default (5) gives 35 tracks total and runs in
   roughly 20–30 minutes on a T4.
3. *(Only if the gtzan_mini path fails)* upload `kaggle.json` when prompted
   so the Kaggle fallback can fetch the audio.

Everything else runs unattended."""))

    # 3 — GPU check
    cells.append(code("""\
# GPU check — warn (don't abort) if there's no CUDA device. Demucs and
# BeatNet will still run on CPU; they'll just be much slower.
import subprocess
try:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                          text=True, timeout=5)
    print(out.stdout or "(nvidia-smi returned nothing)")
except Exception as e:
    print(f"no GPU detected ({e}); CPU-only run will be slow")"""))

    # 4 — Clone repo
    cells.append(code(f"""\
# Clone the GrooveBot repo. Public, no auth needed.
REPO_URL = "{REPO_URL}"
REPO_NAME = "{REPO_NAME}"

import os, subprocess
if not os.path.isdir(REPO_NAME):
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_NAME], check=True)
%cd {{REPO_NAME}}
!git rev-parse --short HEAD"""))

    # 5 — Install requirements
    cells.append(code("""\
# Core + experiments deps. requirements.txt is light; requirements-experiments
# brings in BeatNet (-> torch + madmom + librosa) + Demucs (-> torch + diffq).
!pip install -q -r requirements.txt
!pip install -q -r requirements-experiments.txt

# Smoke-check: every import we need should succeed. A failure here is much
# easier to diagnose than a NameError 20 minutes into the run.
import importlib, sys
for mod in ("mir_eval", "soundfile", "matplotlib", "numpy",
            "BeatNet", "demucs", "torch"):
    try:
        importlib.import_module(mod.split(".")[0])
        print(f"  ok  {mod}")
    except Exception as e:
        print(f"  FAIL  {mod}: {e}", file=sys.stderr)"""))

    # 6 — Data fetch markdown
    cells.append(md("""\
## Data fetch

We try the GitHub mirror `TempoBeatDownbeat/gtzan_mini` first because it
ships audio + a small index already aligned with the Marchand & Peeters
beat annotations. If that repo lacks audio (it sometimes ships
annotations only) we fall back to the Kaggle GTZAN mirror, which needs
your `kaggle.json` API token.

Annotations come from `TempoBeatDownbeat/gtzan_tempo_beat` in either case."""))

    # 7 — gtzan_mini attempt
    cells.append(code("""\
# Try gtzan_mini for audio first.
import os, subprocess
from pathlib import Path

DATA_DIR = Path("data/gtzan")
DATA_DIR.mkdir(parents=True, exist_ok=True)

GTZAN_MINI = DATA_DIR / "gtzan_mini"
if not GTZAN_MINI.exists():
    subprocess.run(["git", "clone", "--depth=1",
                    "https://github.com/TempoBeatDownbeat/gtzan_mini.git",
                    str(GTZAN_MINI)], check=False)

# Locate wav/au audio anywhere under the mini repo. If we find any, prefer it.
audio_candidates = list(GTZAN_MINI.rglob("*.wav")) + list(GTZAN_MINI.rglob("*.au"))
USE_KAGGLE = len(audio_candidates) == 0
AUDIO_ROOT = GTZAN_MINI if not USE_KAGGLE else None
print(f"gtzan_mini audio files found: {len(audio_candidates)}")
print(f"USE_KAGGLE fallback: {USE_KAGGLE}")"""))

    # 8 — Kaggle fallback markdown
    cells.append(md("""\
### Kaggle fallback

If the previous cell printed `USE_KAGGLE fallback: True`, the next cell
prompts you to upload your `kaggle.json` token (Kaggle → Account → Create
New API Token). Otherwise you can skip it."""))

    # 9 — Kaggle fallback code
    cells.append(code("""\
# Only runs if gtzan_mini had no audio.
if USE_KAGGLE:
    from google.colab import files     # type: ignore[reportMissingImports]
    print("Upload kaggle.json (Kaggle → Account → Create New API Token)")
    uploaded = files.upload()

    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !pip install -q kaggle
    !kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/gtzan/kaggle --unzip

    KAGGLE_AUDIO = Path("data/gtzan/kaggle")
    audio_candidates = list(KAGGLE_AUDIO.rglob("*.wav")) + list(KAGGLE_AUDIO.rglob("*.au"))
    AUDIO_ROOT = KAGGLE_AUDIO
    print(f"Kaggle audio files found: {len(audio_candidates)}")
else:
    print("skipping Kaggle fallback (gtzan_mini supplied audio)")"""))

    # 10 — Annotations
    cells.append(code("""\
# Beat annotations from Marchand & Peeters (TempoBeatDownbeat/gtzan_tempo_beat).
ANN_DIR = DATA_DIR / "gtzan_tempo_beat"
if not ANN_DIR.exists():
    subprocess.run(["git", "clone", "--depth=1",
                    "https://github.com/TempoBeatDownbeat/gtzan_tempo_beat.git",
                    str(ANN_DIR)], check=True)
n_ann = sum(1 for _ in ANN_DIR.rglob("*.beats"))
print(f"annotation files: {n_ann}")"""))

    # 11 — Config knob
    cells.append(md("""\
## Configuration

`PER_GENRE_LIMIT` is the knob you turn for a quick smoke vs. full sweep.
With the default (5) you get 35 tracks across 7 vocal genres — enough to
see clear F / CMLt / AMLt gaps without burning hours of GPU time."""))

    # 12 — Run pipeline
    cells.append(code("""\
PER_GENRE_LIMIT = 5
WORK_DIR = Path("data/m0_work")

from experiments.run_gtzan_eval import VOCAL_GENRES, run_pipeline
print(f"genres: {VOCAL_GENRES}")
print(f"per-genre limit: {PER_GENRE_LIMIT}")

report = run_pipeline(
    audio_root=AUDIO_ROOT,
    annotation_root=ANN_DIR,
    work_dir=WORK_DIR,
    genres=VOCAL_GENRES,
    per_genre_limit=PER_GENRE_LIMIT,
    verbose=True,
)
# Always print the full stage breakdown — even (especially!) when zero
# tracks survived. This is how we caught the n_tracks=0 incident on
# 2026-06-07: pre-loop annotation mismatch was invisible until exposed here.
print()
print(report.summary())"""))

    # 13 — Aggregation markdown
    cells.append(md("""\
## Results

Per-genre and overall means. Read the metrics the same way as the README:

- **F** — proximity F-measure (70 ms window).
- **CMLt** — Correct-Metrical-Level, tempo-locked. Drops to 0 on wrong tempo.
- **AMLt** — Allowed-Metrical-Level. Forgives ×2 / ÷2 tempo errors.
- **RT-factor** = process_sec / audio_sec. ≤ 1.0 means realtime-capable.

The CMLt vs AMLt gap is the "octave error" signal — for our humming/vocal
target, a big gap means BeatNet is locking on to a wrong metrical level
even when the absolute beat positions are sane."""))

    # 14 — Aggregate + display + CSV
    cells.append(code("""\
from experiments.run_gtzan_eval import aggregate, save_csv, to_dataframe
import pandas as pd

per_genre, overall = aggregate(report)
df = to_dataframe(per_genre, overall)

CSV_PATH = WORK_DIR / "m0_per_track.csv"
save_csv(report, CSV_PATH)
print(f"per-track CSV: {CSV_PATH}")

# Display the summary table inline. `display` is Colab/Jupyter built-in.
display(df.style.format({
    "f_mean": "{:.3f}",
    "cmlt_mean": "{:.3f}",
    "amlt_mean": "{:.3f}",
    "rt_mean": "{:.2f}",
}).set_caption("M0 — BeatNet (online) on GTZAN vocal-heavy genres"))

df.to_csv(WORK_DIR / "m0_summary.csv", index=False)

# Drop diagnostics. If anything fell out, we want it visible RIGHT NEXT TO
# the (possibly small or zero) summary table — no scrolling required.
if report.drops:
    drops_df = pd.DataFrame(report.drops,
                            columns=["track_id", "stage", "error"])
    by_stage = drops_df["stage"].value_counts().to_dict()
    print(f"\\ndrops by stage: {by_stage}")
    display(drops_df.head(20))
    drops_df.to_csv(WORK_DIR / "m0_drops.csv", index=False)"""))

    # 15 — Closing notes
    cells.append(md(f"""\
## Where the artifacts live

After the run, `data/m0_work/` (under the cloned repo on Colab) contains:

```
data/m0_work/
  beats/                  GTZAN annotations converted to --beats format
  vocal/htdemucs/<id>/    Demucs stems (vocals.wav + other.wav)
  eval/<id>.png           per-track waveform + GT vs detected overlay
  m0_per_track.csv        one row per track
  m0_summary.csv          per-genre + overall means
```

Download whatever you want from the file pane on the left, or zip+drop:

```python
!cd data && zip -r m0_work.zip m0_work
from google.colab import files; files.download("data/m0_work.zip")
```

To re-run with more tracks, change `PER_GENRE_LIMIT` and re-execute the
last two cells; everything before is idempotent.

---

Re-generating this notebook (locally): `python -m tools._build_m0_notebook`
from the repo root. The source of truth is `tools/_build_m0_notebook.py`."""))

    return cells


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    nb = build_notebook()
    NOTEBOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_OUT.write_text(
        json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NOTEBOOK_OUT}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
