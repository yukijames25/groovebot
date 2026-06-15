"""groovebot.style.deam — DEAM (MediaEval) loader for the arousal head.

DEAM (Soleymani et al. 2013; Aljanaki et al. 2017) is the canonical
public corpus for music emotion regression: 1,802 clips x ~45 s with
song-level (static) and per-second (dynamic) valence/arousal averaged
across 5+ annotators. Annotations live on a 1-9 SAM scale; we keep that
scale for training and convert to a 0..1 table bucket at inference time.

This module does NOT download. Run the Kaggle CLI yourself once
(`kaggle datasets download imsparsh/deam-mediaeval-dataset`) and pass
the unpacked `--audio-root` + static-CSV path.

Output: `DeamRecord(audio_path, song_id, arousal, valence)`.
The trainer (`experiments/train_arousal_tl.py`) is the only call site.
"""
from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEAM_SAM_LO = 1.0
DEAM_SAM_HI = 9.0


@dataclass
class DeamRecord:
    """One DEAM clip with its static (song-level) annotations."""
    audio_path: Path
    song_id: int
    arousal: float   # 1..9 SAM scale (DEAM raw)
    valence: float   # 1..9 SAM scale (DEAM raw)


def _strip(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _find_audio(audio_root: Path, song_id: int) -> Path | None:
    """DEAM ships per-song audio as `<song_id>.mp3` under MEMD_audio/
    in the canonical release. The Kaggle redistribution sometimes
    flattens to `<audio_root>/<song_id>.mp3` or keeps the MEMD prefix;
    try the common candidates in order."""
    candidates = [
        audio_root / f"{song_id}.mp3",
        audio_root / "MEMD_audio" / f"{song_id}.mp3",
        audio_root / f"{song_id}.wav",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_static_annotations(
    csv_path: Path,
    audio_root: Path,
    *,
    require_audio_present: bool = True,
) -> list[DeamRecord]:
    """Parse a DEAM static-annotations CSV.

    The canonical files are
        `static_annotations_averaged_songs_1_2000.csv` and
        `static_annotations_averaged_songs_2000_2058.csv`,
    with columns including `song_id`, `valence_mean`, `arousal_mean`.
    Column names sometimes ship as ` valence_mean` (leading space) in
    the upstream release — we normalise by lower+strip+space-to-_.

    Rows with missing audio are silently dropped when
    `require_audio_present` (the default; the trainer needs files on
    disk to embed).
    """
    out: list[DeamRecord] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        norm = {_strip(c): c for c in reader.fieldnames}
        if "song_id" not in norm or "valence_mean" not in norm or "arousal_mean" not in norm:
            raise ValueError(
                f"unexpected DEAM header: {reader.fieldnames!r} -- "
                "need song_id, valence_mean, arousal_mean"
            )
        col_id = norm["song_id"]
        col_v = norm["valence_mean"]
        col_a = norm["arousal_mean"]
        for row in reader:
            try:
                song_id = int(row[col_id])
                valence = float(row[col_v])
                arousal = float(row[col_a])
            except (KeyError, ValueError):
                continue
            apath = _find_audio(audio_root, song_id)
            if apath is None:
                if require_audio_present:
                    continue
                apath = audio_root / f"{song_id}.mp3"
            out.append(DeamRecord(
                audio_path=apath, song_id=song_id,
                arousal=arousal, valence=valence,
            ))
    return out


def read_static_annotations_many(
    csv_paths: Iterable[Path],
    audio_root: Path,
    *,
    require_audio_present: bool = True,
) -> list[DeamRecord]:
    """Concat multiple DEAM CSVs (`_1_2000.csv` + `_2000_2058.csv`)
    into one record list, de-duplicating on song_id (last wins)."""
    by_id: dict[int, DeamRecord] = {}
    for p in csv_paths:
        for r in read_static_annotations(
            Path(p), audio_root,
            require_audio_present=require_audio_present,
        ):
            by_id[r.song_id] = r
    return list(by_id.values())


def song_disjoint_split(
    records: list[DeamRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[DeamRecord], list[DeamRecord], list[DeamRecord]]:
    """Split at the song level. DEAM has one annotation per song so
    "song-level" is the only honest split unit; there is no album/
    artist metadata in the static CSV. Shuffle song_ids with the fixed
    seed and slice into test, val, train (in that order)."""
    rng = random.Random(seed)
    ids = sorted({r.song_id for r in records})
    rng.shuffle(ids)
    n_total = len(ids)
    n_test = int(round(n_total * test_frac))
    n_val = int(round(n_total * val_frac))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test: n_test + n_val])
    train_ids = set(ids[n_test + n_val:])
    by_bucket = {"train": [], "val": [], "test": []}
    for r in records:
        if r.song_id in test_ids:
            by_bucket["test"].append(r)
        elif r.song_id in val_ids:
            by_bucket["val"].append(r)
        elif r.song_id in train_ids:
            by_bucket["train"].append(r)
    return by_bucket["train"], by_bucket["val"], by_bucket["test"]


def sam_to_unit(value: float, lo: float = DEAM_SAM_LO, hi: float = DEAM_SAM_HI) -> float:
    """DEAM 1..9 SAM scale -> 0..1 (clamped). Used to map learned
    arousal into `arousal_bucket()` thresholds. Linear; DEAM annotation
    distributions are roughly symmetric so a piecewise calibration is
    not justified at this scale."""
    if hi <= lo:
        raise ValueError(f"sam_to_unit: hi ({hi}) must exceed lo ({lo})")
    u = (float(value) - lo) / (hi - lo)
    return float(max(0.0, min(1.0, u)))


def unit_to_sam(value: float, lo: float = DEAM_SAM_LO, hi: float = DEAM_SAM_HI) -> float:
    """Inverse of `sam_to_unit`. Useful when comparing the existing
    heuristic 0..1 arousal against DEAM ground truth on the SAM scale."""
    if hi <= lo:
        raise ValueError(f"unit_to_sam: hi ({hi}) must exceed lo ({lo})")
    return float(lo + max(0.0, min(1.0, float(value))) * (hi - lo))


__all__ = [
    "DEAM_SAM_LO", "DEAM_SAM_HI", "DeamRecord",
    "read_static_annotations", "read_static_annotations_many",
    "song_disjoint_split", "sam_to_unit", "unit_to_sam",
]
