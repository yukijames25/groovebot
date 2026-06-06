"""
experiments/run_gtzan_eval.py — engine for the Colab/Kaggle M0 evaluation.

The Colab notebook (`notebooks/m0_gtzan_eval.ipynb`) is a thin wrapper around
the functions defined here:

    select_tracks  →  for each: convert_one_annotation
                                separate_one_vocal      (Demucs, lazy import)
                                evaluate_one_track      (BeatNet, lazy import)
                  →  aggregate, save_csv, (to_dataframe for display)

Heavy deps (Demucs / BeatNet / torch) are imported INSIDE the per-track
functions, never at module load. Selection and aggregation are pure Python and
are covered by `tests/test_run_gtzan_eval.py` locally.

Per-track failures (missing files, BeatNet crashes, Demucs OOM, …) are caught
inside `run_pipeline` and recorded as `TrackResult(error=...)` so one bad
track does not kill the whole sweep.

Scope: GTZAN-Rhythm with the vocal-heavy genres only (no classical/jazz/metal).
The list is exposed as `VOCAL_GENRES` and can be overridden per call.
"""
from __future__ import annotations
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


# Per spec §10.2 + M0 plan: vocal-heavy GTZAN genres only.
VOCAL_GENRES: tuple[str, ...] = (
    "pop", "rock", "hiphop", "reggae", "blues", "country", "disco",
)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackPaths:
    genre: str
    track_id: str           # e.g. "pop.00001"
    audio_path: Path
    annotation_path: Path


@dataclass
class TrackResult:
    """One pipeline result. `score` is None iff `error` is set."""
    genre: str
    track_id: str
    score: Optional["object"] = None     # tools.eval_beat.BeatScores
    error: Optional[str] = None


@dataclass
class GenreSummary:
    genre: str
    n_tracks: int
    f_mean: float
    cmlt_mean: float
    amlt_mean: float
    rt_mean: float


# --------------------------------------------------------------------------- #
# Selection — pure I/O. Tested locally.
# --------------------------------------------------------------------------- #
def _genre_from_filename(p: Path) -> Optional[str]:
    """GTZAN convention: `<genre>.<5-digit-id>.wav` -> genre."""
    parts = p.stem.split(".")
    return parts[0] if len(parts) >= 2 else None


def discover_audio_files(audio_root: Path) -> list[Path]:
    """Find *.wav (and *.au, GTZAN's original) anywhere under audio_root."""
    audio_root = Path(audio_root)
    paths: list[Path] = []
    for ext in ("*.wav", "*.au"):
        paths.extend(audio_root.rglob(ext))
    return sorted(paths)


def _annotation_for(track_stem: str, annotation_root: Path) -> Optional[Path]:
    """Pair an audio stem (`pop.00001`) with its annotation file. Searches
    common conventions (`.beats`, `.txt`, `.lab`) anywhere under root.
    """
    annotation_root = Path(annotation_root)
    for ext in (".beats", ".txt", ".lab"):
        candidates = list(annotation_root.rglob(f"{track_stem}{ext}"))
        if candidates:
            return candidates[0]
    return None


def select_tracks(
    audio_root: Path,
    annotation_root: Path,
    genres: Iterable[str] = VOCAL_GENRES,
    per_genre_limit: int = 5,
) -> list[TrackPaths]:
    """Return up to `per_genre_limit` tracks per genre in `genres`, paired
    with annotation files. Tracks without a matching annotation are skipped.
    Result is sorted by (genre, track_id) for reproducibility.
    """
    wanted = set(genres)
    audio_root, annotation_root = Path(audio_root), Path(annotation_root)
    audio_files = discover_audio_files(audio_root)

    selected: dict[str, list[TrackPaths]] = {g: [] for g in wanted}
    for ap in audio_files:
        g = _genre_from_filename(ap)
        if g not in wanted:
            continue
        if len(selected[g]) >= per_genre_limit:
            continue
        ann = _annotation_for(ap.stem, annotation_root)
        if ann is None:
            continue
        selected[g].append(TrackPaths(
            genre=g, track_id=ap.stem,
            audio_path=ap, annotation_path=ann,
        ))

    flat: list[TrackPaths] = []
    for g in sorted(selected.keys()):
        flat.extend(sorted(selected[g], key=lambda t: t.track_id))
    return flat


# --------------------------------------------------------------------------- #
# Per-track pipeline steps. Heavy deps lazy-imported inside.
# --------------------------------------------------------------------------- #
def convert_one_annotation(track: TrackPaths, beats_out_dir: Path) -> Path:
    """GTZAN-Rhythm annotation -> our --beats format."""
    from tools.prep_dataset import parse_gtzan_rhythm, write_beats_file
    beats_out_dir = Path(beats_out_dir)
    beats_out_dir.mkdir(parents=True, exist_ok=True)
    out = beats_out_dir / f"{track.track_id}.beats"
    times = parse_gtzan_rhythm(str(track.annotation_path))
    write_beats_file(times, str(out))
    return out


def separate_one_vocal(track: TrackPaths, vocal_out_dir: Path,
                       model: str = "htdemucs") -> Path:
    """Run Demucs and return the vocals.wav path. Lazy demucs import."""
    from tools.prep_dataset import separate_vocal
    return Path(separate_vocal(str(track.audio_path), str(vocal_out_dir),
                               model=model))


def evaluate_one_track(vocal_wav: Path, beats_file: Path,
                       png_out: Optional[Path] = None):
    """Causal BeatNet pass + mir_eval scoring + optional overlay PNG."""
    import time
    import numpy as np
    import soundfile as sf
    from groovebot.perception import BeatTrackerPerception
    from tools.eval_beat import (load_beat_annotation, plot_overlay, score_beats)

    wav, sr = sf.read(str(vocal_wav), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    audio_sec = len(wav) / sr

    tracker = BeatTrackerPerception(sample_rate=sr)
    t0 = time.perf_counter()
    events = tracker.process_wav(str(vocal_wav))
    proc_sec = time.perf_counter() - t0
    est = np.array([e.time for e in events], dtype=float)

    gt = load_beat_annotation(str(beats_file))
    scores = score_beats(
        track=Path(vocal_wav).name, bpm=None,
        gt=gt, est=est, audio_sec=audio_sec, proc_sec=proc_sec,
    )
    if png_out is not None:
        Path(png_out).parent.mkdir(parents=True, exist_ok=True)
        plot_overlay(wav, sr, gt, est, str(png_out),
                     title=f"{Path(vocal_wav).name}  "
                           f"F={scores.f_measure:.3f}  "
                           f"CMLt={scores.cmlt:.3f}  AMLt={scores.amlt:.3f}  "
                           f"RT={scores.rt_factor:.2f}x")
    return scores


# --------------------------------------------------------------------------- #
# Orchestration — DI so tests can substitute the heavy parts.
# --------------------------------------------------------------------------- #
def run_pipeline(
    audio_root: Path,
    annotation_root: Path,
    work_dir: Path,
    genres: Iterable[str] = VOCAL_GENRES,
    per_genre_limit: int = 5,
    convert_fn: Optional[Callable[[TrackPaths, Path], Path]] = None,
    separate_fn: Optional[Callable[[TrackPaths, Path], Path]] = None,
    evaluate_fn: Optional[Callable[[Path, Path, Optional[Path]], object]] = None,
    verbose: bool = True,
) -> list[TrackResult]:
    """End-to-end loop: select → convert → separate → evaluate.

    Heavy steps are injectable; defaults pull the lazy-import real ones. Per-
    track exceptions are recorded as TrackResult(error=...) so one bad track
    does not abort the sweep.
    """
    work_dir = Path(work_dir)
    beats_dir = work_dir / "beats"
    vocal_dir = work_dir / "vocal"
    eval_dir = work_dir / "eval"
    for d in (beats_dir, vocal_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    convert_fn = convert_fn or convert_one_annotation
    separate_fn = separate_fn or separate_one_vocal
    evaluate_fn = evaluate_fn or evaluate_one_track

    tracks = select_tracks(audio_root, annotation_root, genres=genres,
                           per_genre_limit=per_genre_limit)
    if verbose:
        print(f"selected {len(tracks)} tracks across "
              f"{len(set(t.genre for t in tracks))} genres")

    results: list[TrackResult] = []
    for i, track in enumerate(tracks, 1):
        tag = f"[{i:>3d}/{len(tracks)}] {track.track_id}"
        try:
            beats_file = convert_fn(track, beats_dir)
            vocal_wav = separate_fn(track, vocal_dir)
            png_out = eval_dir / f"{track.track_id}.png"
            score = evaluate_fn(vocal_wav, beats_file, png_out)
            results.append(TrackResult(genre=track.genre,
                                       track_id=track.track_id, score=score))
            if verbose:
                print(f"{tag}  F={score.f_measure:.3f}  "
                      f"CMLt={score.cmlt:.3f}  AMLt={score.amlt:.3f}  "
                      f"RT={score.rt_factor:.2f}x")
        except Exception as e:
            results.append(TrackResult(genre=track.genre,
                                       track_id=track.track_id,
                                       error=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"{tag}  ERROR: {type(e).__name__}: {e}")
    return results


# --------------------------------------------------------------------------- #
# Aggregation + IO — pure Python (no pandas/torch). Tested locally.
# --------------------------------------------------------------------------- #
def aggregate(results: list[TrackResult]
              ) -> tuple[list[GenreSummary], GenreSummary]:
    """Per-genre and overall means over successful tracks (errors skipped).

    Returns (per_genre_list, overall_summary). If everything errored, the
    overall summary has n_tracks=0 and all means=0.0.
    """
    ok = [r for r in results if r.score is not None and r.error is None]

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    by_genre: dict[str, list[TrackResult]] = {}
    for r in ok:
        by_genre.setdefault(r.genre, []).append(r)

    per_genre = [
        GenreSummary(
            genre=g,
            n_tracks=len(rs),
            f_mean=_mean([r.score.f_measure for r in rs]),
            cmlt_mean=_mean([r.score.cmlt for r in rs]),
            amlt_mean=_mean([r.score.amlt for r in rs]),
            rt_mean=_mean([r.score.rt_factor for r in rs]),
        )
        for g, rs in sorted(by_genre.items())
    ]
    overall = GenreSummary(
        genre="ALL",
        n_tracks=len(ok),
        f_mean=_mean([r.score.f_measure for r in ok]),
        cmlt_mean=_mean([r.score.cmlt for r in ok]),
        amlt_mean=_mean([r.score.amlt for r in ok]),
        rt_mean=_mean([r.score.rt_factor for r in ok]),
    )
    return per_genre, overall


def save_csv(results: list[TrackResult], out_path: Path) -> None:
    """One row per track. Errored tracks have empty metric columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["genre", "track_id", "f_measure", "cmlt", "amlt",
              "rt_factor", "n_gt", "n_est", "error"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row: dict[str, object] = {"genre": r.genre, "track_id": r.track_id,
                                      "error": r.error or ""}
            if r.score is not None:
                row.update({
                    "f_measure": r.score.f_measure,
                    "cmlt": r.score.cmlt,
                    "amlt": r.score.amlt,
                    "rt_factor": r.score.rt_factor,
                    "n_gt": r.score.n_gt,
                    "n_est": r.score.n_est,
                })
            w.writerow(row)


def to_dataframe(per_genre: list[GenreSummary], overall: GenreSummary):
    """Lazy pandas import — for notebook display only. Returns a DataFrame
    whose last row is the ALL/overall summary.
    """
    import pandas as pd
    rows = [asdict(g) for g in per_genre] + [asdict(overall)]
    return pd.DataFrame(rows)
