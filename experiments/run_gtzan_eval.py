"""
experiments/run_gtzan_eval.py — engine for the Colab/Kaggle M0 evaluation.

The Colab notebook (`notebooks/m0_gtzan_eval.ipynb`) is a thin wrapper around
the functions defined here:

    _select_with_report  →  for each: convert_one_annotation
                                      separate_one_vocal      (Demucs, lazy import)
                                      evaluate_one_track      (BeatNet, lazy import)
                          →  PipelineReport(results, stage counts, drops)

Heavy deps (Demucs / BeatNet / torch) are imported INSIDE the per-track
functions, never at module load. Selection and aggregation are pure Python
and are covered by `tests/test_run_gtzan_eval.py` locally.

**Transparency (post-mortem of the n_tracks=0 incident, 2026-06-07).**
Earlier the pipeline swallowed two layers of silent failure:
  (i) `select_tracks` dropped audio files with no annotation match without
      any signal — when the GTZAN audio used `<genre>.<NNNNN>.wav` but the
      annotations used `gtzan_<genre>_<NNNNN>.beats`, EVERY track was dropped
      and the aggregator just reported `n_tracks=0`.
  (ii) per-track exceptions were caught and recorded as TrackResult.error
       but the caller had no easy way to see the dropoff per stage.
The fixes:
  - `_annotation_for` now tries the `gtzan_<genre>_<NNNNN>` and underscore
    variants in addition to the plain `<stem>.beats` form.
  - `run_pipeline` returns `PipelineReport`, which carries per-stage counts
    (audio_found / after_genre_filter / with_annotation / converted /
    separated / evaluated) and a `drops` list of (track_id, stage, error).
    `PipelineReport` is list-like, so callers that did `for r in results:`
    keep working unchanged.

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


# Stage labels used by PipelineReport.drops and TrackResult.stage.
STAGE_PAIR = "pair_annotation"
STAGE_CONVERT = "convert"
STAGE_SEPARATE = "separate"
STAGE_EVALUATE = "evaluate"


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
    """One pipeline result. `score` is None iff `error` is set.

    `stage` records WHICH stage produced `error` (one of STAGE_CONVERT /
    STAGE_SEPARATE / STAGE_EVALUATE); None for successful results.
    """
    genre: str
    track_id: str
    score: Optional["object"] = None     # tools.eval_beat.BeatScores
    error: Optional[str] = None
    stage: Optional[str] = None


@dataclass
class GenreSummary:
    genre: str
    n_tracks: int
    f_mean: float
    cmlt_mean: float
    amlt_mean: float
    rt_mean: float


@dataclass
class PipelineReport:
    """End-to-end run report.

    list-like over `results` so existing code that iterates the return value
    (`for r in results:`) still works.
    """
    results: list[TrackResult]
    n_audio_found: int
    n_after_genre_filter: int
    n_with_annotation: int     # = number of tracks that entered the per-track loop
    n_converted: int
    n_separated: int
    n_evaluated: int
    drops: list[tuple[str, str, str]] = field(default_factory=list)
    # (track_id, stage, error_message) — captured at every stage including
    # the pre-loop annotation pairing.

    # list-like protocol
    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    def summary(self) -> str:
        """Human-readable multi-line summary for the notebook to print."""
        lines = [
            "pipeline report",
            f"  audio files found     : {self.n_audio_found}",
            f"  after genre filter    : {self.n_after_genre_filter}",
            f"  with matched annotation: {self.n_with_annotation}",
            f"  conversion succeeded  : {self.n_converted}",
            f"  separation succeeded  : {self.n_separated}",
            f"  evaluation succeeded  : {self.n_evaluated}",
            f"  total drops           : {len(self.drops)}",
        ]
        if self.drops:
            head = self.drops[:10]
            lines.append("  first drops:")
            for tid, stage, err in head:
                lines.append(f"    {tid:<28s}  [{stage}]  {err}")
            if len(self.drops) > 10:
                lines.append(f"    ... and {len(self.drops) - 10} more")
        return "\n".join(lines)


@dataclass
class _SelectionReport:
    selected: list[TrackPaths]
    n_audio_found: int
    n_after_genre_filter: int
    n_with_annotation: int
    drops: list[tuple[str, str, str]]


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


def _annotation_candidates_for(track_stem: str) -> list[str]:
    """Possible annotation filename stems for a GTZAN audio stem.

    GTZAN audio is `<genre>.<NNNNN>.wav` (e.g. `blues.00000`). Different
    annotation distributions name them differently:
      - `<genre>.<NNNNN>.beats`         original GTZAN naming
      - `gtzan_<genre>_<NNNNN>.beats`   TempoBeatDownbeat/gtzan_tempo_beat
      - `<genre>_<NNNNN>.beats`         underscore-separated variants
    """
    cands = [track_stem]
    parts = track_stem.split(".")
    if len(parts) == 2:
        genre, idx = parts
        cands.append(f"gtzan_{genre}_{idx}")
        cands.append(f"{genre}_{idx}")
    return cands


def _annotation_for(track_stem: str, annotation_root: Path) -> Optional[Path]:
    """Pair an audio stem with its annotation file, trying every common
    GTZAN naming convention. Returns the first hit (`.beats`, then `.txt`,
    then `.lab`)."""
    annotation_root = Path(annotation_root)
    for cand in _annotation_candidates_for(track_stem):
        for ext in (".beats", ".txt", ".lab"):
            hits = list(annotation_root.rglob(f"{cand}{ext}"))
            if hits:
                return hits[0]
    return None


def _select_with_report(
    audio_root: Path,
    annotation_root: Path,
    genres: Iterable[str],
    per_genre_limit: int,
) -> _SelectionReport:
    wanted = set(genres)
    audio_root, annotation_root = Path(audio_root), Path(annotation_root)
    audio_files = discover_audio_files(audio_root)
    n_audio_found = len(audio_files)

    # Stage 1: genre filter (no drop list here — those are simply out of scope).
    in_genre: list[tuple[Path, str]] = []
    for ap in audio_files:
        g = _genre_from_filename(ap)
        if g is not None and g in wanted:
            in_genre.append((ap, g))
    n_after_genre_filter = len(in_genre)

    # Stage 2: pair with annotations. Misses are recorded as drops.
    drops: list[tuple[str, str, str]] = []
    selected_by_genre: dict[str, list[TrackPaths]] = {g: [] for g in wanted}
    n_paired_total = 0
    for ap, g in in_genre:
        ann = _annotation_for(ap.stem, annotation_root)
        if ann is None:
            drops.append((ap.stem, STAGE_PAIR,
                          f"no annotation found under {annotation_root}"))
            continue
        n_paired_total += 1
        # Cap per genre (silent — these aren't drops, just not in this run).
        if len(selected_by_genre[g]) >= per_genre_limit:
            continue
        selected_by_genre[g].append(TrackPaths(
            genre=g, track_id=ap.stem,
            audio_path=ap, annotation_path=ann,
        ))

    flat: list[TrackPaths] = []
    for g in sorted(selected_by_genre.keys()):
        flat.extend(sorted(selected_by_genre[g], key=lambda t: t.track_id))
    return _SelectionReport(
        selected=flat,
        n_audio_found=n_audio_found,
        n_after_genre_filter=n_after_genre_filter,
        n_with_annotation=n_paired_total,
        drops=drops,
    )


def select_tracks(
    audio_root: Path,
    annotation_root: Path,
    genres: Iterable[str] = VOCAL_GENRES,
    per_genre_limit: int = 5,
) -> list[TrackPaths]:
    """Return up to `per_genre_limit` tracks per genre in `genres`, paired
    with annotation files. Tracks without a matching annotation are skipped.
    Use `_select_with_report` (or `run_pipeline`) when you also need to know
    *why* tracks were dropped.
    """
    return _select_with_report(audio_root, annotation_root,
                               genres, per_genre_limit).selected


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
) -> PipelineReport:
    """End-to-end loop: select → convert → separate → evaluate.

    Heavy steps are injectable; defaults pull the lazy-import real ones. Per-
    track exceptions are CAUGHT and recorded — both inside the returned
    `TrackResult.error/stage` AND inside `PipelineReport.drops`. They are
    NOT silently swallowed: every drop appears in the report.
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

    sel = _select_with_report(audio_root, annotation_root, genres,
                              per_genre_limit)
    drops = list(sel.drops)

    if verbose:
        print(f"selection: audio_found={sel.n_audio_found}  "
              f"in_genres={sel.n_after_genre_filter}  "
              f"with_annotation={sel.n_with_annotation}  "
              f"selected={len(sel.selected)}")
        if sel.n_after_genre_filter > 0 and sel.n_with_annotation == 0:
            print("  WARN: no audio matched any annotation. Check that the "
                  "annotation root contains files like 'gtzan_<genre>_<idx>.beats' "
                  "or '<genre>.<idx>.beats'.")

    results: list[TrackResult] = []
    n_converted = n_separated = n_evaluated = 0
    total = len(sel.selected)

    for i, track in enumerate(sel.selected, 1):
        tag = f"[{i:>3d}/{total}] {track.track_id}"

        # --- convert ---
        try:
            beats_file = convert_fn(track, beats_dir)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            drops.append((track.track_id, STAGE_CONVERT, err))
            results.append(TrackResult(genre=track.genre, track_id=track.track_id,
                                       stage=STAGE_CONVERT, error=err))
            if verbose:
                print(f"{tag}  DROP[convert]  {err}")
            continue
        n_converted += 1

        # --- separate (Demucs) ---
        try:
            vocal_wav = separate_fn(track, vocal_dir)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            drops.append((track.track_id, STAGE_SEPARATE, err))
            results.append(TrackResult(genre=track.genre, track_id=track.track_id,
                                       stage=STAGE_SEPARATE, error=err))
            if verbose:
                print(f"{tag}  DROP[separate]  {err}")
            continue
        n_separated += 1

        # --- evaluate (BeatNet + mir_eval) ---
        png_out = eval_dir / f"{track.track_id}.png"
        try:
            score = evaluate_fn(vocal_wav, beats_file, png_out)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            drops.append((track.track_id, STAGE_EVALUATE, err))
            results.append(TrackResult(genre=track.genre, track_id=track.track_id,
                                       stage=STAGE_EVALUATE, error=err))
            if verbose:
                print(f"{tag}  DROP[evaluate]  {err}")
            continue
        n_evaluated += 1

        results.append(TrackResult(genre=track.genre, track_id=track.track_id,
                                   score=score))
        if verbose:
            print(f"{tag}  F={score.f_measure:.3f}  "
                  f"CMLt={score.cmlt:.3f}  AMLt={score.amlt:.3f}  "
                  f"RT={score.rt_factor:.2f}x")

    return PipelineReport(
        results=results,
        n_audio_found=sel.n_audio_found,
        n_after_genre_filter=sel.n_after_genre_filter,
        n_with_annotation=sel.n_with_annotation,
        n_converted=n_converted,
        n_separated=n_separated,
        n_evaluated=n_evaluated,
        drops=drops,
    )


# --------------------------------------------------------------------------- #
# Aggregation + IO — pure Python (no pandas/torch). Tested locally.
# --------------------------------------------------------------------------- #
def aggregate(results: Iterable[TrackResult]
              ) -> tuple[list[GenreSummary], GenreSummary]:
    """Per-genre and overall means over successful tracks (errors skipped).

    Accepts both `list[TrackResult]` and `PipelineReport` (the latter is
    iterable over its results).
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


def save_csv(results: Iterable[TrackResult], out_path: Path) -> None:
    """One row per track. Errored tracks have empty metric columns and a
    populated `stage` column."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["genre", "track_id", "f_measure", "cmlt", "amlt",
              "rt_factor", "n_gt", "n_est", "stage", "error"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row: dict[str, object] = {
                "genre": r.genre, "track_id": r.track_id,
                "stage": r.stage or "", "error": r.error or "",
            }
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
