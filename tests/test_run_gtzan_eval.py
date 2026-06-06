"""experiments/run_gtzan_eval.py — selection + aggregation + DI pipeline.

We never call Demucs or BeatNet here. The heavy steps are passed in as
mock callables; aggregation and selection are tested directly.
"""
from __future__ import annotations
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from experiments.run_gtzan_eval import (
    GenreSummary,
    TrackPaths,
    TrackResult,
    VOCAL_GENRES,
    _genre_from_filename,
    aggregate,
    run_pipeline,
    save_csv,
    select_tracks,
)


# Stand-in for tools.eval_beat.BeatScores so tests don't need the real module.
@dataclass
class _FakeScore:
    f_measure: float
    cmlt: float
    amlt: float
    rt_factor: float
    n_gt: int = 100
    n_est: int = 100


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_track_tree(root: Path, *, audio: list[str], ann: list[str]) -> None:
    """Create dummy audio + annotation files under root for select_tracks
    tests. `audio` and `ann` are lists of bare track stems like 'pop.00001'.
    """
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "ann").mkdir(parents=True, exist_ok=True)
    for stem in audio:
        (root / "audio" / f"{stem}.wav").write_bytes(b"")
    for stem in ann:
        (root / "ann" / f"{stem}.beats").write_text("0.0\n")


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def test_genre_from_filename_handles_gtzan_convention():
    assert _genre_from_filename(Path("pop.00001.wav")) == "pop"
    assert _genre_from_filename(Path("/x/y/hiphop.00042.au")) == "hiphop"


def test_genre_from_filename_returns_none_for_unstructured_name():
    assert _genre_from_filename(Path("recording.wav")) is None


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_select_tracks_filters_by_genre_and_caps_per_genre():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 6 pop + 6 classical + 2 rock; classical must be excluded;
        # pop must cap at 3, rock keeps both.
        audio = (
            [f"pop.{i:05d}" for i in range(6)]
            + [f"classical.{i:05d}" for i in range(6)]
            + [f"rock.{i:05d}" for i in range(2)]
        )
        _make_track_tree(root, audio=audio, ann=audio)

        picked = select_tracks(root / "audio", root / "ann",
                               genres=("pop", "rock"), per_genre_limit=3)
        genres = [t.genre for t in picked]
        assert genres.count("pop") == 3
        assert genres.count("rock") == 2
        assert "classical" not in genres


def test_select_tracks_skips_tracks_without_annotation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = [f"pop.{i:05d}" for i in range(3)]
        _make_track_tree(root, audio=audio, ann=audio[:1])    # only 1 ann
        picked = select_tracks(root / "audio", root / "ann",
                               genres=("pop",), per_genre_limit=10)
        assert len(picked) == 1
        assert picked[0].track_id == "pop.00000"


def test_select_tracks_sort_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Out-of-order genres + ids; output must be sorted.
        audio = ["rock.00002", "pop.00001", "rock.00001", "pop.00002"]
        _make_track_tree(root, audio=audio, ann=audio)
        picked = select_tracks(root / "audio", root / "ann",
                               genres=("pop", "rock"), per_genre_limit=10)
        ids = [(t.genre, t.track_id) for t in picked]
        assert ids == [("pop", "pop.00001"), ("pop", "pop.00002"),
                       ("rock", "rock.00001"), ("rock", "rock.00002")]


def test_vocal_genres_excludes_classical_and_jazz():
    """The M0 plan deliberately drops classical/jazz from the default list."""
    assert "classical" not in VOCAL_GENRES
    assert "jazz" not in VOCAL_GENRES
    assert "metal" not in VOCAL_GENRES


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _ok(genre: str, tid: str, f: float, cmlt: float, amlt: float, rt: float):
    return TrackResult(genre=genre, track_id=tid,
                       score=_FakeScore(f, cmlt, amlt, rt))


def test_aggregate_means_per_genre_and_overall():
    results = [
        _ok("pop", "pop.00001", 0.8, 0.7, 0.9, 0.5),
        _ok("pop", "pop.00002", 0.6, 0.5, 0.7, 0.6),
        _ok("rock", "rock.00001", 0.4, 0.3, 0.5, 0.7),
    ]
    per_genre, overall = aggregate(results)
    by = {g.genre: g for g in per_genre}
    assert by["pop"].n_tracks == 2
    assert by["pop"].f_mean == pytest.approx((0.8 + 0.6) / 2)
    assert by["rock"].n_tracks == 1
    assert by["rock"].f_mean == pytest.approx(0.4)
    assert overall.n_tracks == 3
    assert overall.f_mean == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert overall.rt_mean == pytest.approx((0.5 + 0.6 + 0.7) / 3)


def test_aggregate_excludes_errored_tracks_from_means():
    results = [
        _ok("pop", "pop.00001", 1.0, 1.0, 1.0, 0.5),
        TrackResult(genre="pop", track_id="pop.00002", error="BeatNet crashed"),
    ]
    per_genre, overall = aggregate(results)
    assert overall.n_tracks == 1
    assert overall.f_mean == 1.0


def test_aggregate_all_errored_returns_zero_means():
    results = [TrackResult(genre="pop", track_id="x", error="boom")]
    per_genre, overall = aggregate(results)
    assert per_genre == []
    assert overall.n_tracks == 0
    assert overall.f_mean == 0.0


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_save_csv_roundtrip_columns_and_rows():
    results = [
        _ok("pop", "pop.00001", 0.8, 0.7, 0.9, 0.5),
        TrackResult(genre="pop", track_id="pop.00002", error="BeatNet OOM"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "out.csv"
        save_csv(results, p)
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["track_id"] == "pop.00001"
    assert float(rows[0]["f_measure"]) == pytest.approx(0.8)
    assert rows[0]["error"] == ""
    assert rows[1]["error"] == "BeatNet OOM"
    assert rows[1]["f_measure"] == ""        # blank for errored row


# --------------------------------------------------------------------------- #
# run_pipeline — DI: skip Demucs + BeatNet entirely.
# --------------------------------------------------------------------------- #
def test_run_pipeline_with_mocks_walks_select_and_collects_scores():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = [f"pop.{i:05d}" for i in range(3)] + ["classical.00001"]
        _make_track_tree(root, audio=audio, ann=audio)

        captured: list[str] = []

        def fake_convert(track: TrackPaths, beats_out_dir: Path) -> Path:
            out = beats_out_dir / f"{track.track_id}.beats"
            out.write_text("0.5\n1.0\n")
            return out

        def fake_separate(track: TrackPaths, vocal_out_dir: Path) -> Path:
            captured.append(track.track_id)
            return track.audio_path     # pretend the raw audio is the vocal

        def fake_evaluate(vocal_wav, beats_file, png_out=None) -> _FakeScore:
            return _FakeScore(f_measure=0.91, cmlt=0.5, amlt=0.9, rt_factor=0.4)

        results = run_pipeline(
            audio_root=root / "audio",
            annotation_root=root / "ann",
            work_dir=root / "work",
            genres=("pop",),
            per_genre_limit=2,
            convert_fn=fake_convert,
            separate_fn=fake_separate,
            evaluate_fn=fake_evaluate,
            verbose=False,
        )
        # classical was excluded, pop capped to 2, both ok.
        assert [r.track_id for r in results] == ["pop.00000", "pop.00001"]
        assert captured == ["pop.00000", "pop.00001"]
        assert all(r.error is None for r in results)
        assert all(r.score.f_measure == 0.91 for r in results)


def test_run_pipeline_records_per_track_errors_without_aborting():
    """A failure on one track must not stop the pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = ["pop.00001", "pop.00002", "pop.00003"]
        _make_track_tree(root, audio=audio, ann=audio)

        def boom_on_middle(track: TrackPaths, vocal_out_dir: Path) -> Path:
            if track.track_id == "pop.00002":
                raise RuntimeError("simulated Demucs OOM")
            return track.audio_path

        results = run_pipeline(
            audio_root=root / "audio",
            annotation_root=root / "ann",
            work_dir=root / "work",
            genres=("pop",),
            per_genre_limit=3,
            convert_fn=lambda t, d: (d / f"{t.track_id}.beats"),  # no write needed
            separate_fn=boom_on_middle,
            evaluate_fn=lambda v, b, p=None: _FakeScore(0.5, 0.5, 0.5, 0.5),
            verbose=False,
        )
        errored = [r for r in results if r.error]
        ok = [r for r in results if r.score]
        assert [r.track_id for r in errored] == ["pop.00002"]
        assert "Demucs OOM" in errored[0].error
        assert len(ok) == 2
