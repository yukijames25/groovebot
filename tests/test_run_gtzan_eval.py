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
    PipelineReport,
    STAGE_CONVERT,
    STAGE_EVALUATE,
    STAGE_PAIR,
    STAGE_SEPARATE,
    TrackPaths,
    TrackResult,
    VOCAL_GENRES,
    _annotation_candidates_for,
    _annotation_for,
    _genre_from_filename,
    _select_with_report,
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
        assert errored[0].stage == STAGE_SEPARATE
        assert len(ok) == 2


# --------------------------------------------------------------------------- #
# Annotation pairing — the bug that caused the n_tracks=0 incident.
# --------------------------------------------------------------------------- #
def test_annotation_candidates_cover_known_layouts():
    cands = _annotation_candidates_for("blues.00000")
    assert "blues.00000" in cands              # original GTZAN naming
    assert "gtzan_blues_00000" in cands        # TempoBeatDownbeat/gtzan_tempo_beat
    assert "blues_00000" in cands              # underscore-only variant


def test_annotation_candidates_for_unstructured_stem():
    cands = _annotation_candidates_for("just_a_name")
    # No `.<idx>` to split — just the stem itself.
    assert cands == ["just_a_name"]


def test_annotation_for_finds_gtzan_tempo_beat_form(tmp_path: Path):
    """Audio `blues.00000.wav` <-> annotation `gtzan_blues_00000.beats`.

    Regression: before 2026-06-07 this returned None, dropping every GTZAN
    track silently."""
    ann_root = tmp_path / "gtzan_tempo_beat" / "beats"
    ann_root.mkdir(parents=True)
    (ann_root / "gtzan_blues_00000.beats").write_text("0.5\n1.0\n")
    found = _annotation_for("blues.00000", tmp_path)
    assert found is not None
    assert found.name == "gtzan_blues_00000.beats"


def test_annotation_for_still_finds_original_naming(tmp_path: Path):
    (tmp_path / "blues.00000.beats").write_text("0.5\n")
    assert _annotation_for("blues.00000", tmp_path).name == "blues.00000.beats"


def test_annotation_for_returns_none_when_no_match(tmp_path: Path):
    assert _annotation_for("blues.00000", tmp_path) is None


def test_select_tracks_finds_tempobeatdownbeat_layout(tmp_path: Path):
    """End-to-end: gtzan_mini-style audio + TempoBeatDownbeat-style ann."""
    audio_root = tmp_path / "gtzan_mini" / "genres"
    ann_root = tmp_path / "gtzan_tempo_beat" / "beats"
    (audio_root / "blues").mkdir(parents=True)
    (audio_root / "pop").mkdir(parents=True)
    ann_root.mkdir(parents=True)
    for genre, idx in [("blues", "00000"), ("blues", "00001"),
                       ("pop", "00010")]:
        (audio_root / genre / f"{genre}.{idx}.wav").write_bytes(b"")
        (ann_root / f"gtzan_{genre}_{idx}.beats").write_text("0.5\n1.0\n")

    picked = select_tracks(tmp_path / "gtzan_mini", tmp_path / "gtzan_tempo_beat",
                           genres=("blues", "pop"), per_genre_limit=5)
    assert [(t.genre, t.track_id) for t in picked] == [
        ("blues", "blues.00000"),
        ("blues", "blues.00001"),
        ("pop",   "pop.00010"),
    ]
    # Each TrackPaths.annotation_path points at the TempoBeatDownbeat file.
    for t in picked:
        assert t.annotation_path.name.startswith("gtzan_")


# --------------------------------------------------------------------------- #
# PipelineReport — stage counts + drops list.
# --------------------------------------------------------------------------- #
def test_selection_report_counts_per_stage(tmp_path: Path):
    audio_root = tmp_path / "audio"
    ann_root = tmp_path / "ann"
    audio_root.mkdir()
    ann_root.mkdir()
    # 4 pop audios, but only 2 have annotations; 2 classical (filtered out).
    for stem in ["pop.00000", "pop.00001", "pop.00002", "pop.00003",
                 "classical.00000", "classical.00001"]:
        (audio_root / f"{stem}.wav").write_bytes(b"")
    for stem in ["pop.00000", "pop.00001"]:
        (ann_root / f"{stem}.beats").write_text("0.5\n")

    sel = _select_with_report(audio_root, ann_root,
                              genres=("pop",), per_genre_limit=10)
    assert sel.n_audio_found == 6
    assert sel.n_after_genre_filter == 4
    assert sel.n_with_annotation == 2
    assert len(sel.selected) == 2
    # Two pop tracks without annotation should appear as drops.
    drop_ids = sorted([d[0] for d in sel.drops])
    assert drop_ids == ["pop.00002", "pop.00003"]
    assert all(d[1] == STAGE_PAIR for d in sel.drops)


def test_run_pipeline_returns_pipeline_report_with_full_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 3 pop with annotations + 1 with NO annotation -> drop at pair stage.
        # 1 rock -> excluded by genre filter.
        audio = ["pop.00000", "pop.00001", "pop.00002", "pop.00099",
                 "rock.00000"]
        ann = ["pop.00000", "pop.00001", "pop.00002"]
        _make_track_tree(root, audio=audio, ann=ann)

        report = run_pipeline(
            audio_root=root / "audio",
            annotation_root=root / "ann",
            work_dir=root / "work",
            genres=("pop",),
            per_genre_limit=10,
            convert_fn=lambda t, d: (d / f"{t.track_id}.beats"),
            separate_fn=lambda t, d: t.audio_path,
            evaluate_fn=lambda v, b, p=None: _FakeScore(0.8, 0.7, 0.9, 0.5),
            verbose=False,
        )
        assert isinstance(report, PipelineReport)
        assert report.n_audio_found == 5
        assert report.n_after_genre_filter == 4
        assert report.n_with_annotation == 3
        assert report.n_converted == 3
        assert report.n_separated == 3
        assert report.n_evaluated == 3
        # pop.00099 is the one pair-stage drop.
        assert ("pop.00099", STAGE_PAIR) in [(d[0], d[1]) for d in report.drops]


def test_pipeline_report_is_list_like_for_back_compat():
    """Code that does `for r in results` and `len(results)` must keep working."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = ["pop.00000", "pop.00001"]
        _make_track_tree(root, audio=audio, ann=audio)
        report = run_pipeline(
            audio_root=root / "audio",
            annotation_root=root / "ann",
            work_dir=root / "work",
            genres=("pop",), per_genre_limit=10,
            convert_fn=lambda t, d: (d / f"{t.track_id}.beats"),
            separate_fn=lambda t, d: t.audio_path,
            evaluate_fn=lambda v, b, p=None: _FakeScore(1.0, 1.0, 1.0, 0.3),
            verbose=False,
        )
        assert len(report) == 2
        ids = [r.track_id for r in report]
        assert ids == ["pop.00000", "pop.00001"]
        assert report[0].track_id == "pop.00000"


def test_pipeline_report_records_each_stage_drop_with_correct_label():
    """One drop per stage — convert / separate / evaluate — should be tagged
    with the matching STAGE_* constant."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio = ann = ["pop.00000", "pop.00001", "pop.00002"]
        _make_track_tree(root, audio=audio, ann=ann)

        def conv(track, beats_out_dir):
            if track.track_id == "pop.00000":
                raise RuntimeError("convert exploded")
            return beats_out_dir / f"{track.track_id}.beats"

        def sep(track, vocal_out_dir):
            if track.track_id == "pop.00001":
                raise RuntimeError("separate exploded")
            return track.audio_path

        def ev(v, b, p=None):
            raise RuntimeError("evaluate exploded")    # always fails

        report = run_pipeline(
            audio_root=root / "audio",
            annotation_root=root / "ann",
            work_dir=root / "work",
            genres=("pop",), per_genre_limit=10,
            convert_fn=conv, separate_fn=sep, evaluate_fn=ev,
            verbose=False,
        )
        stages_by_track = {d[0]: d[1] for d in report.drops}
        assert stages_by_track == {
            "pop.00000": STAGE_CONVERT,
            "pop.00001": STAGE_SEPARATE,
            "pop.00002": STAGE_EVALUATE,
        }
        assert report.n_converted == 2     # pop.00001 and pop.00002 converted
        assert report.n_separated == 1     # only pop.00002
        assert report.n_evaluated == 0
        # Each TrackResult also carries the stage tag for CSV.
        for r in report.results:
            assert r.stage in (STAGE_CONVERT, STAGE_SEPARATE, STAGE_EVALUATE)


def test_pipeline_report_summary_mentions_zero_with_annotation(tmp_path: Path):
    """The summary string is what the notebook prints; verify the key lines."""
    audio_root = tmp_path / "audio"
    ann_root = tmp_path / "ann"
    audio_root.mkdir()
    ann_root.mkdir()
    (audio_root / "pop.00000.wav").write_bytes(b"")   # no annotation match
    report = run_pipeline(
        audio_root=audio_root, annotation_root=ann_root,
        work_dir=tmp_path / "work",
        genres=("pop",), per_genre_limit=10,
        convert_fn=lambda t, d: d, separate_fn=lambda t, d: d,
        evaluate_fn=lambda v, b, p=None: _FakeScore(1, 1, 1, 1),
        verbose=False,
    )
    s = report.summary()
    assert "audio files found     : 1" in s
    assert "with matched annotation: 0" in s
    assert "evaluation succeeded  : 0" in s
    assert "pop.00000" in s
    assert "pair_annotation" in s


def test_save_csv_includes_stage_column():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "out.csv"
        save_csv([
            _ok("pop", "pop.00001", 1.0, 1.0, 1.0, 0.5),
            TrackResult(genre="pop", track_id="pop.00002",
                        error="boom", stage=STAGE_SEPARATE),
        ], p)
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["stage"] == ""
        assert rows[1]["stage"] == STAGE_SEPARATE


# --------------------------------------------------------------------------- #
# Demucs output-path discovery (prep_dataset._find_vocal_output)
# --------------------------------------------------------------------------- #
def test_find_vocal_output_canonical_path(tmp_path: Path):
    from tools.prep_dataset import _find_vocal_output
    p = tmp_path / "htdemucs" / "blues.00000" / "vocals.wav"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    found = _find_vocal_output(tmp_path, "htdemucs", "blues.00000")
    assert found == p


def test_find_vocal_output_falls_back_to_underscore_variant(tmp_path: Path):
    """Some Demucs versions normalise dots to underscores in the stem dir."""
    from tools.prep_dataset import _find_vocal_output
    # Demucs wrote to <root>/htdemucs/blues_00000/vocals.wav (underscore!)
    p = tmp_path / "htdemucs" / "blues_00000" / "vocals.wav"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    found = _find_vocal_output(tmp_path, "htdemucs", "blues.00000")
    assert found == p


def test_find_vocal_output_raises_when_truly_missing(tmp_path: Path):
    from tools.prep_dataset import _find_vocal_output
    (tmp_path / "htdemucs").mkdir()
    with pytest.raises(RuntimeError, match="vocals.wav was not found"):
        _find_vocal_output(tmp_path, "htdemucs", "blues.00000")
