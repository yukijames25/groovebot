"""experiments.run_m0p_t2_damp — end-to-end DAMP route on synthetic data."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments.run_m0p_t2_damp import (
    aggregate,
    beats_from_backing,
    chroma_from_backing,
    melody_from_consensus,
    run_arrangement,
    run_pipeline,
    save_csv,
)
from groovebot.align.dtw_align import OfflineDTWAligner
from tools.ingest_damp import discover_arrangements


# --------------------------------------------------------------------------- #
# Synthetic stand-in
# --------------------------------------------------------------------------- #
SR = 22050
BPM = 150           # 0.4 s per beat
N_BEATS = 24        # 9.6 s, >5 s after mir_eval.trim_beats
PERIOD = 60.0 / BPM
DUR = N_BEATS * PERIOD


def _synth_backing(seed: int = 0) -> np.ndarray:
    """Backing = click track at every beat + arpeggio that varies per beat.

    The clicks make librosa.beat lock on cleanly; the arpeggio gives
    backing chroma the per-beat structure DTW needs."""
    n = int(DUR * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float32)
    click_len = int(0.020 * SR)
    click_env = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)
    click_tone = (np.sin(2 * np.pi * 1000.0 * np.arange(click_len) / SR)
                  .astype(np.float32) * click_env * 0.5)
    for i in range(N_BEATS):
        idx = int(i * PERIOD * SR)
        end = min(idx + click_len, n)
        out[idx:end] += click_tone[: end - idx]
    freqs = (261.63, 329.63, 392.00, 523.25)
    ramp = 0.05
    for i in range(N_BEATS):
        f = freqs[i % len(freqs)]
        beat_start = i * PERIOD
        seg = (t >= beat_start) & (t < beat_start + PERIOD)
        local = t[seg] - beat_start
        env = np.minimum(local / ramp, 1.0) * \
              np.minimum((PERIOD - local) / ramp, 1.0)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        out[seg] += 0.3 * (np.sin(2 * np.pi * f * local) * env).astype(np.float32)
    rng = np.random.default_rng(seed)
    out += 0.001 * rng.standard_normal(n).astype(np.float32)
    return out


def _synth_vocal(seed: int = 0, noise: float = 0.005) -> np.ndarray:
    """Just the arpeggio (no clicks) + a little noise. Each call returns a
    distinct rendition by varying the noise seed."""
    n = int(DUR * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float32)
    freqs = (261.63, 329.63, 392.00, 523.25)
    ramp = 0.05
    for i in range(N_BEATS):
        f = freqs[i % len(freqs)]
        beat_start = i * PERIOD
        seg = (t >= beat_start) & (t < beat_start + PERIOD)
        local = t[seg] - beat_start
        env = np.minimum(local / ramp, 1.0) * \
              np.minimum((PERIOD - local) / ramp, 1.0)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        out[seg] = (np.sin(2 * np.pi * f * local) * env).astype(np.float32)
    rng = np.random.default_rng(seed)
    out += noise * rng.standard_normal(n).astype(np.float32)
    return out


def _make_arrangement(d: Path, n_renditions: int = 3) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "backing.wav"), _synth_backing(seed=0), SR)
    for i in range(n_renditions):
        rid = f"singer{i:02d}"
        sf.write(str(d / f"vocal_{rid}.wav"), _synth_vocal(seed=i + 1), SR)
    return d


# --------------------------------------------------------------------------- #
# Unit-ish tests
# --------------------------------------------------------------------------- #
def test_beats_from_backing_finds_some_beats():
    backing = _synth_backing()
    beats = beats_from_backing(backing, SR)
    # We don't pin a count — librosa.beat's grouping may differ from our
    # click cadence. We just want enough beats to score against.
    assert beats.ndim == 1
    assert beats.size >= 8
    assert beats.min() >= 0
    assert beats.max() <= DUR


def test_chroma_from_backing_has_12_rows():
    backing = _synth_backing()
    chroma = chroma_from_backing(backing, SR, hop_length=512)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] > 0


def test_melody_from_consensus_returns_12_T():
    # Median across three constant F0 contours -> one-hot pitch class.
    f0s = [
        np.full(50, 440.0),    # A
        np.full(50, 442.0),    # A (rounds to same MIDI)
        np.full(50, 441.0),    # A
    ]
    chroma = melody_from_consensus(f0s)
    assert chroma.shape == (12, 50)
    assert chroma[9].sum() == 50.0    # all frames -> pitch class A (=9)


def test_melody_from_consensus_empty_returns_zero_matrix():
    chroma = melody_from_consensus([])
    assert chroma.shape == (12, 0)


def test_aggregate_groups_by_kind_and_arrangement():
    rows = [
        {"arrangement_id": "song_a", "feature_kind": "chroma",
         "f_measure": 0.8, "cmlt": 0.6, "amlt": 0.7, "rt_factor": 0.1},
        {"arrangement_id": "song_a", "feature_kind": "pitch",
         "f_measure": 0.5, "cmlt": 0.4, "amlt": 0.5, "rt_factor": 0.2},
        {"arrangement_id": "song_b", "feature_kind": "chroma",
         "f_measure": 0.9, "cmlt": 0.7, "amlt": 0.8, "rt_factor": 0.1},
    ]
    per_kind, per_arr, overall = aggregate(rows)
    kn = {p["feature_kind"]: p["n"] for p in per_kind}
    an = {p["arrangement_id"]: p["n"] for p in per_arr}
    assert kn == {"chroma": 2, "pitch": 1}
    assert an == {"song_a": 2, "song_b": 1}
    assert overall["n"] == 3


def test_aggregate_empty():
    per_kind, per_arr, overall = aggregate([])
    assert per_kind == []
    assert per_arr == []
    assert overall["n"] == 0


def test_save_csv_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "out.csv"
    save_csv(rows, p)
    with open(p, "r", encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    assert [r["a"] for r in loaded] == ["1", "2"]


# --------------------------------------------------------------------------- #
# End-to-end on synthetic data
# --------------------------------------------------------------------------- #
def test_run_arrangement_designated_skips_reference_rendition(tmp_path):
    arr_dir = _make_arrangement(tmp_path / "synth", n_renditions=3)
    arrangements = discover_arrangements(tmp_path)
    assert len(arrangements) == 1
    aligner = OfflineDTWAligner(sample_rate=SR, hop_length=512)
    rows = run_arrangement(
        arrangements[0], tmp_path / "work", aligner,
        melody_mode="designated", designated="singer00",
        make_png=False,
    )
    # 2 query renditions (singer01, singer02) x 2 paths = 4 rows.
    assert len(rows) == 4
    query_ids = {r["track"] for r in rows}
    assert query_ids == {"singer01", "singer02"}
    kinds = sorted(r["feature_kind"] for r in rows)
    assert kinds == ["chroma", "chroma", "pitch", "pitch"]


def test_run_arrangement_consensus_scores_every_rendition(tmp_path):
    arr_dir = _make_arrangement(tmp_path / "synth", n_renditions=3)
    arrangements = discover_arrangements(tmp_path)
    aligner = OfflineDTWAligner(sample_rate=SR, hop_length=512)
    rows = run_arrangement(
        arrangements[0], tmp_path / "work", aligner,
        melody_mode="consensus", make_png=False,
    )
    # 3 query renditions x 2 paths = 6 rows.
    assert len(rows) == 6
    assert {r["track"] for r in rows} == {"singer00", "singer01", "singer02"}


def test_run_arrangement_unknown_designated_raises(tmp_path):
    _make_arrangement(tmp_path / "synth", n_renditions=2)
    arrangements = discover_arrangements(tmp_path)
    aligner = OfflineDTWAligner(sample_rate=SR, hop_length=512)
    with pytest.raises(ValueError):
        run_arrangement(
            arrangements[0], tmp_path / "work", aligner,
            melody_mode="designated", designated="ghost",
            make_png=False,
        )


def test_run_pipeline_writes_all_csvs_and_pngs(tmp_path):
    _make_arrangement(tmp_path / "synth", n_renditions=3)
    out_dir = tmp_path / "work"
    rows, per_kind, per_arr, overall = run_pipeline(
        root=tmp_path,
        out_dir=out_dir,
        sample_rate=SR,
        hop_length=512,
        melody_mode="designated",
        designated="singer00",
        make_png=True,
        verbose=False,
    )
    # 2 queries x 2 paths
    assert len(rows) == 4
    for name in ("m0p_t2_damp_per_path.csv",
                 "m0p_t2_damp_per_kind.csv",
                 "m0p_t2_damp_per_arrangement.csv",
                 "m0p_t2_damp_overall.csv"):
        assert (out_dir / name).exists()
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 4
    # Both paths should produce some signal — wide guardrail, this is
    # a wiring test, not a benchmark.
    by_kind = {p["feature_kind"]: p for p in per_kind}
    assert by_kind["chroma"]["f_mean"] >= 0.2, by_kind
    assert by_kind["pitch"]["f_mean"]  >= 0.2, by_kind
