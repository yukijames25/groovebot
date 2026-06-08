"""experiments.run_m0p_align — pipeline smoke + aggregation tests."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.run_m0p_align import (
    aggregate,
    find_pairs,
    run_pipeline,
    save_csv,
)


def _make_song(dir_path: Path, sr: int = 22050,
               n_beats: int = 24, beat_period: float = 0.5) -> tuple[Path, Path]:
    """Write a (audio, beats) pair to `dir_path`. Long enough (>=10s) for
    mir_eval.trim_beats (which drops beats before 5s) to leave us with a
    meaningful number of beats to score."""
    freqs = (261.63, 329.63, 392.00, 523.25)   # C major arpeggio (C4-E4-G4-C5)
    duration = n_beats * beat_period
    n = int(duration * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for i in range(n_beats):
        f = freqs[i % len(freqs)]
        t0 = i * beat_period
        seg = (t >= t0) & (t < t0 + beat_period)
        local_t = t[seg] - t0
        env = np.exp(-3 * local_t / beat_period).astype(np.float32)
        out[seg] = (np.sin(2 * np.pi * f * local_t) * env).astype(np.float32)
    wav = dir_path / "song.wav"
    sf.write(str(wav), out, sr)
    beats = dir_path / "song.beats"
    np.savetxt(str(beats), np.arange(n_beats) * beat_period, fmt="%.6f")
    return wav, beats


def test_find_pairs_discovers_wav_beats(tmp_path):
    wav, beats = _make_song(tmp_path)
    pairs = find_pairs(tmp_path)
    assert pairs == [(wav, beats)]


def test_find_pairs_ignores_wav_without_beats(tmp_path):
    sr = 22050
    sf.write(str(tmp_path / "orphan.wav"),
             np.zeros(sr, dtype=np.float32), sr)
    assert find_pairs(tmp_path) == []


def test_aggregate_empty_returns_zeroed_overall():
    per_rate, overall = aggregate([])
    assert per_rate == []
    assert overall["n"] == 0
    assert overall["f_mean"] == 0.0


def test_aggregate_groups_by_rate():
    rows = [
        {"rate": 0.95, "f_measure": 0.8, "cmlt": 0.6, "amlt": 0.7, "rt_factor": 0.5},
        {"rate": 0.95, "f_measure": 0.6, "cmlt": 0.5, "amlt": 0.6, "rt_factor": 0.6},
        {"rate": 1.05, "f_measure": 0.9, "cmlt": 0.8, "amlt": 0.85, "rt_factor": 0.4},
    ]
    per_rate, overall = aggregate(rows)
    rate_to_n = {p["rate"]: p["n"] for p in per_rate}
    assert rate_to_n == {0.95: 2, 1.05: 1}
    assert abs(overall["f_mean"] - np.mean([0.8, 0.6, 0.9])) < 1e-6


def test_save_csv_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "out.csv"
    save_csv(rows, p)
    with open(p, "r", encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    assert [r["a"] for r in loaded] == ["1", "2"]
    assert [r["b"] for r in loaded] == ["x", "y"]


def test_pipeline_smoke_identity_rate(tmp_path):
    """End-to-end at rate=1.0: pipeline produces CSVs, a PNG, and a usable
    F-measure. The threshold is intentionally loose — this is a wiring test,
    not a SOTA check."""
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    _make_song(refs_dir)
    out_dir = tmp_path / "work"
    rows, _per_rate, overall = run_pipeline(
        root=refs_dir, out_dir=out_dir,
        feature_kind="chroma", rates=[1.0],
        sample_rate=22050, hop_length=512,
        make_png=True, verbose=False,
    )
    assert len(rows) == 1
    assert (out_dir / "m0p_per_track.csv").exists()
    assert (out_dir / "m0p_per_rate.csv").exists()
    assert (out_dir / "m0p_overall.csv").exists()
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1
    # At rate=1.0 chroma_cqt on a clean C-major arpeggio should give DTW
    # an unambiguous identity warp. F=0.5 is a wide guardrail; the actual
    # number is logged in the CSV.
    assert overall["f_mean"] >= 0.5
