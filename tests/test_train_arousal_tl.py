"""Tests for experiments.train_arousal_tl — stub end-to-end + metric math.

Real DEAM never enters CI: tests stay on the `--synthetic-stub` path
which generates a target-conditional fake embedding dataset, plus
synthesised audio fixtures for the heuristic helper.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments.train_arousal_tl import (
    compute_heuristic_arousal_unit, main, pearson_r, regression_metrics,
)
from groovebot.style.deam import DeamRecord


# ----------------------------------------------------------- metric math

def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["n"] == 5


def test_regression_metrics_mean_predictor_is_zero_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.full_like(y, y.mean())
    m = regression_metrics(y, pred)
    assert m["r2"] == pytest.approx(0.0)
    assert m["pearson_r"] == 0.0  # constant prediction collapses pearson


def test_regression_metrics_handles_empty():
    m = regression_metrics(np.zeros(0), np.zeros(0))
    assert m["n"] == 0


def test_pearson_r_perfect_anti_correlation():
    a = np.arange(10, dtype=float)
    b = -a
    assert pearson_r(a, b) == pytest.approx(-1.0)


def test_pearson_r_constant_collapses_to_zero():
    a = np.ones(10)
    b = np.arange(10, dtype=float)
    assert pearson_r(a, b) == 0.0


# ------------------------------------------------------------ stub run

def test_stub_main_writes_report(tmp_path: Path):
    out = tmp_path / "out"
    rc = main([
        "--synthetic-stub",
        "--out-dir", str(out),
        "--epochs", "3",
        "--n-stub-songs", "120",
        "--early-stopping-patience", "0",
    ])
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["is_stub"] is True
    assert "arousal" in report["targets"]
    assert "valence" in report["targets"]
    assert report["heuristic_arousal_vs_deam"]["n"] > 0
    assert (out / "style_head_arousal.pt").exists()


def test_stub_main_signal_driven_above_chance(tmp_path: Path):
    """With 3 epochs the head should still cross R^2 > 0.3 on the
    target-conditional stub (synthetic_records writes a linear signal
    into the embedding)."""
    out = tmp_path / "out"
    main([
        "--synthetic-stub",
        "--out-dir", str(out),
        "--epochs", "5",
        "--n-stub-songs", "200",
        "--early-stopping-patience", "0",
    ])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # Both heads should learn the signal cleanly under the stub.
    assert report["targets"]["arousal"]["test"]["r2"] > 0.5
    assert report["targets"]["valence"]["test"]["r2"] > 0.5


# ----------------------------------- heuristic correlation against a fixture

def _write_sine(path: Path, sr: int, duration_sec: float, amp: float) -> None:
    t = np.arange(int(sr * duration_sec)) / sr
    sig = (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), sig, sr)


def _write_clicks(path: Path, sr: int, duration_sec: float, bpm: float) -> None:
    n = int(sr * duration_sec)
    sig = np.zeros(n, dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    click_len = max(1, sr // 200)
    for start in range(0, n, period):
        end = min(n, start + click_len)
        sig[start:end] = 1.0
    sf.write(str(path), sig, sr)


def test_compute_heuristic_arousal_unit_assigns_dense_higher(tmp_path: Path):
    """The helper must reflect estimate_arousal's known property:
    dense clicks score higher than a quiet sine. This indirectly
    verifies the pipeline: it loads + center-crops + scores."""
    sr = 22050
    quiet = tmp_path / "quiet.wav"
    dense = tmp_path / "dense.wav"
    _write_sine(quiet, sr, 6.0, amp=0.005)
    _write_clicks(dense, sr, 6.0, bpm=180.0)
    records = [
        DeamRecord(audio_path=quiet, song_id=1, arousal=2.0, valence=5.0),
        DeamRecord(audio_path=dense, song_id=2, arousal=8.0, valence=5.0),
    ]
    out = compute_heuristic_arousal_unit(records, window_sec=5.0, verbose=False)
    assert out[2] > out[1]
    assert 0.0 <= out[1] <= 1.0
    assert 0.0 <= out[2] <= 1.0
