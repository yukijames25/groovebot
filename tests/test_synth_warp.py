"""tools.synth_warp — time-stretch rate semantics + file output round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from tools.synth_warp import (
    DEFAULT_RATES,
    synth_one,
    synth_warp_audio,
    warped_beat_times,
)


def _sine(freq: float = 440.0, dur: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_rate_one_preserves_duration():
    sr = 22050
    y = _sine(dur=1.0, sr=sr)
    y_warp = synth_warp_audio(y, sr, 1.0)
    # Phase-vocoder framing adds a little slop, but not much.
    assert abs(len(y_warp) - len(y)) <= int(sr * 0.05)


def test_rate_half_roughly_doubles_duration():
    sr = 22050
    y = _sine(dur=1.0, sr=sr)
    y_warp = synth_warp_audio(y, sr, 0.5)
    assert 1.8 * sr < len(y_warp) < 2.2 * sr


def test_rate_double_roughly_halves_duration():
    sr = 22050
    y = _sine(dur=1.0, sr=sr)
    y_warp = synth_warp_audio(y, sr, 2.0)
    assert 0.4 * sr < len(y_warp) < 0.6 * sr


def test_non_positive_rate_raises():
    with pytest.raises(ValueError):
        synth_warp_audio(_sine(), 22050, 0.0)
    with pytest.raises(ValueError):
        synth_warp_audio(_sine(), 22050, -1.0)


def test_warped_beat_times_is_inverse_rate_scaling():
    beats = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    assert np.allclose(warped_beat_times(beats, 2.0),
                       [0.0, 0.25, 0.5, 0.75])
    assert np.allclose(warped_beat_times(beats, 0.5),
                       [0.0, 1.0, 2.0, 3.0])


def test_default_rates_bracket_one():
    assert min(DEFAULT_RATES) < 1.0 < max(DEFAULT_RATES)


def test_synth_one_writes_wav_and_beats(tmp_path):
    sr = 22050
    y = _sine(dur=1.0, sr=sr)
    ref_beats = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    wav_path, beats_path = synth_one(y, sr, ref_beats, 0.95, tmp_path, "song")
    assert wav_path.exists()
    assert beats_path.exists()
    assert wav_path.name == "song_r0.95.wav"
    loaded = np.loadtxt(beats_path)
    assert np.allclose(loaded, ref_beats / 0.95, atol=1e-5)
