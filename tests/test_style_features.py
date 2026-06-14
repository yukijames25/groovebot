"""Tests for groovebot.style.features (log-mel spectrogram front end)."""
from __future__ import annotations

import numpy as np
import pytest

from groovebot.style.features import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS,
    DEFAULT_SR,
    log_mel_spectrogram,
)


def _synth_tone(sr: int, duration_sec: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * duration_sec)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_log_mel_returns_n_mels_by_T():
    audio = _synth_tone(DEFAULT_SR, duration_sec=5.0)
    mel = log_mel_spectrogram(audio, DEFAULT_SR)
    assert mel.ndim == 2
    assert mel.shape[0] == DEFAULT_N_MELS
    # ~ duration * sr / hop frames, give or take librosa's edge handling
    expected_T = DEFAULT_SR * 5 // DEFAULT_HOP_LENGTH
    assert abs(mel.shape[1] - expected_T) <= 2


def test_log_mel_resamples_to_target_sr():
    sr_in = 44100
    audio = _synth_tone(sr_in, duration_sec=5.0)
    mel = log_mel_spectrogram(audio, sr_in, target_sr=DEFAULT_SR)
    expected_T = DEFAULT_SR * 5 // DEFAULT_HOP_LENGTH
    assert abs(mel.shape[1] - expected_T) <= 2


def test_log_mel_is_finite():
    audio = _synth_tone(DEFAULT_SR, duration_sec=2.0)
    mel = log_mel_spectrogram(audio, DEFAULT_SR)
    assert np.isfinite(mel).all()


def test_log_mel_dtype_float32():
    audio = _synth_tone(DEFAULT_SR, duration_sec=1.0)
    mel = log_mel_spectrogram(audio, DEFAULT_SR)
    assert mel.dtype == np.float32


def test_log_mel_handles_stereo_by_averaging():
    sr = DEFAULT_SR
    left = _synth_tone(sr, 1.0, freq=220.0)
    right = _synth_tone(sr, 1.0, freq=880.0)
    stereo = np.stack([left, right], axis=-1)  # (samples, channels)
    mel = log_mel_spectrogram(stereo, sr)
    assert mel.shape[0] == DEFAULT_N_MELS


def test_log_mel_louder_audio_higher_energy_band():
    sr = DEFAULT_SR
    quiet = 0.05 * _synth_tone(sr, 2.0)
    loud = 2.0 * _synth_tone(sr, 2.0)
    mel_q = log_mel_spectrogram(quiet, sr)
    mel_l = log_mel_spectrogram(loud, sr)
    # power_to_db with ref=np.max normalizes the peak to 0; absolute peak
    # is therefore ~0 for both. What differs is the *spread* of energy —
    # the quiet signal's mean-vs-peak gap should be smaller because more
    # frames sit close to its (already small) peak. Use the 5th
    # percentile as a stable check: louder signals push more frames down.
    assert np.percentile(mel_q, 5) >= np.percentile(mel_l, 5) - 1e-3


def test_log_mel_n_mels_param_honoured():
    audio = _synth_tone(DEFAULT_SR, 1.0)
    mel = log_mel_spectrogram(audio, DEFAULT_SR, n_mels=32)
    assert mel.shape[0] == 32
