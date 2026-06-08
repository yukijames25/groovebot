"""groovebot.align.features — pyin_f0 / consensus_f0 / f0_to_pitch_chroma."""
from __future__ import annotations

import numpy as np

from groovebot.align.features import (
    consensus_f0,
    f0_to_pitch_chroma,
    pyin_f0,
)


def _sine(freq: float = 440.0, dur: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_pyin_f0_tracks_a4_within_a_quarter_tone():
    sr = 22050
    f0 = pyin_f0(_sine(440.0, dur=1.0, sr=sr), sr, hop_length=512)
    voiced = f0[np.isfinite(f0)]
    assert voiced.size > 0
    # A4 = 440 Hz; pyin should land within a quarter-tone of it.
    assert abs(float(np.median(voiced)) - 440.0) < 12.0


def test_f0_to_pitch_chroma_one_hot_per_voiced_frame():
    f0 = np.array([440.0, 440.0, np.nan, 261.63], dtype=float)   # A, A, _, C
    chroma = f0_to_pitch_chroma(f0)
    assert chroma.shape == (12, 4)
    assert chroma[9, 0] == 1.0    # A
    assert chroma[9, 1] == 1.0
    assert chroma[:, 2].sum() == 0.0  # unvoiced
    assert chroma[0, 3] == 1.0    # C


def test_consensus_f0_frame_wise_median_ignores_nan():
    a = np.array([440.0, 440.0, np.nan, 261.63])
    b = np.array([440.0, np.nan, np.nan, 261.63])
    c = np.array([442.0, 442.0, 442.0, 261.63])
    med = consensus_f0([a, b, c])
    assert med.shape == (4,)
    # Frame 0: median(440, 440, 442) = 440
    assert med[0] == 440.0
    # Frame 1: median(440, NaN, 442) = 441
    assert med[1] == 441.0
    # Frame 2: median(NaN, NaN, 442) = 442
    assert med[2] == 442.0
    # Frame 3: all 261.63 -> 261.63
    assert abs(med[3] - 261.63) < 1e-6


def test_consensus_f0_handles_different_lengths():
    a = np.array([440.0, 440.0, 440.0])
    b = np.array([442.0, 442.0])      # shorter; frame 2 padded with NaN
    med = consensus_f0([a, b])
    assert med.shape == (3,)
    assert med[0] == 441.0
    assert med[1] == 441.0
    assert med[2] == 440.0   # only `a` contributes here


def test_consensus_f0_all_nan_frame_is_nan():
    a = np.array([np.nan, 440.0])
    b = np.array([np.nan, 442.0])
    med = consensus_f0([a, b])
    assert np.isnan(med[0])
    assert med[1] == 441.0


def test_consensus_f0_empty_input_returns_empty():
    out = consensus_f0([])
    assert out.shape == (0,)
