"""groovebot.align.dtw_align — identity & shape tests for offline DTW."""
from __future__ import annotations

import numpy as np
import pytest

from groovebot.align.dtw_align import OfflineDTWAligner, map_reference_beats


def _chroma_like_song(T: int = 120, period: int = 10) -> np.ndarray:
    """A (12, T) one-hot-per-frame feature sequence with the active pitch
    class cycling every `period` frames. Good fixture for DTW because the
    sequence is unique enough to avoid ambiguous matches."""
    feats = np.zeros((12, T), dtype=np.float32)
    for i in range(T):
        pc = (i // period) % 12
        feats[pc, i] = 1.0
    return feats


def test_identity_warp_path_is_near_diagonal():
    feats = _chroma_like_song(T=120, period=10)
    aligner = OfflineDTWAligner(sample_rate=22050, hop_length=512)
    wp = aligner.align(feats, feats)
    # With identical inputs and zero-cost diagonal moves, DTW should stay on
    # the diagonal. Allow ±1 for edge effects.
    diff = np.abs(wp[:, 0].astype(int) - wp[:, 1].astype(int))
    assert int(diff.max()) <= 1


def test_identity_recovers_beats_within_one_frame():
    sr, hop = 22050, 512
    feats = _chroma_like_song(T=120, period=10)
    aligner = OfflineDTWAligner(sample_rate=sr, hop_length=hop)
    wp = aligner.align(feats, feats)
    # Beats at every `period` frames, in seconds.
    ref_beats = np.arange(12) * (10 * hop / sr)
    recovered = aligner.map_reference_beats(wp, ref_beats)
    assert recovered.size == ref_beats.size
    tol = hop / sr   # one frame, in seconds
    assert float(np.max(np.abs(recovered - ref_beats))) <= tol


def test_empty_inputs_return_empty():
    out = map_reference_beats(np.empty((0, 2), dtype=int), np.array([0.1, 0.2]),
                              hop_length=512, sample_rate=22050)
    assert out.size == 0
    out = map_reference_beats(np.array([[0, 0]]), np.empty(0),
                              hop_length=512, sample_rate=22050)
    assert out.size == 0


def test_beats_outside_path_range_are_dropped():
    sr, hop = 22050, 512
    # Path covers ref frames 0..50 only.
    wp = np.column_stack([np.arange(51), np.arange(51)]).astype(int)
    # Last beat at 5s lies at ref frame ~215, far beyond the path range.
    ref_beats = np.array([0.0, 0.1, 5.0])
    recovered = map_reference_beats(wp, ref_beats, hop_length=hop,
                                    sample_rate=sr)
    assert recovered.size == 2   # the 5.0s beat is dropped


def test_feature_dim_mismatch_raises():
    aligner = OfflineDTWAligner(sample_rate=22050)
    a = np.zeros((12, 10), dtype=np.float32)
    b = np.zeros((6, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        aligner.align(a, b)


def test_non_2d_input_raises():
    aligner = OfflineDTWAligner(sample_rate=22050)
    a = np.zeros((12,), dtype=np.float32)
    b = np.zeros((12, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        aligner.align(a, b)


def test_frame_rate_helper():
    aligner = OfflineDTWAligner(sample_rate=22050, hop_length=512)
    assert abs(aligner.frame_rate - 22050 / 512) < 1e-9
