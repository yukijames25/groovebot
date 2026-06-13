"""Lever A: OfflineDTWAligner.band_rad (Sakoe-Chiba band).

Pins three behaviours:

  - default `band_rad=None` preserves the legacy back-compat path
    (no `global_constraints` keyword passed to librosa).
  - a wide band (~0.5) is permissive enough to leave the
    identity-recovery from `test_dtw_align.py` unchanged.
  - a narrow band suppresses off-diagonal drift: when the query is the
    reference plus a small lateral shift, the constrained warp path
    stays inside the band, whereas the unconstrained path can swing
    further off-diagonal at the boundary.
"""
from __future__ import annotations

import numpy as np

from groovebot.align.dtw_align import OfflineDTWAligner


def _chroma_like_song(T: int = 200, period: int = 10) -> np.ndarray:
    feats = np.zeros((12, T), dtype=np.float32)
    for i in range(T):
        feats[(i // period) % 12, i] = 1.0
    return feats


def test_band_rad_default_is_none_for_back_compat():
    aligner = OfflineDTWAligner(sample_rate=22050)
    assert aligner.band_rad is None


def test_identity_warp_with_wide_band_recovers_beats():
    """A wide band (0.5) should not change the identity-case recovery —
    the warp path is still the diagonal, so reference beats round-trip
    within one frame on the query side."""
    sr, hop = 22050, 512
    feats = _chroma_like_song(T=120, period=10)
    aligner = OfflineDTWAligner(sample_rate=sr, hop_length=hop, band_rad=0.5)
    wp = aligner.align(feats, feats)
    ref_beats = np.arange(12) * (10 * hop / sr)
    recovered = aligner.map_reference_beats(wp, ref_beats)
    tol = hop / sr
    assert recovered.size == ref_beats.size
    assert float(np.max(np.abs(recovered - ref_beats))) <= tol


def test_narrow_band_constrains_warp_to_diagonal():
    """With a narrow band, the warp path is forced inside `band_rad *
    max(Tq, Tr)` of the diagonal. Verified directly on the path
    coordinates."""
    sr, hop = 22050, 512
    qry = _chroma_like_song(T=200, period=10)
    ref = _chroma_like_song(T=200, period=10)
    # Lateral shift in the query that an unconstrained DTW might absorb
    # by drifting off-diagonal; the band forces it back.
    qry_shifted = np.roll(qry, shift=2, axis=1)
    band = 0.05
    aligner = OfflineDTWAligner(
        sample_rate=sr, hop_length=hop, band_rad=band,
    )
    wp = aligner.align(qry_shifted, ref)
    Tq, Tr = qry_shifted.shape[1], ref.shape[1]
    # Project query frame onto reference axis along the diagonal slope.
    expected_r = wp[:, 0].astype(float) * (Tr / Tq)
    dev_frames = np.abs(wp[:, 1].astype(float) - expected_r)
    allowed_frames = band * float(max(Tq, Tr))
    # Small tolerance for librosa's band edge inclusivity.
    assert float(dev_frames.max()) <= allowed_frames + 1.5, (
        f"warp path drifted {dev_frames.max():.1f}f off-diagonal, "
        f"band allows {allowed_frames:.1f}f"
    )


def test_band_rad_returns_a_valid_warp_path_shape():
    sr, hop = 22050, 512
    feats = _chroma_like_song(T=60, period=5)
    aligner = OfflineDTWAligner(sample_rate=sr, hop_length=hop, band_rad=0.2)
    wp = aligner.align(feats, feats)
    assert wp.ndim == 2 and wp.shape[1] == 2
    assert wp.dtype.kind in "iu"
    assert wp.size > 0
