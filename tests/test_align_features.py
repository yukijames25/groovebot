"""groovebot.align.features — shape & sparsity smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from groovebot.align.features import extract_align_features


def _sine(freq: float = 440.0, dur: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_chroma_returns_12_T_float32():
    sr = 22050
    feats = extract_align_features(_sine(440.0, dur=1.0, sr=sr), sr,
                                   kind="chroma", hop_length=512)
    assert feats.ndim == 2
    assert feats.shape[0] == 12
    assert feats.shape[1] > 0
    assert feats.dtype == np.float32


def test_pitch_returns_12_T_float32():
    sr = 22050
    # 1s of monophonic A4 - long enough for pyin without dragging the suite.
    feats = extract_align_features(_sine(440.0, dur=1.0, sr=sr), sr,
                                   kind="pitch", hop_length=512)
    assert feats.shape[0] == 12
    assert feats.shape[1] > 0
    assert feats.dtype == np.float32


def test_pitch_chroma_is_one_hot_per_frame():
    """Each frame has at most one nonzero pitch class (one-hot or all-zero
    when unvoiced)."""
    sr = 22050
    feats = extract_align_features(_sine(440.0, dur=1.0, sr=sr), sr,
                                   kind="pitch", hop_length=512)
    per_frame_nonzero = (feats > 0).sum(axis=0)
    assert int(per_frame_nonzero.max()) <= 1


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        extract_align_features(_sine(440.0), 22050, kind="garbage")  # type: ignore[arg-type]


def test_multichannel_audio_is_averaged_to_mono():
    sr = 22050
    mono = _sine(440.0, dur=0.5, sr=sr)
    stereo = np.stack([mono, mono], axis=-1)
    f_mono = extract_align_features(mono, sr, kind="chroma")
    f_stereo = extract_align_features(stereo, sr, kind="chroma")
    assert f_stereo.shape == f_mono.shape
