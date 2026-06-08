"""groovebot.align.features — feature extraction for offline reference alignment.

Both feature kinds return a (12, T) ndarray suitable for DTW, so callers
(`OfflineDTWAligner`, later online aligners) can swap features without
changing the cost-matrix code.

- `"chroma"`: librosa.feature.chroma_cqt. Good for harmonised vocals or full
  mixes where pitch class is stable across the harmonic stack.
- `"pitch"`:  librosa.pyin -> f0 contour -> one-hot pitch class per frame.
  Good for monophonic humming where chroma is noisy because there is no
  harmonic content.
"""
from __future__ import annotations
from typing import Literal

import librosa
import numpy as np


FeatureKind = Literal["chroma", "pitch"]


def extract_align_features(
    audio: np.ndarray,
    sr: int,
    kind: FeatureKind = "chroma",
    hop_length: int = 512,
    fmin: float | None = None,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a (12, T) feature sequence usable as a DTW input.

    Multi-channel audio is averaged to mono first.
    """
    audio = _to_mono(audio)
    if kind == "chroma":
        return librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=hop_length,
        ).astype(np.float32)
    if kind == "pitch":
        fmin_hz = float(fmin) if fmin is not None else float(librosa.note_to_hz("C2"))
        fmax_hz = float(fmax) if fmax is not None else float(librosa.note_to_hz("C7"))
        f0, _voiced, _voiced_probs = librosa.pyin(
            y=audio, sr=sr,
            fmin=fmin_hz, fmax=fmax_hz,
            hop_length=hop_length,
        )
        return _f0_to_pitch_chroma(f0)
    raise ValueError(f"unknown feature kind: {kind!r}")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        # librosa convention is (channels, samples). soundfile gives (samples,
        # channels). Average whichever axis is larger -> mono.
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)


def _f0_to_pitch_chroma(f0_hz: np.ndarray) -> np.ndarray:
    """Bin a per-frame f0 contour into a (12, T) one-hot pitch-class matrix."""
    T = len(f0_hz)
    chroma = np.zeros((12, T), dtype=np.float32)
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    if voiced.any():
        midi = librosa.hz_to_midi(f0_hz[voiced])
        pc = np.mod(np.round(midi).astype(int), 12)
        idx_t = np.nonzero(voiced)[0]
        chroma[pc, idx_t] = 1.0
    return chroma
