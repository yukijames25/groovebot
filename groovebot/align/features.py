"""groovebot.align.features — feature extraction for offline reference alignment.

Both feature kinds return a (12, T) ndarray suitable for DTW, so callers
(`OfflineDTWAligner`, later online aligners) can swap features without
changing the cost-matrix code.

- `"chroma"`: librosa.feature.chroma_cqt. Good for harmonised vocals or full
  mixes where pitch class is stable across the harmonic stack.
- `"pitch"`:  librosa.pyin -> f0 contour -> one-hot pitch class per frame.
  Good for monophonic humming where chroma is noisy because there is no
  harmonic content.

For the DAMP route (§9.x), `pyin_f0` and `consensus_f0` expose the raw F0
contour primitive so a melody reference can be built from multiple
renditions of the same arrangement (frame-wise median, leave-one-out).
"""
from __future__ import annotations
import warnings
from typing import Literal, Sequence

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
        f0 = pyin_f0(audio, sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        return f0_to_pitch_chroma(f0)
    raise ValueError(f"unknown feature kind: {kind!r}")


def pyin_f0(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float | None = None,
    fmax: float | None = None,
) -> np.ndarray:
    """Return the pyin F0 contour for `audio` (NaN where unvoiced).

    Exposed so the DAMP route can median-pool F0 across renditions before
    binning to a pitch-class chroma."""
    audio = _to_mono(audio)
    fmin_hz = float(fmin) if fmin is not None else float(librosa.note_to_hz("C2"))
    fmax_hz = float(fmax) if fmax is not None else float(librosa.note_to_hz("C7"))
    f0, _voiced, _voiced_probs = librosa.pyin(
        y=audio, sr=sr,
        fmin=fmin_hz, fmax=fmax_hz,
        hop_length=hop_length,
    )
    return f0


def consensus_f0(f0_contours: Sequence[np.ndarray]) -> np.ndarray:
    """Frame-wise nanmedian of multiple F0 contours.

    Inputs are padded to the longest length with NaN so contours of slightly
    different length still align frame-by-frame. NaN frames are ignored in
    the median; if every contour is NaN at frame `t`, the result is NaN at
    `t` (carried through `f0_to_pitch_chroma` as an unvoiced frame).

    Use for DAMP-style "melody reference = median across renditions, leave-
    one-out" semantics (spec §9.x DAMP route).
    """
    if not f0_contours:
        return np.empty(0, dtype=float)
    L = max(len(c) for c in f0_contours)
    arr = np.full((len(f0_contours), L), np.nan, dtype=float)
    for i, c in enumerate(f0_contours):
        arr[i, : len(c)] = c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice
        return np.nanmedian(arr, axis=0)


def f0_to_pitch_chroma(f0_hz: np.ndarray) -> np.ndarray:
    """Bin a per-frame F0 contour (Hz, NaN where unvoiced) into a (12, T)
    one-hot pitch-class matrix."""
    T = len(f0_hz)
    chroma = np.zeros((12, T), dtype=np.float32)
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    if voiced.any():
        midi = librosa.hz_to_midi(f0_hz[voiced])
        pc = np.mod(np.round(midi).astype(int), 12)
        idx_t = np.nonzero(voiced)[0]
        chroma[pc, idx_t] = 1.0
    return chroma


# Back-compat: keep the old private name pointing at the public one so
# any external import that grabbed `_f0_to_pitch_chroma` still works.
_f0_to_pitch_chroma = f0_to_pitch_chroma


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        # librosa convention is (channels, samples). soundfile gives (samples,
        # channels). Average whichever axis is larger -> mono.
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
