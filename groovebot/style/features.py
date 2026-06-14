"""groovebot.style.features — log-mel spectrogram for the StyleCNN input.

The startup window (≈5-10 s) is the only thing the style head sees, so we
keep the front end close to standard MIR practice (chroma_cqt of
`groovebot.align.features` is the timing path's; this one is mel).

Output shape: (n_mels, T) float32. T is variable. The model handles the
time dim via `AdaptiveAvgPool2d`, so we deliberately do *not* pad/crop
here — that keeps the feature function pure and lets a caller window
however they want.
"""
from __future__ import annotations

import librosa
import numpy as np


DEFAULT_SR = 22050
DEFAULT_N_MELS = 64
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_FFT = 2048


def log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a (n_mels, T) log-mel spectrogram, float32.

    Multi-channel audio is averaged to mono first. If `sr != target_sr` the
    signal is resampled so the mel filter bank sees a fixed sample rate
    regardless of source.

    Log scale is `librosa.power_to_db` (dB-like). The dynamic range is
    typically -80..+0; downstream training applies per-batch normalization
    so absolute scale is not critical.
    """
    mono = _to_mono(audio)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    sr_eff = target_sr
    f_max_eff = float(fmax) if fmax is not None else float(sr_eff) / 2.0
    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=sr_eff,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=float(fmin),
        fmax=f_max_eff,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32, copy=False)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
