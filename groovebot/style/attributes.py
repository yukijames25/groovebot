"""groovebot.style.attributes — tempo + arousal from raw audio (no learning).

These two attributes are computed by signal-processing heuristics rather
than the StyleCNN heads because:

  - tempo is reliably nailed by `librosa.beat.beat_track` for most music
    in the 60-200 BPM band; training a regression head adds risk for no
    benefit at v1.
  - arousal is a 0..1 felt-energy score combining RMS loudness and
    onset density. A learned head would need a labelled corpus we don't
    yet have. The heuristic is good enough to drive the arousal bucket
    in `table.select_move`, and can be replaced by a model head later
    without changing call sites (`select.py` only sees the float).

If a future revision wants either as a learned head, register it in
`StyleCNN.heads` and route the inference through `select.py`.
"""
from __future__ import annotations

import librosa
import numpy as np


_AROUSAL_RMS_SCALE = 0.20      # RMS ≈ 0.2 saturates the score (mastered music range)
_AROUSAL_ONSET_RATE_SAT = 6.0  # onsets / sec at which the score saturates
_AROUSAL_BUCKETS = ("low", "mid", "high")


def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    """Return a single BPM estimate. Mono-folds multi-channel input first.

    Uses `librosa.beat.beat_track`, which returns one scalar tempo (the
    dynamic programming consensus). Octave errors are possible — same
    caveat as any monophonic tempo estimator. The selector treats tempo
    as informational; the table keys on the arousal bucket, not BPM.
    """
    mono = _to_mono(audio)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    # librosa may return a (1,) array or a scalar depending on version.
    tempo_val = np.asarray(tempo).reshape(-1)
    return float(tempo_val[0]) if tempo_val.size else 0.0


def estimate_arousal(audio: np.ndarray, sr: int) -> float:
    """0..1 felt-energy score blending RMS loudness and onset density.

    Defined as:

        rms_score   = clip(mean(RMS) / RMS_SCALE,           0, 1)
        onset_score = clip(onsets_per_sec / ONSET_RATE_SAT, 0, 1)
        arousal     = sqrt(rms_score * onset_score)         # geometric mean

    Geometric (not arithmetic) mean is deliberate: `librosa.onset.onset_detect`
    happily finds 30+ "onsets" in 4 s of near-silent noise because its peak
    picker has no absolute amplitude gate. A near-silent input has tiny RMS
    but inflated onset density, and an arithmetic mean would call that
    moderate-energy when a listener would call it silent. The geometric
    form self-gates: rms_score≈0 collapses arousal regardless of onset
    noise. For energetic music the two scores are comparable and the
    geometric mean tracks the arithmetic one within a few percent.
    """
    mono = _to_mono(audio)
    duration_sec = float(len(mono)) / float(sr)
    if duration_sec <= 0:
        return 0.0
    rms = librosa.feature.rms(y=mono)[0]
    rms_score = float(np.clip(np.mean(rms) / _AROUSAL_RMS_SCALE, 0.0, 1.0))
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onsets_per_sec = float(len(onsets)) / max(duration_sec, 1e-6)
    onset_score = float(np.clip(onsets_per_sec / _AROUSAL_ONSET_RATE_SAT, 0.0, 1.0))
    return float(np.sqrt(rms_score * onset_score))


def arousal_bucket(arousal: float) -> str:
    """Map a 0..1 arousal value to "low" / "mid" / "high".

    Thresholds: <0.33 low, <0.66 mid, else high. Buckets are the table's
    primary discretisation; tempo is intentionally not used as a separate
    axis because BPM is already a leaky proxy for felt energy.
    """
    a = float(arousal)
    if a < 0.33:
        return "low"
    if a < 0.66:
        return "mid"
    return "high"


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
