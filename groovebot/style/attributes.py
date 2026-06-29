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

Bucket thresholds:
    `arousal_bucket()` reads tertile boundaries from
    `groovebot/style/affect_calibration.json` (written by
    `tools/calibrate_affect.py`). If the file is missing, falls back to
    the absolute 0.33 / 0.66 thresholds. The single source of truth lives
    in the JSON for auditability (n, date, source split).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import librosa
import numpy as np


_AROUSAL_RMS_SCALE = 0.20      # RMS ≈ 0.2 saturates the score (mastered music range)
_AROUSAL_ONSET_RATE_SAT = 6.0  # onsets / sec at which the score saturates
_AROUSAL_BUCKETS = ("low", "mid", "high")

# Fallback (used iff affect_calibration.json is absent).
_FALLBACK_LOW_MAX = 0.33
_FALLBACK_HIGH_MIN = 0.66

_CALIBRATION_PATH = Path(__file__).resolve().parent / "affect_calibration.json"
_CALIBRATION_CACHE: dict | None = None


def _load_calibration() -> Mapping[str, float] | None:
    """Return cached {bucket_low_max, bucket_high_min} from
    affect_calibration.json, or None if missing/unreadable. Cached at
    module level so each `arousal_bucket()` call is a dict lookup."""
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    try:
        data = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
        a = data["arousal"]
        _CALIBRATION_CACHE = {
            "bucket_low_max": float(a["bucket_low_max"]),
            "bucket_high_min": float(a["bucket_high_min"]),
        }
        return _CALIBRATION_CACHE
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
        return None


def reload_calibration() -> None:
    """Forget the cached calibration so the next `arousal_bucket()` call
    re-reads the JSON. Useful in tests after a fixture rewrites the file."""
    global _CALIBRATION_CACHE
    _CALIBRATION_CACHE = None


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

    Thresholds come from `affect_calibration.json` (tertiles of the
    DEAM-trained head's predictions on a held-out val split). If that
    file is missing the function falls back to absolute 0.33 / 0.66 so
    units without calibration still produce a legal bucket.

    Buckets are the table's primary discretisation; tempo is intentionally
    not used as a separate axis because BPM is already a leaky proxy for
    felt energy.
    """
    cal = _load_calibration()
    if cal is not None:
        lo, hi = cal["bucket_low_max"], cal["bucket_high_min"]
    else:
        lo, hi = _FALLBACK_LOW_MAX, _FALLBACK_HIGH_MIN
    a = float(arousal)
    if a < lo:
        return "low"
    if a < hi:
        return "mid"
    return "high"


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
