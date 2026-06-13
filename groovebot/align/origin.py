"""groovebot.align.origin — GT-non-referenced per-rendition origin anchor.

Lever B for the DAMP-S-AG MIDI route. The diagnostic showed many renditions
carry a small systematic time offset against the MIDI timeline (LOW case
moved F 0.07 -> 0.41 with +0.15 s of shift). DTW's corner-pinning means
that offset gets absorbed as boundary scramble instead of a clean
translation, dragging the score down.

This module estimates that offset *without ever consulting the GT beat
grid* — the calibration is data-derived only, so applying it to the
recovered beats before scoring is not test leakage.

The signal we cross-correlate is:

  - query side:    `librosa.onset.onset_strength(query_audio)` — a smooth
                   per-frame attack envelope.
  - reference side: a synthetic onset envelope built from MIDI *note-on*
                   times (NOT MIDI beats). Note-ons are an audible-attack
                   feature, distinct from the score's metric beat grid.

We return the time lag (sec) such that `query_time ≈ midi_time + lag`:

  - positive lag: the rendition's sung onsets land later in absolute time
    than the corresponding MIDI note-ons (the rendition is "behind"
    the score). The caller subtracts `lag` from the recovered beat
    times to bring them onto the MIDI timeline before scoring.
"""
from __future__ import annotations

import librosa
import numpy as np


def estimate_origin_offset(
    query_audio: np.ndarray,
    midi_note_onsets_sec: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    max_lag_sec: float = 2.0,
    smooth_frames: int = 3,
) -> float:
    """Estimate per-rendition time lag (sec) between query and MIDI.

    Cross-correlates the query's onset-strength envelope against a
    synthetic envelope formed from MIDI note-on times; the lag that
    maximises the inner product within ``[-max_lag_sec, +max_lag_sec]``
    is returned.

    `midi_note_onsets_sec` must be the note-on times from
    `MidiReference.note_onsets` — NOT `MidiReference.beats`, which is GT.

    Returns 0.0 when there are too few onsets on either side to compute a
    correlation. The return value is intended to be subtracted from
    DTW-recovered beat times: ``recovered_in_midi_time = recovered - lag``.
    """
    a = _to_mono(query_audio)
    if a.size == 0:
        return 0.0
    onset_env = librosa.onset.onset_strength(
        y=a, sr=sr, hop_length=hop_length,
    ).astype(np.float64)
    if onset_env.size == 0 or float(onset_env.max()) <= 0.0:
        return 0.0

    frame_rate = sr / hop_length
    midi_env = _midi_onset_envelope(
        midi_note_onsets_sec,
        n_frames=len(onset_env),
        frame_rate=frame_rate,
        smooth_frames=smooth_frames,
    )
    if float(midi_env.max()) <= 0.0:
        return 0.0

    # Mean-subtract both signals — correlate's amplitude bias would
    # otherwise prefer dense overlap regions regardless of structure.
    q = onset_env - onset_env.mean()
    m = midi_env - midi_env.mean()

    max_lag_frames = max(1, int(round(max_lag_sec * frame_rate)))
    best_lag = _xcorr_argmax_within_lag(q, m, max_lag_frames)
    return float(best_lag / frame_rate)


def _midi_onset_envelope(
    midi_note_onsets_sec: np.ndarray,
    *,
    n_frames: int,
    frame_rate: float,
    smooth_frames: int,
) -> np.ndarray:
    """Place a unit pulse at each MIDI note-on frame, then box-smooth.

    Smoothing trades a tiny localisation cost (~`smooth_frames`/`frame_rate`
    seconds) for robustness: `onset_strength` has its own ~1-2 frame
    spread, and a single-frame pulse would correlate poorly against it.
    """
    env = np.zeros(int(n_frames), dtype=np.float64)
    onsets = np.asarray(midi_note_onsets_sec, dtype=float)
    if onsets.size == 0:
        return env
    frames = np.round(onsets * frame_rate).astype(int)
    keep = (frames >= 0) & (frames < n_frames)
    frames = frames[keep]
    if frames.size == 0:
        return env
    np.add.at(env, frames, 1.0)
    if smooth_frames > 1:
        kernel = np.ones(int(smooth_frames), dtype=np.float64) / float(smooth_frames)
        env = np.convolve(env, kernel, mode="same")
    return env


def _xcorr_argmax_within_lag(
    q: np.ndarray, m: np.ndarray, max_lag_frames: int,
) -> int:
    """Return the integer lag k in [-max_lag, +max_lag] maximising
    sum_i q[i] * m[i - k].

    Equivalent to scipy.signal.correlate(q, m, 'full') argmax restricted
    to a lag window, but written without scipy so the import surface
    stays librosa-only.
    """
    L = len(q)
    if L == 0 or len(m) == 0:
        return 0
    max_k = int(min(max_lag_frames, L - 1, len(m) - 1))
    best_k = 0
    best_val = -np.inf
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            # q[k:k+W] * m[0:W]; need W <= L - k and W <= len(m).
            W = min(L - k, len(m))
            if W <= 0:
                continue
            val = float(np.dot(q[k:k + W], m[:W]))
        else:
            W = min(L, len(m) + k)
            if W <= 0:
                continue
            val = float(np.dot(q[:W], m[-k:-k + W]))
        if val > best_val:
            best_val = val
            best_k = k
    return best_k


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
