"""Lever 1: OfflineDTWAligner.subseq + groovebot.align.features.trim_silence.

The DAMP-S-AG diagnostic showed that the MIDI-route low scores came from
forcing the rendition's t=0 to MIDI[0]. These tests pin the behaviour of
the two fixes for that:

  - `OfflineDTWAligner(subseq=True)` lets the reference axis float at both
    ends, so a query that corresponds to a *sub-region* of the reference
    can be matched without paying boundary cost.
  - `trim_silence` snaps the audio so the first sung note sits at t=0,
    aligning the trimmed timeline with MIDI[0..] when the singer is
    actually singing the song from the beginning.
"""
from __future__ import annotations

import numpy as np

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import trim_silence


# --------------------------------------------------------------------------- #
# subseq DTW
# --------------------------------------------------------------------------- #
def _chroma_like_song(T: int = 200, period: int = 10) -> np.ndarray:
    feats = np.zeros((12, T), dtype=np.float32)
    for i in range(T):
        feats[(i // period) % 12, i] = 1.0
    return feats


def test_subseq_default_is_false_for_backward_compat():
    aligner = OfflineDTWAligner(sample_rate=22050)
    assert aligner.subseq is False


def test_full_dtw_forces_query_zero_to_ref_zero():
    """Baseline: full DTW pins the corners. librosa returns wp end -> start,
    so wp[-1] is (start of X, start of Y) = (0, 0). The path has to pay
    boundary cost from (0, 0) up to the actual matching ref region."""
    ref = _chroma_like_song(T=200, period=10)
    qry = ref[:, 60:140].copy()
    aligner = OfflineDTWAligner(sample_rate=22050, subseq=False)
    wp = aligner.align(qry, ref)
    assert int(wp[-1, 0]) == 0
    assert int(wp[-1, 1]) == 0


def test_subseq_finds_a_matching_sub_region_in_the_reference():
    """The behavioural difference we care about: full DTW pins the start of
    the path to ref=0 (the corner); subseq DTW pulls the start towards
    the actual matching ref region. The end-point in subseq depends on
    librosa's backtrack tie-breaking (it can stop early once the cost
    matrix flattens), so we only assert that subseq moved the start away
    from the corner."""
    ref = _chroma_like_song(T=200, period=10)
    qry = ref[:, 60:140].copy()

    wp_full = OfflineDTWAligner(sample_rate=22050, subseq=False).align(qry, ref)
    wp_sub  = OfflineDTWAligner(sample_rate=22050, subseq=True ).align(qry, ref)

    # Full DTW: path starts at the (0, 0) corner.
    assert int(wp_full[-1, 0]) == 0
    assert int(wp_full[-1, 1]) == 0

    # Subseq DTW: matched ref start is in (or close to) the real sub-region.
    sub_start = int(wp_sub[-1, 1])
    assert sub_start >= 50, (
        f"subseq did not move start away from ref=0 (got {sub_start})"
    )


def test_subseq_warp_path_is_mostly_diagonal():
    """Inside the matched region the warp path should track the identity for
    the bulk of frames. We allow a few frames of horizontal slack at the
    endpoints (a librosa subseq backtracking artifact)."""
    ref = _chroma_like_song(T=200, period=10)
    qry = ref[:, 60:140].copy()
    aligner = OfflineDTWAligner(sample_rate=22050, subseq=True)
    wp = aligner.align(qry, ref)
    start_ref = int(wp[-1, 1])
    dev = np.abs(wp[:, 0] - (wp[:, 1] - start_ref))
    # Median deviation should be zero — bulk of the path is the diagonal.
    assert int(np.median(dev)) == 0
    # Most frames should be diagonal: at least 80% have dev <= 1.
    assert (dev <= 1).mean() >= 0.8, (
        f"subseq path drifted too far off-diagonal: {dev}"
    )


# --------------------------------------------------------------------------- #
# trim_silence
# --------------------------------------------------------------------------- #
def _signal_with_pads(sr: int = 22050, lead_s: float = 0.4,
                     sound_s: float = 1.0, trail_s: float = 0.3,
                     freq: float = 440.0) -> tuple[np.ndarray, float, float]:
    lead = np.zeros(int(lead_s * sr), dtype=np.float32)
    trail = np.zeros(int(trail_s * sr), dtype=np.float32)
    t = np.arange(int(sound_s * sr)) / sr
    sound = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    audio = np.concatenate([lead, sound, trail])
    return audio, lead_s, trail_s


def test_trim_silence_drops_leading_and_trailing(tmp_path):
    sr = 22050
    audio, lead_s, trail_s = _signal_with_pads(sr=sr)
    trimmed, leading, trailing = trim_silence(audio, sr,
                                              frame_length=2048,
                                              hop_length=512)
    # The RMS frame_length (~93 ms) sets the per-side resolution. Use a
    # 200 ms tolerance so the test isn't tied to exact frame placement.
    tol = 0.2
    assert abs(leading - lead_s) <= tol
    assert abs(trailing - trail_s) <= tol
    assert 0 < len(trimmed) <= len(audio)


def test_trim_silence_returns_audio_unchanged_when_below_threshold():
    sr = 22050
    silent = np.zeros(int(sr), dtype=np.float32)
    trimmed, leading, trailing = trim_silence(silent, sr)
    assert leading == 0.0
    assert trailing == 0.0
    assert len(trimmed) == len(silent)


def test_trim_silence_threshold_tunable():
    """A higher threshold (more permissive of silence) trims more aggressively."""
    sr = 22050
    audio, _, _ = _signal_with_pads(sr=sr, sound_s=1.0)
    # Mild signal: trimming at -25 dB picks up only the loudest part.
    trimmed_tight, leading_tight, _ = trim_silence(
        audio, sr, db_threshold=-25.0)
    trimmed_loose, leading_loose, _ = trim_silence(
        audio, sr, db_threshold=-60.0)
    # Tighter threshold leaves less audio than looser threshold.
    assert leading_tight >= leading_loose
    assert len(trimmed_tight) <= len(trimmed_loose)


def test_trim_silence_multichannel_to_mono():
    sr = 22050
    audio, _, _ = _signal_with_pads(sr=sr)
    stereo = np.stack([audio, audio], axis=-1)
    trimmed, _, _ = trim_silence(stereo, sr)
    assert trimmed.ndim == 1
