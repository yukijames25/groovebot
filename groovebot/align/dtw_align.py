"""groovebot.align.dtw_align — offline DTW alignment.

For the M0' Tier 1 feasibility check (spec §9.x). Online alignment (online
DTW / score following) is M2's job and lives behind the `ReferenceAligner`
Protocol (§5.2). The feature pipeline (`features.py`) is shared.

NFR-2 (causal/online) applies to the production system, not to M0' offline
verification.
"""
from __future__ import annotations
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class OfflineDTWAligner:
    """Wrap librosa.sequence.dtw with frame-rate bookkeeping.

    Stateless apart from (sample_rate, hop_length, metric, subseq); call
    `align()` with a fresh (query, reference) pair each time.

    When `subseq=True`, the query (`X`) must be matched entirely but the
    reference (`Y`) gets boundary slack: the path may start at any
    reference frame and end at any reference frame. Designed for
    DAMP-style rendition-vs-MIDI alignment where the rendition does
    not necessarily begin at MIDI frame 0 (see diagnostic notes in
    docs/SYSTEM_SPEC.md §9.x DAMP).
    """
    sample_rate: int
    hop_length: int = 512
    metric: str = "euclidean"  # robust to all-zero pitch-chroma columns
    subseq: bool = False

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def align(
        self,
        query_feats: np.ndarray,
        ref_feats: np.ndarray,
    ) -> np.ndarray:
        """Return the warp path as an (N, 2) int array of (query_frame, ref_frame).

        librosa orders rows from end -> start; we return that as-is so callers
        relying on librosa conventions get what they expect.
        `map_reference_beats` is order-agnostic.
        """
        if query_feats.ndim != 2 or ref_feats.ndim != 2:
            raise ValueError("features must be 2-D (D, T)")
        if query_feats.shape[0] != ref_feats.shape[0]:
            raise ValueError(
                f"feature dim mismatch: query D={query_feats.shape[0]}, "
                f"ref D={ref_feats.shape[0]}"
            )
        _D, wp = librosa.sequence.dtw(
            X=query_feats, Y=ref_feats,
            metric=self.metric, subseq=self.subseq,
        )
        return np.asarray(wp, dtype=int)

    def map_reference_beats(
        self,
        warp_path: np.ndarray,
        ref_beats_sec: np.ndarray,
    ) -> np.ndarray:
        """Map reference beat times to query-side times via the warp path."""
        return map_reference_beats(
            warp_path=warp_path,
            ref_beats_sec=ref_beats_sec,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )


def map_reference_beats(
    warp_path: np.ndarray,
    ref_beats_sec: np.ndarray,
    hop_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Interpolate reference-frame -> query-frame along the warp path.

    Beats whose reference frame falls outside the path's r-range are dropped.
    For each unique r in the path we take the mean q (DTW paths are monotonic,
    so mean ≈ median and is faster).
    """
    wp = np.asarray(warp_path, dtype=int)
    ref_beats_sec = np.asarray(ref_beats_sec, dtype=float)
    if wp.size == 0 or ref_beats_sec.size == 0:
        return np.empty(0, dtype=float)

    q_frames = wp[:, 0]
    r_frames = wp[:, 1]
    order = np.argsort(r_frames)
    r_frames = r_frames[order]
    q_frames = q_frames[order]

    uniq_r, inv = np.unique(r_frames, return_inverse=True)
    sums = np.bincount(inv, weights=q_frames.astype(float),
                       minlength=len(uniq_r))
    counts = np.bincount(inv, minlength=len(uniq_r))
    mean_q = sums / counts

    frame_rate = sample_rate / hop_length
    ref_beats_frames = ref_beats_sec * frame_rate
    q_for_beats = np.interp(
        ref_beats_frames, uniq_r.astype(float), mean_q,
        left=np.nan, right=np.nan,
    )
    valid = ~np.isnan(q_for_beats)
    return q_for_beats[valid] / frame_rate
