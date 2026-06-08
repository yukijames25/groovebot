"""groovebot.align.reference — build a Tier-2 reference bundle for offline
alignment of real renditions (singing / humming) against a known song.

Per spec §9.x M0' Tier 2: we split the reference song into a vocal stem
(Demucs) and derive two feature views from that stem:

- **chroma** — for singing renditions (harmonic content -> chroma stable).
- **melody** — pyin F0 binned into one-hot pitch classes, for humming
  renditions (no harmonic stack -> chroma is noisy, pitch contour is the
  signal).

Both views share `(12, T)` shape so the same `OfflineDTWAligner` can run
against either without changing its cost path.

Demucs is loaded lazily (it lives on the experiments profile, not the
local profile). Tests pass a precomputed vocal stem via the `vocal_audio`
kwarg so the Demucs import is never triggered.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from groovebot.align.features import extract_align_features


@dataclass
class ReferenceBundle:
    """Tier-2 reference bundle.

    Holds the per-song state we need to align *any* number of renditions
    of the same song without re-running Demucs / pyin per rendition. Built
    once per song by `build_reference()`.
    """
    beats: np.ndarray            # reference beat times (seconds)
    vocal_audio: np.ndarray      # mono vocal stem at `sample_rate`
    chroma: np.ndarray           # (12, T) chroma of vocal stem (singing query)
    melody: np.ndarray           # (12, T) one-hot pitch class from pyin (humming query)
    sample_rate: int
    hop_length: int


def build_reference(
    audio: np.ndarray,
    sr: int,
    beats: np.ndarray,
    hop_length: int = 512,
    *,
    vocal_audio: np.ndarray | None = None,
) -> ReferenceBundle:
    """Build a `ReferenceBundle` from a reference song's full-mix audio.

    By default the full mix is fed to Demucs to extract the vocal stem.
    Pass `vocal_audio` explicitly to skip Demucs (e.g. a pre-separated
    stem on disk, or a synthetic stand-in in tests).

    `beats` are the reference beat times in seconds.
    """
    if vocal_audio is None:
        vocal_audio = demucs_vocal(audio, sr)
    vocal_mono = _to_mono(vocal_audio)
    chroma = extract_align_features(
        vocal_mono, sr, kind="chroma", hop_length=hop_length,
    )
    melody = extract_align_features(
        vocal_mono, sr, kind="pitch", hop_length=hop_length,
    )
    return ReferenceBundle(
        beats=np.asarray(beats, dtype=float),
        vocal_audio=vocal_mono,
        chroma=chroma,
        melody=melody,
        sample_rate=int(sr),
        hop_length=int(hop_length),
    )


def demucs_vocal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run Demucs and return the mono vocal stem at `sr`.

    Lazy-imports the Demucs entry point via `tools.prep_dataset.separate_vocal`
    so this module can be imported without Demucs installed. Falls back with
    a clear error message when Demucs is absent.

    Writes the input to a temp WAV (Demucs's CLI works on files) and reads
    back the resulting vocals.wav. This is slower than a direct in-memory
    call but matches the existing prep path exactly, so we don't fork two
    Demucs invocations to maintain.
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from tools.prep_dataset import separate_vocal   # lazy import

    mono = _to_mono(audio)
    with tempfile.TemporaryDirectory(prefix="grv_demucs_") as tmp:
        in_path = Path(tmp) / "ref.wav"
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        sf.write(str(in_path), mono, sr)
        vocals_path = separate_vocal(str(in_path), str(out_dir))
        vocals, sr_v = sf.read(vocals_path, dtype="float32", always_2d=False)
    if sr_v != sr:
        raise RuntimeError(
            f"Demucs returned vocals at sr={sr_v}, expected {sr}. "
            "Resample upstream or change the reference sample rate."
        )
    return _to_mono(vocals)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)
