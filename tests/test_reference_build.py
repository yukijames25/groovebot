"""groovebot.align.reference — bundle construction + Demucs lazy-import."""
from __future__ import annotations

import numpy as np
import pytest

from groovebot.align.reference import ReferenceBundle, build_reference


def _sine_arpeggio(sr: int = 22050, n_beats: int = 12,
                   beat_period: float = 0.5) -> np.ndarray:
    """A simple mono sinusoid arpeggio. Sustained tones (smooth envelope)
    so pyin can track the F0 across beats."""
    freqs = (261.63, 329.63, 392.00, 523.25)  # C-E-G-C
    n = int(n_beats * beat_period * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    ramp = 0.05  # 50 ms attack/release
    for i in range(n_beats):
        f = freqs[i % len(freqs)]
        seg = (t >= i * beat_period) & (t < (i + 1) * beat_period)
        local = t[seg] - i * beat_period
        env = np.minimum(local / ramp, 1.0) * \
              np.minimum((beat_period - local) / ramp, 1.0)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        out[seg] = (np.sin(2 * np.pi * f * local) * env).astype(np.float32)
    return out


def test_build_reference_with_supplied_vocal_skips_demucs():
    """Passing `vocal_audio` must take the Demucs-free fast path."""
    sr = 22050
    audio = _sine_arpeggio(sr=sr, n_beats=10)
    beats = np.arange(10) * 0.5
    bundle = build_reference(audio, sr, beats, hop_length=512,
                             vocal_audio=audio)
    assert isinstance(bundle, ReferenceBundle)
    assert bundle.chroma.shape[0] == 12
    assert bundle.melody.shape[0] == 12
    assert bundle.chroma.shape[1] > 0
    assert bundle.melody.shape[1] > 0
    assert bundle.sample_rate == sr
    assert bundle.hop_length == 512
    assert np.array_equal(bundle.beats, beats.astype(float))


def test_build_reference_keeps_vocal_audio_in_bundle():
    sr = 22050
    audio = _sine_arpeggio(sr=sr, n_beats=8)
    bundle = build_reference(audio, sr, np.arange(8) * 0.5, hop_length=512,
                             vocal_audio=audio)
    assert bundle.vocal_audio.ndim == 1
    assert len(bundle.vocal_audio) == len(audio)


def test_build_reference_handles_stereo_vocal():
    sr = 22050
    mono = _sine_arpeggio(sr=sr, n_beats=8)
    stereo = np.stack([mono, mono], axis=-1)   # (samples, 2)
    bundle = build_reference(_sine_arpeggio(sr=sr, n_beats=8), sr,
                             np.arange(8) * 0.5,
                             hop_length=512, vocal_audio=stereo)
    assert bundle.vocal_audio.ndim == 1
    assert bundle.chroma.shape[0] == 12


def test_demucs_vocal_raises_clear_error_when_demucs_missing():
    """No Demucs in the local profile -> a RuntimeError with an install hint,
    NOT an ImportError that leaks to the caller. Reuses the message from
    tools.prep_dataset.separate_vocal."""
    pytest.importorskip("soundfile")
    try:
        import demucs.separate  # noqa: F401
    except Exception:
        from groovebot.align.reference import demucs_vocal
        with pytest.raises(RuntimeError):
            demucs_vocal(np.zeros(22050, dtype=np.float32), 22050)
    else:
        pytest.skip("demucs is installed; not testing the error path")
