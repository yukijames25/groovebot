"""Tests for groovebot.style.backbone (PannsBackbone wrapper).

The real PANNs CNN14 ckpt is 340 MB and is NOT required to run these
tests — `PannsBackbone` accepts an `_inference_fn` test seam that
returns a (2048,) ndarray, and that path covers every public method
that does not need the real model.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from groovebot.style.backbone import EMBEDDING_DIM, PANNS_SR, PannsBackbone


SR = 22050


def _mock_inference(audio: np.ndarray, sr: int) -> np.ndarray:
    """Deterministic 2048-d vector keyed on the audio's RMS."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-9))
    rng = np.random.default_rng(int(rms * 1e6) % (2**31))
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def test_embed_shape_with_mock():
    bb = PannsBackbone(_inference_fn=_mock_inference)
    audio = 0.1 * np.random.default_rng(0).standard_normal(SR * 5).astype(np.float32)
    emb = bb.embed(audio, SR)
    assert emb.shape == (EMBEDDING_DIM,)
    assert emb.dtype == np.float32


def test_embed_resamples_to_panns_sr():
    seen_sr = []

    def fn(audio, sr):
        seen_sr.append(sr)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    bb = PannsBackbone(_inference_fn=fn)
    audio = np.zeros(44100 * 3, dtype=np.float32)
    bb.embed(audio, 44100)
    assert seen_sr == [PANNS_SR]


def test_embed_handles_stereo():
    bb = PannsBackbone(_inference_fn=_mock_inference)
    left = 0.1 * np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32)
    right = 0.1 * np.random.default_rng(2).standard_normal(SR * 2).astype(np.float32)
    stereo = np.stack([left, right], axis=-1)
    emb = bb.embed(stereo, SR)
    assert emb.shape == (EMBEDDING_DIM,)


def test_is_real_is_false_with_mock():
    bb = PannsBackbone(_inference_fn=_mock_inference)
    assert bb.is_real is False


def test_missing_ckpt_raises_clear_error_on_first_real_load(tmp_path):
    bb = PannsBackbone(checkpoint_path=tmp_path / "missing.pth")
    audio = np.zeros(SR * 2, dtype=np.float32)
    with pytest.raises(FileNotFoundError, match="PANNs CNN14 checkpoint"):
        bb.embed(audio, SR)


def test_embed_file_uses_npy_cache(tmp_path):
    bb = PannsBackbone(_inference_fn=_mock_inference)
    wav_path = tmp_path / "clip.wav"
    audio = 0.05 * np.random.default_rng(7).standard_normal(SR * 4).astype(np.float32)
    sf.write(str(wav_path), audio, SR)
    cache_dir = tmp_path / "cache"
    emb1 = bb.embed_file(wav_path, cache_dir=cache_dir)
    assert (cache_dir / "clip.npy").exists()
    # Modify the on-disk wav; cached value should still come back.
    sf.write(str(wav_path), np.zeros_like(audio), SR)
    emb2 = bb.embed_file(wav_path, cache_dir=cache_dir)
    assert np.array_equal(emb1, emb2)


def test_embed_file_without_cache_dir_still_works(tmp_path):
    bb = PannsBackbone(_inference_fn=_mock_inference)
    wav_path = tmp_path / "clip.wav"
    audio = 0.05 * np.random.default_rng(8).standard_normal(SR * 2).astype(np.float32)
    sf.write(str(wav_path), audio, SR)
    emb = bb.embed_file(wav_path)
    assert emb.shape == (EMBEDDING_DIM,)
