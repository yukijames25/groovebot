"""groovebot.style.backbone — frozen PANNs CNN14 audio embedding (v3).

PANNs (Kong et al. 2020) CNN14 is the public-domain pretrained backbone
we use to lift GTZAN / MTG-Jamendo above the from-scratch ceiling we hit
in v2. The model is downloaded from Zenodo once
(`Cnn14_mAP=0.431.pth`, ~340 MB; place under `data/raw/`) and stays
frozen at inference. The output is a 2048-d clip embedding.

This wrapper is intentionally minimal:

  * lazy load — importing this module does NOT touch the checkpoint
    or pull in `panns_inference`. Both happen on the first `embed()`
    call. Lets pytest mock the backbone with `_inference_fn` and stay
    green without the 340 MB ckpt.
  * `embed(audio, sr)` returns a 2048-d float32 vector, with mono-fold
    + 32 kHz resample baked in (PANNs CNN14's training sample rate).
  * `embed_file(path, cache_dir=...)` caches the embedding to
    `<cache_dir>/<stem>.npy` so the next call returns the disk array
    without touching the GPU/CPU CNN.

Spec §14 module note v3 (transfer learning track) — see README v3.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np


PANNS_SR = 32000
EMBEDDING_DIM = 2048


class PannsBackbone:
    """Frozen PANNs CNN14 embedder with .npy file cache."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cpu",
        sample_rate: int = PANNS_SR,
        _inference_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ):
        """`checkpoint_path` is the Cnn14_mAP=0.431.pth on disk; we do
        not download it for you (Windows-friendly: the upstream
        `panns_inference` shells out to `wget`). `device='cpu'` is the
        default; pass `'cuda'` if you have a GPU.

        `_inference_fn` is the test/mock seam: when supplied, it is
        called as `_inference_fn(mono_audio_32k, sr=PANNS_SR)` and must
        return a (2048,) ndarray. The real backbone is then not loaded.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.sample_rate = int(sample_rate)
        self._tagger = None
        self._inference_fn = _inference_fn

    @property
    def is_real(self) -> bool:
        """True iff this backbone is the genuine PANNs CNN14
        (not a test mock)."""
        return self._inference_fn is None

    def _ensure_loaded(self) -> None:
        if self._tagger is not None or self._inference_fn is not None:
            return
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PANNs CNN14 checkpoint not found at {self.checkpoint_path}. "
                "Download Cnn14_mAP=0.431.pth from "
                "https://zenodo.org/record/3987831 to data/raw/ (~340 MB). "
                "See README v3 'Run' section."
            )
        from panns_inference import AudioTagging
        # AudioTagging will skip its own wget when the checkpoint exists.
        self._tagger = AudioTagging(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
        )

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return a 2048-d float32 embedding for one clip.

        Multi-channel audio is mono-folded; non-32 kHz audio is
        resampled to `self.sample_rate` first.
        """
        mono = _to_mono(audio)
        if sr != self.sample_rate:
            import librosa  # local import keeps module import light
            mono = librosa.resample(mono, orig_sr=sr, target_sr=self.sample_rate)
        if self._inference_fn is not None:
            emb = self._inference_fn(mono, self.sample_rate)
            return np.asarray(emb, dtype=np.float32).reshape(-1)
        self._ensure_loaded()
        import torch
        x = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            _clipwise, emb = self._tagger.inference(x)
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def embed_file(
        self,
        path: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> np.ndarray:
        """Embed an on-disk audio file. With `cache_dir`, writes/reads
        `<cache_dir>/<stem>.npy` so repeated runs skip the CNN."""
        import soundfile as sf
        path = Path(path)
        if cache_dir is not None:
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            cache_path = cache / (path.stem + ".npy")
            if cache_path.exists():
                return np.load(str(cache_path)).astype(np.float32)
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            emb = self.embed(audio, sr)
            np.save(str(cache_path), emb)
            return emb
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return self.embed(audio, sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)


__all__ = ["PannsBackbone", "PANNS_SR", "EMBEDDING_DIM"]
