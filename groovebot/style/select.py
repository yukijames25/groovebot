"""groovebot.style.select — top-level GrooveStyleSelector.

Two modes, both produce the same `GrooveStyle` text-label contract:

  * **v1/v2 CNN path** (StyleCNN end-to-end from log-mel):
        log_mel_spectrogram -> StyleCNN -> (genre_probs, mood_probs)
        attributes.estimate_tempo  / arousal -> tempo / bucket
        table.select_move -> GrooveStyle

  * **v3 transfer-learning path** (frozen PANNs CNN14 + MLP head):
        PannsBackbone.embed -> StyleHead -> (genre_probs, mood_probs)
        (tempo / arousal / table identical to above)

The public `select(audio, sr) -> GrooveStyle` signature does not change
between v1, v2, v3. Pick a constructor:

  * `GrooveStyleSelector()`         — v1/v2 CNN, random weights
  * `GrooveStyleSelector(model=...)`— v1/v2 CNN, loaded weights
  * `GrooveStyleSelector(backbone=PannsBackbone(...), head=StyleHead())`
        — v3 transfer-learning
  * `GrooveStyleSelector.from_panns(ckpt_path, head_weights=...)`
        — convenience for the v3 path
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from groovebot.style.attributes import (
    arousal_bucket,
    estimate_arousal,
    estimate_tempo,
)
from groovebot.style.backbone import EMBEDDING_DIM, PANNS_SR, PannsBackbone
from groovebot.style.deam import sam_to_unit
from groovebot.style.features import (
    DEFAULT_N_MELS,
    DEFAULT_SR,
    log_mel_spectrogram,
)
from groovebot.style.model import (
    GENRES, MOODS, StyleCNN, StyleHead, StyleRegressionHead,
)
from groovebot.style.table import select_move


# An arousal source is `(audio, sr) -> 0..1`. The heuristic
# `estimate_arousal` already matches; the v3 learned head needs a
# small adapter (`make_panns_arousal_fn`) to compose backbone + head
# + DEAM calibrator into the same shape.
ArousalFn = Callable[[np.ndarray, int], float]


@dataclass
class GrooveStyle:
    """Text-label output of `GrooveStyleSelector`. Unchanged since v1."""
    move: str
    intensity: float
    genre: str
    mood: str
    mood_probs: dict[str, float]
    genre_probs: dict[str, float]
    tempo_bpm: float
    arousal: float
    arousal_bucket: str

    def as_text(self) -> str:
        return (
            f"{self.move}@{self.intensity:.2f} "
            f"({self.genre}/{self.mood}, {self.tempo_bpm:.0f}BPM, "
            f"arousal={self.arousal:.2f}/{self.arousal_bucket})"
        )


class GrooveStyleSelector:
    """End-to-end style selector. Two model paths share one selector
    (the call sites in M2 stay stable across the v2 → v3 migration)."""

    def __init__(
        self,
        model: StyleCNN | None = None,
        *,
        backbone: PannsBackbone | None = None,
        head: StyleHead | None = None,
        arousal_fn: ArousalFn | None = None,
        target_sr: int = DEFAULT_SR,
        n_mels: int = DEFAULT_N_MELS,
        device: str | torch.device | None = None,
    ):
        """`arousal_fn` overrides the v2 `estimate_arousal` heuristic
        with any `(audio, sr) -> 0..1` callable. Build the v3 DEAM-
        learned one with `make_panns_arousal_fn(backbone, head)`."""
        self.target_sr = int(target_sr)
        self.n_mels = int(n_mels)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.arousal_fn: ArousalFn = arousal_fn or estimate_arousal

        if backbone is not None or head is not None:
            if backbone is None or head is None:
                raise ValueError(
                    "GrooveStyleSelector: backbone and head must be passed "
                    "together (both define the v3 transfer-learning path)."
                )
            if model is not None:
                raise ValueError(
                    "GrooveStyleSelector: pass either `model` (v1/v2 CNN) "
                    "or `backbone`+`head` (v3), not both."
                )
            self.mode = "embedding"
            self.backbone = backbone
            self.head = head
            self.head.to(self.device)
            self.head.eval()
            self.model = None
        else:
            self.mode = "cnn"
            if model is None:
                model = StyleCNN(n_mels=self.n_mels)
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            self.backbone = None
            self.head = None

    @classmethod
    def from_panns(
        cls,
        checkpoint_path: str | Path,
        *,
        head_weights: str | Path | None = None,
        emb_dim: int = EMBEDDING_DIM,
        device: str | torch.device | None = None,
    ) -> "GrooveStyleSelector":
        """Build a v3 selector. `checkpoint_path` is the PANNs CNN14
        ckpt (`Cnn14_mAP=0.431.pth`); `head_weights` is a torch state
        dict for `StyleHead` (skip for random head — useful for smoke
        tests)."""
        dev = "cpu" if device is None else str(device)
        backbone = PannsBackbone(checkpoint_path, device=dev)
        head = StyleHead(emb_dim=emb_dim)
        if head_weights is not None:
            ck = torch.load(str(head_weights), map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            head.load_state_dict(sd)
        return cls(backbone=backbone, head=head, device=device)

    def _predict_cnn(self, audio: np.ndarray, sr: int) -> tuple[dict, dict]:
        mel = log_mel_spectrogram(
            audio, sr, target_sr=self.target_sr, n_mels=self.n_mels,
        )
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        probs = self.model.predict_probs(x)
        gp = probs["genre"].squeeze(0).cpu().numpy()
        mp = probs["mood"].squeeze(0).cpu().numpy()
        return (
            {g: float(p) for g, p in zip(GENRES, gp)},
            {m: float(p) for m, p in zip(MOODS, mp)},
        )

    def _predict_embedding(self, audio: np.ndarray, sr: int) -> tuple[dict, dict]:
        emb = self.backbone.embed(audio, sr)
        x = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        probs = self.head.predict_probs(x)
        gp = probs["genre"].squeeze(0).cpu().numpy()
        mp = probs["mood"].squeeze(0).cpu().numpy()
        return (
            {g: float(p) for g, p in zip(GENRES, gp)},
            {m: float(p) for m, p in zip(MOODS, mp)},
        )

    def select(self, audio: np.ndarray, sr: int) -> GrooveStyle:
        if self.mode == "embedding":
            genre_probs, mood_probs = self._predict_embedding(audio, sr)
        else:
            genre_probs, mood_probs = self._predict_cnn(audio, sr)

        genre = max(genre_probs, key=genre_probs.get)
        mood_argmax = max(mood_probs, key=mood_probs.get)

        tempo_bpm = estimate_tempo(audio, sr)
        arousal_val = float(self.arousal_fn(audio, sr))
        arousal_val = max(0.0, min(1.0, arousal_val))
        bucket = arousal_bucket(arousal_val)

        move, intensity = select_move(genre, bucket, mood_probs)
        return GrooveStyle(
            move=move,
            intensity=intensity,
            genre=genre,
            mood=mood_argmax,
            mood_probs=mood_probs,
            genre_probs=genre_probs,
            tempo_bpm=tempo_bpm,
            arousal=arousal_val,
            arousal_bucket=bucket,
        )


def make_panns_arousal_fn(
    backbone: PannsBackbone,
    head: StyleRegressionHead,
    *,
    target: str = "arousal",
    calibrator: Callable[[float], float] = sam_to_unit,
    device: str | torch.device | None = None,
) -> ArousalFn:
    """Build an `(audio, sr) -> 0..1` callable that runs
    `backbone -> head -> calibrator`.

    `target` picks which output of the regression head to use (the
    head ships both `arousal` and `valence`). `calibrator` maps the
    head's raw output to 0..1 — defaults to `sam_to_unit` which is the
    DEAM 1..9 SAM linear normaliser.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    head = head.to(dev)
    head.eval()

    def _fn(audio: np.ndarray, sr: int) -> float:
        emb = backbone.embed(audio, sr)
        x = torch.from_numpy(emb).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred = head(x)[target]
        return float(calibrator(float(pred.item())))

    return _fn
