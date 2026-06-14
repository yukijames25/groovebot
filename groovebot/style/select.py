"""groovebot.style.select — top-level GrooveStyleSelector.

Wires the pieces together:

    audio (startup 5-10 s window)
        ├── features.log_mel_spectrogram ─► StyleCNN ─► (genre_probs, mood_probs)
        ├── attributes.estimate_tempo    ─► tempo BPM
        └── attributes.estimate_arousal  ─► arousal 0..1  ─► arousal_bucket
                                                                │
                                table.select_move(genre,arousal_bucket,mood_probs)
                                                                │
                                                                ▼
                                                          GrooveStyle
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from groovebot.style.attributes import (
    arousal_bucket,
    estimate_arousal,
    estimate_tempo,
)
from groovebot.style.features import (
    DEFAULT_N_MELS,
    DEFAULT_SR,
    log_mel_spectrogram,
)
from groovebot.style.model import GENRES, MOODS, StyleCNN
from groovebot.style.table import select_move


@dataclass
class GrooveStyle:
    """Text-label output of `GrooveStyleSelector`.

    `move` is one of `groovebot.style.table.MOVES`. `intensity` is the
    0..1 "how big / how fast" scalar from the table. The remaining fields
    expose the underlying attributes so the downstream renderer (M2) and
    learned generator (M3) can use whichever signal they want.
    """
    move: str
    intensity: float
    genre: str
    mood: str           # argmax mood label (label only — the table used soft probs)
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
    """End-to-end style selector. v1: text-label output only.

    Holds one `StyleCNN` and runs feature → model → table on each call.
    Stateless across calls (the selector is meant to be invoked once per
    song, during the startup window).
    """

    def __init__(
        self,
        model: StyleCNN | None = None,
        *,
        target_sr: int = DEFAULT_SR,
        n_mels: int = DEFAULT_N_MELS,
        device: str | torch.device | None = None,
    ):
        self.target_sr = int(target_sr)
        self.n_mels = int(n_mels)
        if model is None:
            model = StyleCNN(n_mels=self.n_mels)
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def select(self, audio: np.ndarray, sr: int) -> GrooveStyle:
        """Pick a GrooveStyle for the given startup window."""
        mel = log_mel_spectrogram(
            audio, sr,
            target_sr=self.target_sr, n_mels=self.n_mels,
        )
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        probs = self.model.predict_probs(x)
        genre_probs_arr = probs["genre"].squeeze(0).cpu().numpy()
        mood_probs_arr = probs["mood"].squeeze(0).cpu().numpy()
        genre_probs = {g: float(p) for g, p in zip(GENRES, genre_probs_arr)}
        mood_probs = {m: float(p) for m, p in zip(MOODS, mood_probs_arr)}

        genre = max(genre_probs, key=genre_probs.get)
        mood_argmax = max(mood_probs, key=mood_probs.get)

        tempo_bpm = estimate_tempo(audio, sr)
        arousal_val = estimate_arousal(audio, sr)
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
