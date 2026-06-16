"""groovebot.style.select — top-level GrooveStyleSelector.

Two model paths, one public contract (`GrooveStyle`):

  * **v1/v2 CNN path** (StyleCNN end-to-end from log-mel):
        log_mel_spectrogram -> StyleCNN -> (genre_probs, mood_probs)
        attributes.estimate_tempo  / arousal -> tempo / bucket
        table.select_move -> GrooveStyle

  * **v3 transfer-learning path** (frozen PANNs CNN14 + MLP heads):
        PannsBackbone.embed -> StyleHead          -> (genre_probs, mood_probs)
                            \\-> StyleRegressionHead -> (arousal, valence)
        (tempo / table identical)

Affect source policy (the spec's "DEAM-learned by default, heuristic
as fallback" rule):

  * If a `regression_head` is wired in, the learned DEAM arousal +
    valence is the default (one backbone embed is shared with the
    classification head — no double work).
  * Else, the v2 `estimate_arousal()` heuristic is the fallback. It
    has Pearson r ≈ 0.42 against DEAM truth, so use it only when PANNs
    is unavailable or the runtime budget cannot fit the embedding.
  * An explicit `arousal_fn` / `valence_fn` always overrides both.

Mood source switch (`mood_source`):

  * `"head"` (default) — mood comes from the MTG-trained classification
    head (or StyleCNN softmax in v1/v2).
  * `"va"` — mood is derived from (arousal, valence) via the
    circumplex map in `mood_from_va.py`. Requires a valence source
    (a `regression_head` or an explicit `valence_fn`). The four V/A
    quadrants (happy/aggressive/calm/sad) are the cleanly-mapped
    classes; epic/dark default to 0 unless `mood_va_prototypes`
    includes their (draft) prototypes — see `mood_from_va.py`.

Build a selector with one of:

  * `GrooveStyleSelector()`         — v1/v2 CNN, random weights
  * `GrooveStyleSelector(model=...)`— v1/v2 CNN, loaded weights
  * `GrooveStyleSelector(backbone=..., head=...)`
        — v3 classification; heuristic arousal fallback
  * `GrooveStyleSelector(backbone=..., head=..., regression_head=...)`
        — v3 with learned DEAM arousal+valence (recommended)
  * `GrooveStyleSelector.from_panns(ckpt, head_weights=...,
        regression_head_weights=...)`
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

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
from groovebot.style.mood_from_va import (
    DEFAULT_QUADRANT_PROTOTYPES,
    MoodPrototype,
    mood_probs_from_va,
)
from groovebot.style.table import select_move


# An affect source is `(audio, sr) -> 0..1`. Both the heuristic
# `estimate_arousal` and the v3 learned head wrappers conform.
ArousalFn = Callable[[np.ndarray, int], float]
ValenceFn = Callable[[np.ndarray, int], float]

MoodSource = Literal["head", "va"]


@dataclass
class GrooveStyle:
    """Text-label output of `GrooveStyleSelector`. Unchanged since v1
    (public contract — call sites in M2 read these fields)."""
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
    """End-to-end style selector. v1/v2 CNN and v3 PANNs paths share
    one selector so M2 call sites do not change across migrations."""

    def __init__(
        self,
        model: StyleCNN | None = None,
        *,
        backbone: PannsBackbone | None = None,
        head: StyleHead | None = None,
        regression_head: StyleRegressionHead | None = None,
        arousal_fn: ArousalFn | None = None,
        valence_fn: ValenceFn | None = None,
        mood_source: MoodSource = "head",
        mood_va_prototypes: Mapping[str, MoodPrototype] = DEFAULT_QUADRANT_PROTOTYPES,
        target_sr: int = DEFAULT_SR,
        n_mels: int = DEFAULT_N_MELS,
        device: str | torch.device | None = None,
    ):
        """`regression_head` co-wires DEAM-learned arousal + valence
        as the default (heuristic stays as fallback when PANNs is
        unavailable). Explicit `arousal_fn` / `valence_fn` override.
        See module docstring for the full source policy."""
        self.target_sr = int(target_sr)
        self.n_mels = int(n_mels)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mood_source: MoodSource = mood_source
        self.mood_va_prototypes = dict(mood_va_prototypes)

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
            self.regression_head = regression_head
            if regression_head is not None:
                regression_head.to(self.device)
                regression_head.eval()
        else:
            self.mode = "cnn"
            if model is None:
                model = StyleCNN(n_mels=self.n_mels)
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            self.backbone = None
            self.head = None
            if regression_head is not None:
                raise ValueError(
                    "GrooveStyleSelector: `regression_head` requires the v3 "
                    "path (`backbone`+`head`); pass them together."
                )
            self.regression_head = None

        # arousal/valence defaults: learned (via regression_head) when
        # the v3 path is wired AND no explicit override was given;
        # otherwise heuristic for arousal and None for valence.
        self.arousal_fn: ArousalFn = arousal_fn or estimate_arousal
        self.valence_fn: ValenceFn | None = valence_fn
        self._arousal_fn_is_default = arousal_fn is None
        self._valence_fn_is_default = valence_fn is None

        if self.mood_source == "va" and self.regression_head is None and self.valence_fn is None:
            raise ValueError(
                "GrooveStyleSelector: mood_source='va' needs a valence "
                "source (pass `regression_head` or `valence_fn`)."
            )

    @classmethod
    def from_panns(
        cls,
        checkpoint_path: str | Path,
        *,
        head_weights: str | Path | None = None,
        regression_head_weights: str | Path | None = None,
        emb_dim: int = EMBEDDING_DIM,
        device: str | torch.device | None = None,
        **selector_kw,
    ) -> "GrooveStyleSelector":
        """Build a v3 selector. `checkpoint_path` is the PANNs CNN14
        ckpt; `head_weights` and `regression_head_weights` are torch
        state dicts (skip them for random heads — useful for smoke
        tests). With `regression_head_weights` set, the selector
        defaults to DEAM-learned arousal + valence."""
        dev = "cpu" if device is None else str(device)
        backbone = PannsBackbone(checkpoint_path, device=dev)
        head = StyleHead(emb_dim=emb_dim)
        if head_weights is not None:
            ck = torch.load(str(head_weights), map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            head.load_state_dict(sd)
        regression_head = None
        if regression_head_weights is not None:
            regression_head = StyleRegressionHead(emb_dim=emb_dim)
            ck = torch.load(str(regression_head_weights), map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            regression_head.load_state_dict(sd)
        return cls(
            backbone=backbone, head=head,
            regression_head=regression_head, device=device,
            **selector_kw,
        )

    # -------------------------------------------------------- predictions

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

    def _predict_head_from_emb(self, emb: np.ndarray) -> tuple[dict, dict]:
        x = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        probs = self.head.predict_probs(x)
        gp = probs["genre"].squeeze(0).cpu().numpy()
        mp = probs["mood"].squeeze(0).cpu().numpy()
        return (
            {g: float(p) for g, p in zip(GENRES, gp)},
            {m: float(p) for m, p in zip(MOODS, mp)},
        )

    def _affect_from_emb(self, emb: np.ndarray) -> tuple[float, float]:
        """Run the regression head on a cached embedding. Returns
        (arousal_unit, valence_unit) in 0..1 after DEAM calibration."""
        x = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.regression_head(x)
            a_raw = float(out["arousal"].item())
            v_raw = float(out["valence"].item())
        return sam_to_unit(a_raw), sam_to_unit(v_raw)

    # ----------------------------------------------------------- main API

    def select(self, audio: np.ndarray, sr: int) -> GrooveStyle:
        if self.mode == "embedding":
            emb = self.backbone.embed(audio, sr)
            genre_probs, head_mood_probs = self._predict_head_from_emb(emb)
            if self.regression_head is not None and self._arousal_fn_is_default:
                arousal_val, valence_val = self._affect_from_emb(emb)
                if not self._valence_fn_is_default:
                    valence_val = float(self.valence_fn(audio, sr))
            else:
                arousal_val = float(self.arousal_fn(audio, sr))
                valence_val = (
                    float(self.valence_fn(audio, sr))
                    if self.valence_fn is not None else None
                )
        else:
            genre_probs, head_mood_probs = self._predict_cnn(audio, sr)
            arousal_val = float(self.arousal_fn(audio, sr))
            valence_val = (
                float(self.valence_fn(audio, sr))
                if self.valence_fn is not None else None
            )

        arousal_val = max(0.0, min(1.0, arousal_val))
        if valence_val is not None:
            valence_val = max(0.0, min(1.0, valence_val))

        if self.mood_source == "va":
            if valence_val is None:
                raise ValueError(
                    "mood_source='va' requires valence; pass `regression_head` "
                    "or `valence_fn` at construction time."
                )
            mood_probs = mood_probs_from_va(
                arousal_val, valence_val,
                prototypes=self.mood_va_prototypes,
            )
        else:
            mood_probs = head_mood_probs

        genre = max(genre_probs, key=genre_probs.get)
        mood_argmax = max(mood_probs, key=mood_probs.get)
        tempo_bpm = estimate_tempo(audio, sr)
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

    Note: when `GrooveStyleSelector` is built with `regression_head=`
    directly, the backbone embedding is shared with the classification
    head (one embed per `select()` call). Use this standalone helper
    only when wiring the regression head outside the selector, since
    each call here does its own backbone embed.
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
