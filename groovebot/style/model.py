"""groovebot.style.model — small CNN with genre + mood multi-head.

v1 scope: classify genre (10-way, GTZAN vocabulary) and mood (6-way) from a
log-mel spectrogram. Tempo and arousal are computed by heuristic in
`attributes.py`, not by this network. The head dict (`self.heads`) is a
`ModuleDict` so future revisions can register a `"tempo"` regression head
or an `"arousal"` regression head without changing the call sites
(`select.py`).

PyTorch is used so we have one trainable framework for both this and the
M3 generator. Keep the model small: it must train on CPU in minutes for
the local feasibility check.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)

MOODS = ("aggressive", "happy", "sad", "calm", "dark", "epic")


class StyleCNN(nn.Module):
    """4-block conv stack + adaptive pool + per-head linear classifier.

    Backbone: Conv-BN-ReLU-MaxPool x4, channel widths 16/32/64/hidden.
    Adaptive average pool to 1x1 so the time dim is variable at inference
    (one 5 s clip, one 10 s clip, and a synthetic test tensor can all share
    the same model).
    """

    def __init__(
        self,
        n_mels: int = 64,
        n_genres: int = len(GENRES),
        n_moods: int = len(MOODS),
        hidden: int = 128,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.backbone = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, hidden),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleDict({
            "genre": nn.Linear(hidden, n_genres),
            "mood": nn.Linear(hidden, n_moods),
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x` shape: (B, 1, n_mels, T) or (B, n_mels, T).

        Returns a dict of logits keyed by head name. Use
        `predict_probs(audio_feat)` on the selector side for the
        softmaxed view.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                f"expected (B,1,n_mels,T) or (B,n_mels,T), got shape {tuple(x.shape)}"
            )
        h = self.backbone(x)
        return {name: head(h) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_probs(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Same call as `forward` but returns softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        return {name: F.softmax(logit, dim=-1) for name, logit in logits.items()}


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )
