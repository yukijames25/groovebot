"""groovebot.style — GrooveStyleSelector (spec §14 module note).

Decides *how* the robot grooves (style) from the song's first 5-10 s window.
Completely independent of the timing/alignment track (`groovebot.align`,
`groovebot.perception`): styles change at song start, timing changes every
control tick. The two pipelines are deliberately decoupled so style work
can progress without DAMP access and timing work can progress without
mood labels.

v1 outputs are plain text labels from a small upper-body-feasible move
vocabulary; no `JointCommand` mapping yet. That bridge belongs to a later
`GrooveGenerator` revision once we know the labels are stable.
"""
from groovebot.style.attributes import (
    arousal_bucket,
    estimate_arousal,
    estimate_tempo,
)
from groovebot.style.features import log_mel_spectrogram
from groovebot.style.model import GENRES, MOODS, StyleCNN
from groovebot.style.select import GrooveStyle, GrooveStyleSelector
from groovebot.style.table import MOVES, select_move

__all__ = [
    "GENRES",
    "GrooveStyle",
    "GrooveStyleSelector",
    "MOODS",
    "MOVES",
    "StyleCNN",
    "arousal_bucket",
    "estimate_arousal",
    "estimate_tempo",
    "log_mel_spectrogram",
    "select_move",
]
