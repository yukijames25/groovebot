"""
types.py — the data contract between brain and body (spec §5.1).

These are the only types the brain hands across module boundaries:
- GrooveContext: what the perception side knows right now.
- JointCommand : what the generator decides the body should do.

Keep this file dependency-free (stdlib + typing only). numpy is referenced as a
string annotation so importing this module does not require numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np  # only for type hints; not required at runtime


@dataclass
class GrooveContext:
    beat_pos: float                                # musical position in beats
    downbeat: bool                                 # is this the bar's downbeat?
    tempo: float                                   # BPM
    arousal: float                                 # 0..1
    valence: float                                 # -1..1
    energy: float                                  # 0..1 instantaneous envelope
    embedding: Optional["np.ndarray"] = None       # M3 voice embedding


@dataclass
class JointCommand:
    targets: dict[str, float] = field(default_factory=dict)
