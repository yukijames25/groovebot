"""Perception subsystem: voice → (beat_pos, downbeat, tempo, arousal, …).

Stays brain-side: no body/sim imports. Heavy ML deps (BeatNet, torch) are
imported lazily inside the classes that need them, so importing this package
does not pull torch on machines that only run the M1 demo.
"""
from .beat_tracker import BeatEvent, BeatTrackerPerception

__all__ = ["BeatEvent", "BeatTrackerPerception"]
