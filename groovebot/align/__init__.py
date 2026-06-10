"""groovebot.align — offline alignment primitives for M0' Tier 1 (spec §9.x).

This package is intentionally librosa-only: no madmom, no torch, no Demucs.
It runs on the Windows dev laptop so the alignment feasibility check is local
and fast. Online alignment (M2's `ReferenceAligner`) will share the feature
pipeline in `features.py` but use a different aligner (online DTW /
score-following), see spec §14.
"""
from groovebot.align.dtw_align import OfflineDTWAligner, map_reference_beats
from groovebot.align.features import (
    consensus_f0,
    extract_align_features,
    f0_to_pitch_chroma,
    pitch_contour_feature,
    pyin_f0,
    trim_silence,
)
from groovebot.align.midi_ref import MidiReference, load_reference_from_midi
from groovebot.align.reference import ReferenceBundle, build_reference

__all__ = [
    "MidiReference",
    "OfflineDTWAligner",
    "ReferenceBundle",
    "build_reference",
    "consensus_f0",
    "extract_align_features",
    "f0_to_pitch_chroma",
    "load_reference_from_midi",
    "map_reference_beats",
    "pitch_contour_feature",
    "pyin_f0",
    "trim_silence",
]
