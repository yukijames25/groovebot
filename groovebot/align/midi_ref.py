"""groovebot.align.midi_ref — build an alignment reference from a MIDI file.

For the DAMP-S-AG MIDI route (spec §9.x DAMP): instead of running
librosa.beat on a (possibly hard-to-decode) backing track, we pull
beats + melody straight from the song's reference MIDI. This is even
cleaner than the backing-audio path — no codec / ffmpeg dependency, and
the beat grid comes from the score itself.

Output mirrors what `DampReferenceBundle` carries, so the runner stays
oblivious to where the reference came from.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MidiReference:
    """MIDI-derived reference for offline alignment.

    `melody` is the one-hot dominant pitch-class chroma per frame (matches
    the (12, T) shape produced by `pyin -> f0_to_pitch_chroma`, so the same
    `OfflineDTWAligner` cost path runs against either).
    `chroma_template` is a column-L2-normalised mixture of every sounding
    pitch class — closer to what `librosa.feature.chroma_cqt` produces on
    polyphonic audio, useful when the query is a full vocal stack rather
    than a single voice.
    """
    beats: np.ndarray            # beat times (sec)
    downbeats: np.ndarray        # downbeat times (sec); may be empty
    melody: np.ndarray           # (12, T) one-hot dominant pitch class
    chroma_template: np.ndarray  # (12, T) column-L2-normalised chroma
    sample_rate: int             # frame-rate denominator (matches aligner)
    hop_length: int              # frame-rate denominator (matches aligner)
    tempo: float                 # representative BPM (median if variable)


def load_reference_from_midi(
    midi_path: str | Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
) -> MidiReference:
    """Parse `midi_path` and rasterize notes into (12, T) frame matrices."""
    import pretty_midi  # lazy: pretty_midi is on the experiments profile

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = np.asarray(pm.get_beats(), dtype=float)
    try:
        downbeats = np.asarray(pm.get_downbeats(), dtype=float)
    except Exception:
        # pretty_midi can fail on free-meter / no-time-signature MIDIs.
        # The runner only needs beats; downbeats are optional.
        downbeats = np.empty(0, dtype=float)

    # Representative tempo. estimate_tempo() falls back on 120 BPM when the
    # input has too few notes — fine for our purposes (it's only logged).
    try:
        tempo = float(pm.estimate_tempo())
    except Exception:
        tempo = 0.0

    melody, chroma_template = _rasterize_notes(pm, sample_rate, hop_length)
    return MidiReference(
        beats=beats,
        downbeats=downbeats,
        melody=melody,
        chroma_template=chroma_template,
        sample_rate=sample_rate,
        hop_length=hop_length,
        tempo=tempo,
    )


def _rasterize_notes(
    pm,  # pretty_midi.PrettyMIDI
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project every (non-drum) note onto (12, T) pitch-class frames.

    Returns `(melody_one_hot, chroma_template_norm)`. Frames cover [0, end_time]
    at the aligner's frame rate. Each note adds 1 to its pitch class on every
    frame it spans; `melody` picks the per-frame argmax; `chroma_template`
    normalises each column to L2 norm 1 (zero columns stay zero).
    """
    frame_rate = sample_rate / hop_length
    duration = pm.get_end_time()
    T = max(1, int(np.ceil(duration * frame_rate)) + 1)
    template = np.zeros((12, T), dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start_f = max(0, int(np.floor(note.start * frame_rate)))
            end_f = min(T, int(np.ceil(note.end * frame_rate)))
            if end_f <= start_f:
                continue
            pc = int(note.pitch) % 12
            template[pc, start_f:end_f] += 1.0

    # One-hot dominant pitch class per voiced frame.
    melody = np.zeros_like(template)
    voiced = template.sum(axis=0) > 0
    if voiced.any():
        argmax_pc = template.argmax(axis=0)
        idx_t = np.nonzero(voiced)[0]
        melody[argmax_pc[idx_t], idx_t] = 1.0

    # L2-normalize columns -> chroma-style template.
    norms = np.linalg.norm(template, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    chroma_template = (template / norms).astype(np.float32)

    return melody, chroma_template
