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
from dataclasses import dataclass, field
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
    `pitch_contour` is the continuous (2, T) feature that pairs with
    `pitch_contour_feature(pyin_f0(...))`: row 0 is the key-normalised
    MIDI semitone of the highest-sounding (melody-on-top) note per frame,
    row 1 is the voicing channel. Designed for the continuous-pitch DTW
    fix to the DAMP-S-AG diagnostic (octave-folded, voicing-asymmetric
    one-hot caused the pitch path to wander).
    """
    beats: np.ndarray            # beat times (sec)
    downbeats: np.ndarray        # downbeat times (sec); may be empty
    melody: np.ndarray           # (12, T) one-hot dominant pitch class
    chroma_template: np.ndarray  # (12, T) column-L2-normalised chroma
    pitch_contour: np.ndarray    # (2, T) continuous semitone + voicing
    sample_rate: int             # frame-rate denominator (matches aligner)
    hop_length: int              # frame-rate denominator (matches aligner)
    tempo: float                 # representative BPM (median if variable)
    note_onsets: np.ndarray = field(  # sorted note-on times (sec)
        default_factory=lambda: np.empty(0, dtype=float)
    )


def load_reference_from_midi(
    midi_path: str | Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    pitch_voicing_weight: float = 5.0,
) -> MidiReference:
    """Parse `midi_path` and rasterize notes into frame matrices."""
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
    pitch_contour = _rasterize_pitch_contour(
        pm, sample_rate, hop_length,
        voicing_weight=pitch_voicing_weight,
    )
    note_onsets = _collect_note_onsets(pm)
    return MidiReference(
        beats=beats,
        downbeats=downbeats,
        melody=melody,
        chroma_template=chroma_template,
        pitch_contour=pitch_contour,
        sample_rate=sample_rate,
        hop_length=hop_length,
        tempo=tempo,
        note_onsets=note_onsets,
    )


def _collect_note_onsets(pm) -> np.ndarray:
    """Sorted, deduped note-on times across non-drum instruments.

    Distinct from `beats`: `beats` is the score's metric grid (used as GT for
    scoring) while `note_onsets` is the audible attack times of melody/harmony
    notes. The origin-anchor uses these *without ever looking at the beat grid*
    so the calibration stays GT-non-referenced (see `groovebot.align.origin`).
    """
    onsets = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onsets.append(float(note.start))
    if not onsets:
        return np.empty(0, dtype=float)
    arr = np.asarray(sorted(set(onsets)), dtype=float)
    return arr


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


def _rasterize_pitch_contour(
    pm,  # pretty_midi.PrettyMIDI
    sample_rate: int,
    hop_length: int,
    *,
    voicing_weight: float,
) -> np.ndarray:
    """Project the melody-on-top MIDI pitch onto a (2, T) continuous-semitone
    feature paired with `features.pitch_contour_feature` on the query side.

    For each frame we take the *highest* sounding (non-drum) MIDI pitch as
    the melody — DAMP-S-AG MIDIs are essentially monophonic for the lead
    voice, and the melody-on-top heuristic is robust to incidental
    harmonisation. Row 0 is the key-normalised MIDI semitone (median
    subtracted across voiced frames); row 1 is `voicing_weight` on voiced
    frames and 0 on rests, so query/reference voicing asymmetry costs the
    same on either side.
    """
    frame_rate = sample_rate / hop_length
    duration = pm.get_end_time()
    T = max(1, int(np.ceil(duration * frame_rate)) + 1)
    contour = np.full(T, np.nan, dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start_f = max(0, int(np.floor(note.start * frame_rate)))
            end_f = min(T, int(np.ceil(note.end * frame_rate)))
            if end_f <= start_f:
                continue
            existing = contour[start_f:end_f]
            # Melody-on-top: keep the highest pitch on each frame.
            new = np.where(
                np.isnan(existing) | (existing < float(note.pitch)),
                float(note.pitch), existing,
            )
            contour[start_f:end_f] = new

    voiced = np.isfinite(contour)
    out = np.zeros((2, T), dtype=np.float32)
    if voiced.any():
        midi_pitch = contour[voiced].astype(np.float64)
        midi_pitch = midi_pitch - float(np.median(midi_pitch))
        out[0, voiced] = midi_pitch.astype(np.float32)
        out[1, voiced] = float(voicing_weight)
    return out
