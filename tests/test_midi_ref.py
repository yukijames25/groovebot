"""groovebot.align.midi_ref — MIDI-derived reference shape + content."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from groovebot.align.midi_ref import (
    MidiReference,
    load_reference_from_midi,
)


def _write_arpeggio_midi(
    out: Path,
    *,
    pitches=(60, 64, 67, 72),   # C4 E4 G4 C5
    beat_period: float = 0.5,
    n_beats: int = 16,
) -> Path:
    """Write a single-instrument MIDI: one note per beat, pitches cycling."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_beats):
        p = pitches[i % len(pitches)]
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=p,
            start=i * beat_period,
            end=(i + 1) * beat_period,
        ))
    pm.instruments.append(inst)
    pm.write(str(out))
    return out


def test_load_reference_from_midi_returns_expected_shapes(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=16)
    ref = load_reference_from_midi(midi, sample_rate=22050, hop_length=512)
    assert isinstance(ref, MidiReference)
    assert ref.sample_rate == 22050
    assert ref.hop_length == 512
    assert ref.melody.shape[0] == 12
    assert ref.chroma_template.shape[0] == 12
    assert ref.melody.shape[1] == ref.chroma_template.shape[1]
    assert ref.melody.shape[1] > 0


def test_beats_match_pretty_midi_get_beats(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=16)
    ref = load_reference_from_midi(midi)
    # pretty_midi defaults to 120 BPM when nothing else is set, so get_beats
    # returns evenly spaced beats. Our wrapper must pass them through verbatim.
    pm = pretty_midi.PrettyMIDI(str(midi))
    np.testing.assert_allclose(ref.beats, pm.get_beats(), rtol=0, atol=1e-9)


def test_melody_is_one_hot_per_voiced_frame(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=8)
    ref = load_reference_from_midi(midi)
    # Each frame is either all-zero (silence) or exactly one pitch class on.
    sums = ref.melody.sum(axis=0)
    assert ((sums == 0) | (sums == 1)).all()
    # Some frames should be active (we have notes).
    assert (sums == 1).any()


def test_chroma_template_columns_unit_norm_or_zero(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=8)
    ref = load_reference_from_midi(midi)
    norms = np.linalg.norm(ref.chroma_template, axis=0)
    nonzero = norms > 0
    if nonzero.any():
        np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-5)
    # All-zero columns are allowed (silence) and should stay zero.


def test_pitch_classes_match_midi_notes(tmp_path):
    """Verify the dominant pitch class in each note's window matches the
    note's MIDI pitch (mod 12)."""
    pitches = (60, 64, 67, 72)
    beat_period = 0.5
    n_beats = 8
    midi = _write_arpeggio_midi(
        tmp_path / "arp.mid",
        pitches=pitches, beat_period=beat_period, n_beats=n_beats,
    )
    ref = load_reference_from_midi(midi, sample_rate=22050, hop_length=512)
    frame_rate = ref.sample_rate / ref.hop_length
    for i in range(n_beats):
        mid_frame = int((i * beat_period + beat_period / 2) * frame_rate)
        if mid_frame >= ref.melody.shape[1]:
            break
        expected_pc = pitches[i % len(pitches)] % 12
        col = ref.melody[:, mid_frame]
        assert col[expected_pc] == 1.0, (
            f"beat {i}: melody[{expected_pc}] should be 1 at frame {mid_frame}"
        )


def test_load_reference_from_str_path_works(tmp_path):
    """Accept both Path and str for the midi_path argument."""
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=4)
    ref = load_reference_from_midi(str(midi))
    assert ref.melody.shape[0] == 12
