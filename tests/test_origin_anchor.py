"""Lever B: GT-non-referenced origin anchor.

Pins these properties of `estimate_origin_offset`:

  - identity case: when the query's onsets sit exactly on the supplied
    MIDI note-on times, the recovered lag is 0 (within one frame).
  - shifted case: when the query is delayed by Δ seconds, the recovered
    lag is +Δ (within one frame).
  - degenerate inputs (empty audio, empty note-on list, silent audio)
    return 0.0 rather than raising — the runner can fall back to "no
    anchor applied".
  - the calibration never touches the MIDI beat grid — we exercise
    `MidiReference.note_onsets` which is distinct from
    `MidiReference.beats`.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pretty_midi

from groovebot.align.midi_ref import load_reference_from_midi
from groovebot.align.origin import estimate_origin_offset


SR = 22050
HOP = 512


def _click_track(onset_times_sec, *, dur_sec: float, sr: int = SR) -> np.ndarray:
    """A 1 kHz tone burst at each onset time with a fast-decay envelope.

    `librosa.onset.onset_strength` reliably picks these up — good enough
    to exercise the cross-correlator without dragging in real audio.
    """
    n = int(round(dur_sec * sr))
    out = np.zeros(n, dtype=np.float32)
    click_len = int(0.020 * sr)
    env = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)
    tone = (np.sin(2 * np.pi * 1000.0 * np.arange(click_len) / sr)
            .astype(np.float32) * env * 0.5)
    for t in onset_times_sec:
        start = int(round(float(t) * sr))
        end = min(start + click_len, n)
        if 0 <= start < n:
            out[start:end] += tone[: end - start]
    return out


# --------------------------------------------------------------------------- #
# Behavioural pins
# --------------------------------------------------------------------------- #
def test_zero_lag_when_query_clicks_match_midi_onsets():
    onsets = np.arange(0.5, 8.0, 0.5)   # 15 onsets at 0.5 s spacing
    audio = _click_track(onsets, dur_sec=9.0)
    lag = estimate_origin_offset(audio, onsets, sr=SR, hop_length=HOP)
    frame_sec = HOP / SR
    # onset_strength has a ~1 frame spread; tolerance of 2 frames is generous.
    assert abs(lag) <= 2 * frame_sec, f"got lag={lag:.4f}s"


def test_lag_recovers_query_shift():
    """If the query is delayed by Δ sec relative to MIDI, the estimator
    returns +Δ (within frame resolution)."""
    midi_onsets = np.arange(0.5, 8.0, 0.5)
    delta = 0.30
    query_onsets = midi_onsets + delta
    audio = _click_track(query_onsets, dur_sec=9.0)
    lag = estimate_origin_offset(audio, midi_onsets, sr=SR, hop_length=HOP)
    frame_sec = HOP / SR
    assert abs(lag - delta) <= 2 * frame_sec, (
        f"expected ≈{delta:.3f}s, got {lag:.4f}s"
    )


def test_max_lag_clamps_search_window():
    """A delay larger than max_lag_sec gets clamped to the search window
    edge — the estimator never returns lags outside its declared range."""
    midi_onsets = np.arange(0.5, 8.0, 0.5)
    delta = 1.8     # large but inside max_lag_sec=2.0
    audio = _click_track(midi_onsets + delta, dur_sec=12.0)
    lag = estimate_origin_offset(
        audio, midi_onsets, sr=SR, hop_length=HOP, max_lag_sec=0.5,
    )
    # Clamped: the estimator can't return values outside ±0.5 s.
    assert abs(lag) <= 0.5 + (HOP / SR)


def test_empty_audio_returns_zero():
    onsets = np.arange(0.5, 5.0, 0.5)
    audio = np.zeros(0, dtype=np.float32)
    assert estimate_origin_offset(audio, onsets, sr=SR) == 0.0


def test_silent_audio_returns_zero():
    """Silent audio has a flat onset envelope; the estimator should return
    0 rather than picking up noise at the edge of the search window."""
    onsets = np.arange(0.5, 5.0, 0.5)
    silent = np.zeros(int(5.0 * SR), dtype=np.float32)
    assert estimate_origin_offset(silent, onsets, sr=SR) == 0.0


def test_empty_midi_onsets_returns_zero():
    audio = _click_track(np.arange(0.5, 5.0, 0.5), dur_sec=5.5)
    assert estimate_origin_offset(audio, np.empty(0), sr=SR) == 0.0


# --------------------------------------------------------------------------- #
# MidiReference plumbing — note_onsets distinct from beats
# --------------------------------------------------------------------------- #
def _write_arpeggio_midi(out: Path, *, n_beats: int = 8,
                        beat_period: float = 0.5) -> Path:
    pitches = (60, 64, 67, 72)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_beats):
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=pitches[i % len(pitches)],
            start=i * beat_period, end=(i + 1) * beat_period,
        ))
    pm.instruments.append(inst)
    pm.write(str(out))
    return out


def test_midi_reference_exposes_note_onsets_distinct_from_beats(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=8,
                                beat_period=0.5)
    ref = load_reference_from_midi(midi, sample_rate=SR, hop_length=HOP)
    # Note onsets land on the score's beats here (one note per beat), but the
    # array is populated as its own field — runner can use note_onsets for
    # anchor estimation without ever consulting ref.beats.
    assert ref.note_onsets.size == 8
    np.testing.assert_allclose(
        ref.note_onsets, np.arange(8) * 0.5, atol=1e-6,
    )
    # Sanity: identity check on the field, not the value.
    assert id(ref.note_onsets) != id(ref.beats)


def test_midi_reference_note_onsets_are_sorted_and_deduped(tmp_path):
    """Two simultaneous notes at the same start (chord) collapse to one
    onset — anchor xcorr would otherwise double-count the chord."""
    midi_path = tmp_path / "chord.mid"
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=0.5, end=1.0))
    pm.instruments.append(inst)
    pm.write(str(midi_path))
    ref = load_reference_from_midi(midi_path, sample_rate=SR, hop_length=HOP)
    assert list(ref.note_onsets) == [0.0, 0.5]
