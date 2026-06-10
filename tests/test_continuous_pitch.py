"""Lever 2: continuous semitone pitch feature + MIDI pitch contour + runner.

The diagnostic showed that the 12-D one-hot pitch class feature was
voicing-asymmetric (MIDI 99% active vs query 72-85% active) and
octave-folded — DTW on the pitch path drifted by up to 19 seconds. The
continuous-semitone replacement pins:

  - `pitch_contour_feature` shape, key-normalisation, voicing channel.
  - `MidiReference.pitch_contour` shape + values match the same convention.
  - Runner: `pitch_mode='continuous'` requires MIDI mode (backing has no
    matching continuous reference).
"""
from __future__ import annotations
import io
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from experiments.run_m0p_t2_damp import run_arrangement
from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import pitch_contour_feature
from groovebot.align.midi_ref import load_reference_from_midi
from tools.ingest_damp import DampArrangement, DampRendition


# --------------------------------------------------------------------------- #
# pitch_contour_feature
# --------------------------------------------------------------------------- #
def test_pitch_contour_returns_2_by_T():
    f0 = np.array([440.0, 440.0, np.nan, 261.63, 261.63], dtype=float)
    out = pitch_contour_feature(f0)
    assert out.shape == (2, 5)
    assert out.dtype == np.float32


def test_pitch_contour_key_normalisation_subtracts_median():
    """With key_normalize=True the median voiced semitone is zero."""
    f0 = np.array([440.0, 440.0, 261.63], dtype=float)
    out = pitch_contour_feature(f0, key_normalize=True)
    # Both A4 (~69 MIDI) and C4 (~60 MIDI). Median across voiced frames is
    # around 67 (since A4 appears twice -> median is 69... actually [60, 69, 69]
    # median is 69). Then row 0 after subtraction is [-9, 0, 0] roughly.
    voiced_row = out[0, :]
    assert abs(float(np.median(voiced_row))) < 1.0


def test_pitch_contour_unvoiced_frames_are_zero_both_rows():
    f0 = np.array([np.nan, 440.0, 0.0, 261.63], dtype=float)
    out = pitch_contour_feature(f0)
    # frames 0 and 2 are unvoiced (NaN, then 0 Hz).
    assert out[0, 0] == 0.0 and out[1, 0] == 0.0
    assert out[0, 2] == 0.0 and out[1, 2] == 0.0
    # voiced frames carry the voicing weight.
    assert out[1, 1] > 0.0
    assert out[1, 3] > 0.0


def test_pitch_contour_voicing_weight_tunable():
    f0 = np.array([440.0], dtype=float)
    out_a = pitch_contour_feature(f0, voicing_weight=1.0)
    out_b = pitch_contour_feature(f0, voicing_weight=10.0)
    assert out_a[1, 0] == 1.0
    assert out_b[1, 0] == 10.0


def test_pitch_contour_all_unvoiced_returns_zeros():
    f0 = np.array([np.nan, np.nan, np.nan], dtype=float)
    out = pitch_contour_feature(f0)
    assert (out == 0).all()


def test_pitch_contour_unnormalised_keeps_absolute_pitch():
    f0 = np.array([440.0], dtype=float)
    out = pitch_contour_feature(f0, key_normalize=False)
    # A4 is MIDI 69; key_normalize=False keeps the absolute value.
    assert abs(float(out[0, 0]) - 69.0) < 1.0


# --------------------------------------------------------------------------- #
# MidiReference.pitch_contour
# --------------------------------------------------------------------------- #
def _write_arpeggio_midi(out: Path, *, pitches=(60, 64, 67, 72),
                        beat_period: float = 0.5, n_beats: int = 16) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_beats):
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=pitches[i % len(pitches)],
            start=i * beat_period,
            end=(i + 1) * beat_period,
        ))
    pm.instruments.append(inst)
    pm.write(str(out))
    return out


def test_midi_reference_exposes_pitch_contour(tmp_path):
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=16)
    ref = load_reference_from_midi(midi, sample_rate=22050, hop_length=512)
    assert ref.pitch_contour.shape == (2, ref.melody.shape[1])


def test_midi_pitch_contour_is_key_normalised(tmp_path):
    """For an arpeggio whose pitches are evenly spaced, the median voiced
    semitone after key normalisation should be near zero."""
    midi = _write_arpeggio_midi(
        tmp_path / "arp.mid", pitches=(60, 64, 67, 72), n_beats=16,
    )
    ref = load_reference_from_midi(midi)
    voiced = ref.pitch_contour[1] > 0
    voiced_pitch = ref.pitch_contour[0, voiced]
    assert voiced_pitch.size > 0
    # Median should land within 2 semitones of zero (medians of (60,64,67,72)
    # rasterised at non-uniform durations may shift a bit).
    assert abs(float(np.median(voiced_pitch))) < 2.0


def test_midi_pitch_contour_voicing_channel_matches_melody_voicing(tmp_path):
    """Frames where the (12, T) melody is active should also be voiced in
    the (2, T) pitch_contour (same notes drive both)."""
    midi = _write_arpeggio_midi(tmp_path / "arp.mid", n_beats=8)
    ref = load_reference_from_midi(midi)
    melody_voiced = ref.melody.sum(axis=0) > 0
    contour_voiced = ref.pitch_contour[1] > 0
    # They should agree on the majority of frames (boundary alignment may
    # differ by a frame on each edge of each note).
    agreement = float((melody_voiced == contour_voiced).mean())
    assert agreement >= 0.95


# --------------------------------------------------------------------------- #
# Runner: pitch_mode plumbing
# --------------------------------------------------------------------------- #
def _make_minimal_arrangement(tmp_path: Path) -> DampArrangement:
    """Tiny arrangement: MIDI + 1 short sinusoid vocal — just enough for the
    runner to type-check the pitch_mode='continuous' branch."""
    arr_dir = tmp_path / "arr"
    arr_dir.mkdir()
    midi = _write_arpeggio_midi(arr_dir / "reference.midi",
                                pitches=(60, 64), n_beats=20,
                                beat_period=0.5)
    sr = 22050
    n = int(20 * 0.5 * sr)
    t = np.arange(n) / sr
    audio = np.zeros(n, dtype=np.float32)
    for i in range(20):
        f = 261.63 if i % 2 == 0 else 329.63
        beat_start = i * 0.5
        seg = (t >= beat_start) & (t < beat_start + 0.5)
        local = t[seg] - beat_start
        env = np.minimum(local / 0.05, 1.0) * \
              np.minimum((0.5 - local) / 0.05, 1.0)
        env = np.clip(env, 0, 1).astype(np.float32)
        audio[seg] = (np.sin(2 * np.pi * f * local) * env).astype(np.float32)
    vocal = arr_dir / "vocal_solo.wav"
    sf.write(str(vocal), audio, sr)
    return DampArrangement(
        arrangement_id="arr",
        arrangement_dir=arr_dir,
        backing_wav=None,
        renditions=(DampRendition(rendition_id="solo", vocal_wav=vocal),),
        reference_midi=midi,
    )


def test_runner_continuous_pitch_requires_midi_mode(tmp_path):
    """pitch_mode='continuous' in backing mode should raise — there is no
    matching continuous reference."""
    arr_dir = tmp_path / "arr"
    arr_dir.mkdir()
    backing_path = arr_dir / "backing.wav"
    sf.write(str(backing_path), np.zeros(22050, dtype=np.float32), 22050)
    vocal_path = arr_dir / "vocal_solo.wav"
    sf.write(str(vocal_path), np.zeros(22050, dtype=np.float32), 22050)
    arrangement = DampArrangement(
        arrangement_id="arr",
        arrangement_dir=arr_dir,
        backing_wav=backing_path,
        renditions=(DampRendition(rendition_id="solo", vocal_wav=vocal_path),),
        reference_midi=None,
    )
    aligner = OfflineDTWAligner(sample_rate=22050)
    with pytest.raises(ValueError):
        run_arrangement(
            arrangement, tmp_path / "work", aligner,
            reference_source="backing", pitch_mode="continuous",
        )


def test_runner_continuous_pitch_smoke_in_midi_mode(tmp_path):
    """End-to-end MIDI mode with pitch_mode='continuous'. We just check the
    plumbing — both paths produce rows, the pitch row uses the (2, T)
    feature internally."""
    arrangement = _make_minimal_arrangement(tmp_path)
    aligner = OfflineDTWAligner(sample_rate=22050, hop_length=512, subseq=True)
    rows = run_arrangement(
        arrangement, tmp_path / "work", aligner,
        reference_source="midi",
        pitch_mode="continuous",
        silence_trim=False,
        make_png=False,
    )
    # 1 query x 2 paths = 2 rows.
    assert len(rows) == 2
    assert sorted(r["feature_kind"] for r in rows) == ["chroma", "pitch"]
