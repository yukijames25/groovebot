"""experiments.run_m0p_t2 — discovery + classify + aggregate + end-to-end."""
from __future__ import annotations
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from experiments.run_m0p_t2 import (
    SongInputs,
    aggregate,
    classify_rendition_kind,
    discover_songs,
    run_pipeline,
    run_song,
    save_csv,
)
from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.reference import build_reference


# --------------------------------------------------------------------------- #
# Synthetic fixtures — same arpeggio shape used in Tier 1 tests.
# --------------------------------------------------------------------------- #
def _sine_arpeggio(sr: int = 22050, n_beats: int = 20,
                   beat_period: float = 0.5) -> np.ndarray:
    freqs = (261.63, 329.63, 392.00, 523.25)
    n = int(n_beats * beat_period * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    ramp = 0.05
    for i in range(n_beats):
        f = freqs[i % len(freqs)]
        seg = (t >= i * beat_period) & (t < (i + 1) * beat_period)
        local = t[seg] - i * beat_period
        env = np.minimum(local / ramp, 1.0) * \
              np.minimum((beat_period - local) / ramp, 1.0)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        out[seg] = (np.sin(2 * np.pi * f * local) * env).astype(np.float32)
    return out


def _make_song_dir(d: Path, rate: float = 1.05,
                   sr: int = 22050, n_beats: int = 20) -> Path:
    """Drop a complete Tier-2 song dir at `d` and return it.

    - original.wav = arpeggio at rate=1.0
    - original.beats = beats at every beat_period
    - rendition_sing.wav = arpeggio + low noise, time-stretched by `rate`
    - rendition_hum.wav = same time-stretched arpeggio (monophonic; pyin OK)
    - gt.beats = warped beat times (t / rate)
    """
    d.mkdir(parents=True, exist_ok=True)
    beat_period = 0.5
    orig = _sine_arpeggio(sr=sr, n_beats=n_beats, beat_period=beat_period)
    sf.write(str(d / "original.wav"), orig, sr)
    np.savetxt(str(d / "original.beats"),
               np.arange(n_beats) * beat_period, fmt="%.6f")

    sing = librosa.effects.time_stretch(orig.astype(np.float32), rate=rate)
    rng = np.random.default_rng(0)
    sing = sing + 0.005 * rng.standard_normal(len(sing)).astype(np.float32)
    sf.write(str(d / "rendition_sing.wav"), sing.astype(np.float32), sr)

    # Humming rendition: same time-stretched signal (clean F0).
    hum = librosa.effects.time_stretch(orig.astype(np.float32), rate=rate)
    sf.write(str(d / "rendition_hum.wav"), hum.astype(np.float32), sr)

    # GT beats are the warped reference beats.
    gt = (np.arange(n_beats) * beat_period) / rate
    np.savetxt(str(d / "gt.beats"), gt, fmt="%.6f")
    return d


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_classify_rendition_kind():
    assert classify_rendition_kind("rendition_hum.wav") == "pitch"
    assert classify_rendition_kind("rendition_humming.wav") == "pitch"
    assert classify_rendition_kind("rendition_sing.wav") == "chroma"
    assert classify_rendition_kind("rendition_vocal.wav") == "chroma"
    assert classify_rendition_kind("rendition_anonymous.wav") == "chroma"
    assert classify_rendition_kind("RENDITION_HUM.WAV") == "pitch"   # case-insensitive


def test_discover_songs_filters_incomplete(tmp_path):
    full = tmp_path / "complete"
    _make_song_dir(full, n_beats=8)   # has everything

    missing = tmp_path / "missing_renditions"
    missing.mkdir()
    (missing / "original.wav").write_bytes(b"\0")
    (missing / "original.beats").write_text("0.0\n")
    (missing / "gt.beats").write_text("0.0\n")
    # no rendition_*.wav -> filtered out

    no_orig = tmp_path / "no_original"
    no_orig.mkdir()
    (no_orig / "rendition_sing.wav").write_bytes(b"\0")
    (no_orig / "gt.beats").write_text("0.0\n")

    songs = discover_songs(tmp_path)
    assert len(songs) == 1
    assert songs[0].song_dir == full
    assert {p.name for p in songs[0].renditions} == {
        "rendition_sing.wav", "rendition_hum.wav",
    }


def test_discover_songs_empty_root_returns_empty(tmp_path):
    assert discover_songs(tmp_path / "does_not_exist") == []
    assert discover_songs(tmp_path) == []


def test_aggregate_empty():
    per_kind, per_song, overall = aggregate([])
    assert per_kind == []
    assert per_song == []
    assert overall["n"] == 0


def test_aggregate_groups_by_kind_and_song():
    rows = [
        {"song": "sA", "feature_kind": "chroma", "f_measure": 0.8,
         "cmlt": 0.7, "amlt": 0.75, "rt_factor": 0.1},
        {"song": "sA", "feature_kind": "pitch",  "f_measure": 0.5,
         "cmlt": 0.5, "amlt": 0.6, "rt_factor": 0.2},
        {"song": "sB", "feature_kind": "chroma", "f_measure": 0.9,
         "cmlt": 0.8, "amlt": 0.85, "rt_factor": 0.1},
    ]
    per_kind, per_song, overall = aggregate(rows)
    kind_n = {p["feature_kind"]: p["n"] for p in per_kind}
    assert kind_n == {"chroma": 2, "pitch": 1}
    song_n = {p["song"]: p["n"] for p in per_song}
    assert song_n == {"sA": 2, "sB": 1}
    assert overall["n"] == 3


def test_save_csv_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "out.csv"
    save_csv(rows, p)
    with open(p, "r", encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    assert [r["a"] for r in loaded] == ["1", "2"]


# --------------------------------------------------------------------------- #
# End-to-end (synthetic stand-in; Demucs bypassed by pre-built bundle)
# --------------------------------------------------------------------------- #
def test_run_song_with_prebuilt_bundle(tmp_path):
    """End-to-end on one song: bypass Demucs by passing a pre-built bundle.

    The original IS the vocal stem (mono sinusoid), so `build_reference`
    with `vocal_audio=original` produces a valid Tier-2 bundle without
    touching Demucs."""
    song_dir = _make_song_dir(tmp_path / "synth", rate=1.05, n_beats=20)
    out_dir = tmp_path / "work"
    out_dir.mkdir()

    sr = 22050
    audio, _ = sf.read(str(song_dir / "original.wav"), dtype="float32")
    beats = np.loadtxt(str(song_dir / "original.beats"))
    bundle = build_reference(audio, sr, beats, hop_length=512,
                             vocal_audio=audio)

    inputs = SongInputs(
        song_dir=song_dir,
        original_wav=song_dir / "original.wav",
        original_beats=song_dir / "original.beats",
        gt_beats=song_dir / "gt.beats",
        renditions=sorted(song_dir.glob("rendition_*.wav")),
    )
    aligner = OfflineDTWAligner(sample_rate=sr, hop_length=512)
    rows = run_song(inputs, out_dir, aligner, bundle=bundle, make_png=True)

    # Two renditions: sing + hum
    by_kind = {r["feature_kind"]: r for r in rows}
    assert set(by_kind) == {"chroma", "pitch"}
    # Both should produce non-trivial F. Threshold is wide on purpose —
    # this is a wiring smoke test, not a SOTA benchmark.
    assert by_kind["chroma"]["f_measure"] >= 0.4, by_kind["chroma"]
    assert by_kind["pitch"]["f_measure"]  >= 0.3, by_kind["pitch"]

    pngs = sorted(out_dir.glob("*.png"))
    assert {p.name for p in pngs} == {
        "synth_rendition_sing.png", "synth_rendition_hum.png",
    }


def test_run_pipeline_via_monkeypatched_demucs(tmp_path, monkeypatch):
    """Full pipeline test — discovery + run_song + aggregate + CSV writing.

    We monkeypatch `demucs_vocal` to be the identity function so the
    runner's `build_reference()` call works without Demucs."""
    song_dir = _make_song_dir(tmp_path / "synth", rate=1.05, n_beats=20)
    out_dir = tmp_path / "work"

    monkeypatch.setattr(
        "groovebot.align.reference.demucs_vocal",
        lambda audio, sr: np.asarray(audio, dtype=np.float32),
    )

    rows, per_kind, per_song, overall = run_pipeline(
        root=tmp_path,
        out_dir=out_dir,
        sample_rate=22050,
        hop_length=512,
        make_png=True,
        verbose=False,
    )

    assert len(rows) == 2
    assert (out_dir / "m0p_t2_per_rendition.csv").exists()
    assert (out_dir / "m0p_t2_per_kind.csv").exists()
    assert (out_dir / "m0p_t2_per_song.csv").exists()
    assert (out_dir / "m0p_t2_overall.csv").exists()
    assert len(list(out_dir.glob("*.png"))) == 2
    assert overall["n"] == 2
    assert overall["f_mean"] >= 0.3   # both renditions contribute, wide guardrail
