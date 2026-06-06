"""BeatTrackerPerception wrapper.

We split the tests by whether BeatNet is importable. Either way, the wrapper's
own helpers and its no-BeatNet error path must work.
"""
from __future__ import annotations
import os
import tempfile

import numpy as np
import pytest

from groovebot.perception import BeatEvent, BeatTrackerPerception
from groovebot.perception.beat_tracker import (
    _to_beat_events,
    _tempo_bpm_from_events,
)


def test_to_beat_events_marks_downbeats_correctly():
    raw = np.array([[0.0, 1], [0.5, 2], [1.0, 3], [1.5, 4], [2.0, 1]])
    events = _to_beat_events(raw)
    assert [e.beat_in_bar for e in events] == [1, 2, 3, 4, 1]
    assert [e.is_downbeat for e in events] == [True, False, False, False, True]
    assert events[2].time == 1.0


def test_to_beat_events_handles_none_and_empty():
    assert _to_beat_events(None) == []
    assert _to_beat_events(np.zeros((0, 2))) == []


def test_tempo_bpm_recovers_known_grid():
    events = _to_beat_events(np.array([[t, 1] for t in np.arange(0, 4.0, 0.5)]))
    assert _tempo_bpm_from_events(events) == pytest.approx(120.0)


def test_tempo_bpm_returns_none_for_too_few_events():
    assert _tempo_bpm_from_events([]) is None
    assert _tempo_bpm_from_events(
        [BeatEvent(time=0.0, beat_in_bar=1, is_downbeat=True)]
    ) is None


def test_wrapper_without_beatnet_raises_clean_runtime_error():
    """If BeatNet is absent, asking the wrapper to do work fails with a
    RuntimeError carrying the install hint — not ModuleNotFoundError."""
    pytest.importorskip  # always present, just a guard for typo regressions
    try:
        import BeatNet  # noqa: F401
        pytest.skip("BeatNet is installed; this case verifies the missing-dep path")
    except Exception:
        pass

    trk = BeatTrackerPerception()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(RuntimeError, match="BeatNet"):
            trk.process_wav(path)
    finally:
        try: os.unlink(path)
        except OSError: pass


def test_wrapper_against_real_beatnet_on_synthetic_click():
    """If BeatNet is installed (e.g. on Colab/Kaggle CI), verify it actually
    finds beats in a synthetic 120 BPM click track. Skips locally."""
    pytest.importorskip("BeatNet")
    import soundfile as sf
    from tools.eval_beat import synth_click_wav, click_grid, score_beats

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "click.wav")
        synth_click_wav(wav_path, bpm=120.0, seconds=8.0, with_vocal=False)

        trk = BeatTrackerPerception(sample_rate=22050)
        events = trk.process_wav(wav_path)
        assert len(events) > 0
        est = np.array([e.time for e in events], dtype=float)

        wav, sr = sf.read(wav_path)
        gt = click_grid(120.0, len(wav) / sr)
        s = score_beats("click_120", 120.0, gt, est,
                        audio_sec=len(wav)/sr, proc_sec=1.0)
        # Pure click @ constant BPM is a generous case — BeatNet should sail
        # past F=0.8 on this even with conservative thresholds.
        assert s.f_measure > 0.8
