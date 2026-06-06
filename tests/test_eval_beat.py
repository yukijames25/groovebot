"""tools/eval_beat.py scoring harness — covered without BeatNet.

We feed the scoring layer hand-crafted "tracker outputs" so we can pin down
exactly what F-measure / CMLt / AMLt should do at the values we care about
(perfect, jittered-within-window, half-tempo, double-tempo, empty).

Also verifies the synth-WAV fixture round-trips through soundfile and the
overlay PNG actually writes a non-empty file.
"""
from __future__ import annotations
import os
import tempfile

import numpy as np
import soundfile as sf

from tools.eval_beat import (
    click_grid,
    plot_overlay,
    score_beats,
    synth_click_wav,
)


def test_click_grid_count_and_spacing():
    gt = click_grid(120.0, 8.0)
    # 120 BPM == one beat every 0.5 s. 8 s -> 17 beats (including t=0).
    assert len(gt) == 17
    assert np.allclose(np.diff(gt), 0.5)


def test_score_perfect_estimate_scores_1():
    gt = click_grid(120.0, 8.0)
    s = score_beats("p", 120.0, gt, gt, audio_sec=8.0, proc_sec=1.0)
    assert s.f_measure == 1.0
    assert s.cmlt == 1.0
    assert s.amlt == 1.0
    assert s.rt_factor == 0.125


def test_score_30ms_jitter_stays_near_1_for_f_but_drops_continuity():
    """mir_eval's default F-window is 70 ms; 30 ms RMS jitter should leave F
    high (>=0.9) but break continuity (CMLt/AMLt) somewhere."""
    rng = np.random.default_rng(0)
    gt = click_grid(120.0, 8.0)
    est = gt + rng.normal(0, 0.03, len(gt))
    s = score_beats("j", 120.0, gt, est, audio_sec=8.0, proc_sec=1.0)
    assert s.f_measure >= 0.9
    assert s.cmlt < 1.0


def test_score_half_tempo_separates_cmlt_from_amlt():
    """Half-tempo tracker: CMLt should fall (wrong grid), AMLt should hold
    (allows tempo halving)."""
    gt = click_grid(120.0, 8.0)
    est = gt[::2]
    s = score_beats("h", 120.0, gt, est, audio_sec=8.0, proc_sec=1.0)
    assert s.cmlt == 0.0
    assert s.amlt == 1.0


def test_score_empty_estimate_is_zero_not_crash():
    gt = click_grid(120.0, 8.0)
    s = score_beats("e", 120.0, gt, np.array([]), audio_sec=8.0, proc_sec=1.0)
    assert s.f_measure == 0.0
    assert s.cmlt == 0.0
    assert s.amlt == 0.0


def test_synth_wav_roundtrip_and_plot_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "click.wav")
        png_path = os.path.join(tmp, "overlay.png")
        synth_click_wav(wav_path, bpm=120.0, seconds=4.0, with_vocal=True)
        wav, sr = sf.read(wav_path, dtype="float32")
        assert sr == 22050
        # ~4 s of audio
        assert abs(len(wav) / sr - 4.0) < 0.05
        # Energy should be non-trivial (clicks + tone).
        assert float(np.mean(wav ** 2)) > 1e-4

        gt = click_grid(120.0, len(wav) / sr)
        est = gt + 0.01     # 10 ms uniform offset
        plot_overlay(wav, sr, gt, est, png_path, title="smoke")
        assert os.path.getsize(png_path) > 0
