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
    build_parser,
    click_grid,
    load_beat_annotation,
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


def test_load_beat_annotation_reads_single_column():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write("0.0\n0.5\n1.0\n1.5\n2.0\n")
        arr = load_beat_annotation(p)
        assert arr.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_load_beat_annotation_handles_comments_blanks_and_extra_columns():
    """The annotation parser must tolerate Ballroom-style multi-column rows
    and the comment / blank lines we'd hand-write in fixtures."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write(
                "# header comment\n"
                "\n"
                "0.000\t1\n"
                "0.500\t2\n"
                "; trailing semicolon comment\n"
                "1.000\t3\n"
                "1.500\t4\n"
                "2.000\t1\n"
            )
        arr = load_beat_annotation(p)
        assert arr.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_load_beat_annotation_sorts_unsorted_input():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write("1.0\n0.5\n2.0\n0.0\n1.5\n")
        arr = load_beat_annotation(p)
        assert arr.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_load_beat_annotation_rejects_garbage():
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write("not_a_number\n")
        with pytest.raises(ValueError):
            load_beat_annotation(p)


def test_score_via_beats_file_perfect():
    """--beats path: a file whose beats match the tracker exactly -> F=1."""
    gt_arr = click_grid(120.0, 8.0)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "perfect.beats")
        with open(p, "w") as f:
            for t in gt_arr:
                f.write(f"{t:.6f}\n")
        gt = load_beat_annotation(p)
        s = score_beats("perfect.wav", bpm=None, gt=gt, est=gt,
                        audio_sec=8.0, proc_sec=1.0)
        assert s.bpm is None
        assert s.f_measure == 1.0
        assert s.cmlt == 1.0


def test_score_via_beats_file_with_jitter_drops_continuity():
    rng = np.random.default_rng(0)
    gt_arr = click_grid(120.0, 8.0)
    est_arr = gt_arr + rng.normal(0, 0.03, len(gt_arr))
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            for t in gt_arr:
                f.write(f"{t:.6f}\n")
        gt = load_beat_annotation(p)
        s = score_beats("track.wav", bpm=None, gt=gt, est=est_arr,
                        audio_sec=8.0, proc_sec=1.0)
        assert s.bpm is None
        assert s.f_measure >= 0.9
        assert s.cmlt < 1.0


def test_argparse_bpm_and_beats_are_mutually_exclusive():
    import pytest
    ap = build_parser()
    # neither -> error
    with pytest.raises(SystemExit):
        ap.parse_args(["eval", "--wav", "x.wav"])
    # both -> error
    with pytest.raises(SystemExit):
        ap.parse_args(["eval", "--wav", "x.wav", "--bpm", "120", "--beats", "b"])
    # either alone -> OK
    args = ap.parse_args(["eval", "--wav", "x.wav", "--bpm", "120"])
    assert args.bpm == 120 and args.beats is None
    args = ap.parse_args(["eval", "--wav", "x.wav", "--beats", "b.txt"])
    assert args.bpm is None and args.beats == "b.txt"


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
