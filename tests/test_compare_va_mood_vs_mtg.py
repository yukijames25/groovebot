"""Smoke tests for experiments/compare_va_mood_vs_mtg.py.

We test the stub mode end-to-end (no audio, no PANNs, no checkpoints
required) plus the small helpers that compute the aggregates.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest

from experiments.compare_va_mood_vs_mtg import (
    accuracy_vs_gt,
    agreement_rate,
    build_parser,
    calm_sad_stability,
    confusion_matrix,
    main,
    per_class_profile,
    synthetic_rows,
)
from groovebot.style.model import MOODS


def test_stub_main_writes_report_and_csv(tmp_path):
    out = tmp_path / "report"
    rc = main(["--synthetic-stub", "--n-stub", "32", "--out-dir", str(out)])
    assert rc == 0
    rep_path = out / "report.json"
    csv_path = out / "per_clip.csv"
    assert rep_path.exists()
    assert csv_path.exists()
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    assert rep["is_stub"] is True
    assert rep["n_clips"] == 32
    assert "agreement_rate" in rep
    assert "calm_sad_stability" in rep
    assert "per_class_profile_by_mtg_mood" in rep
    # confusion matrix is full (MOODS x MOODS)
    cm = rep["confusion_matrix_mtg_rows_va_cols"]
    assert set(cm.keys()) == set(MOODS)
    for row in cm.values():
        assert set(row.keys()) == set(MOODS)
    # CSV header sanity
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][:4] == ["path", "mood_gt", "mtg_mood", "va_mood"]
    assert len(rows) == 33  # 32 stub clips + header


def test_stub_main_with_aux_moods_flag(tmp_path):
    """Including aux moods must be reflected in the report flag."""
    out = tmp_path / "report"
    main(["--synthetic-stub", "--include-aux-moods",
          "--n-stub", "16", "--out-dir", str(out)])
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["include_aux_moods"] is True


def test_main_without_args_fails(tmp_path):
    """Without --synthetic-stub and without the required real-data
    args, the script must reject the run with a non-zero exit."""
    out = tmp_path / "report"
    rc = main(["--out-dir", str(out)])
    assert rc != 0


def test_agreement_rate_basic():
    rows = [
        {"mtg_mood": "happy", "va_mood": "happy"},
        {"mtg_mood": "calm",  "va_mood": "sad"},
        {"mtg_mood": "sad",   "va_mood": "sad"},
    ]
    assert agreement_rate(rows) == pytest.approx(2 / 3)


def test_confusion_matrix_counts_cells():
    rows = [
        {"mtg_mood": "calm", "va_mood": "calm"},
        {"mtg_mood": "calm", "va_mood": "sad"},
        {"mtg_mood": "sad",  "va_mood": "calm"},
    ]
    cm = confusion_matrix(rows, "mtg_mood", "va_mood")
    assert cm["calm"]["calm"] == 1
    assert cm["calm"]["sad"] == 1
    assert cm["sad"]["calm"] == 1


def test_calm_sad_stability_subset():
    rows = [
        {"mtg_mood": "calm", "va_mood": "calm"},
        {"mtg_mood": "sad",  "va_mood": "calm"},
        {"mtg_mood": "happy", "va_mood": "happy"},  # outside the pair
    ]
    s = calm_sad_stability(rows)
    assert s["n_clips_in_pair"] == 2
    # agreement rate over the {calm,sad}^2 subset only
    assert s["agreement_rate"] == pytest.approx(0.5)


def test_accuracy_vs_gt_skips_when_no_gt():
    rows = [
        {"mood_gt": None, "mtg_mood": "calm"},
        {"mood_gt": None, "mtg_mood": "sad"},
    ]
    assert accuracy_vs_gt(rows, "mtg_mood") == {"n": 0, "accuracy": None}


def test_accuracy_vs_gt_with_partial_gt():
    rows = [
        {"mood_gt": "calm", "mtg_mood": "calm"},
        {"mood_gt": "sad",  "mtg_mood": "calm"},
        {"mood_gt": None,   "mtg_mood": "happy"},
    ]
    out = accuracy_vs_gt(rows, "mtg_mood")
    assert out["n"] == 2
    assert out["accuracy"] == 0.5


def test_per_class_profile_aggregates():
    rows = [
        {"mtg_mood": "calm", "arousal_unit": 0.1, "valence_unit": 0.9},
        {"mtg_mood": "calm", "arousal_unit": 0.2, "valence_unit": 0.8},
        {"mtg_mood": "sad",  "arousal_unit": 0.1, "valence_unit": 0.1},
    ]
    p = per_class_profile(rows, "mtg_mood")
    assert p["calm"]["n"] == 2
    assert p["calm"]["a_mean"] == pytest.approx(0.15)
    assert p["calm"]["v_mean"] == pytest.approx(0.85)
    assert p["sad"]["n"] == 1


def test_synthetic_rows_have_expected_shape():
    rows = synthetic_rows(n=10, seed=1)
    assert len(rows) == 10
    for r in rows:
        assert r["mtg_mood"] in MOODS
        assert r["va_mood"] in MOODS
        assert 0.0 <= r["arousal_unit"] <= 1.0
        assert 0.0 <= r["valence_unit"] <= 1.0
        assert set(r["mtg_probs"].keys()) == set(MOODS)
        assert set(r["va_probs"].keys()) == set(MOODS)


def test_parser_smoke():
    p = build_parser()
    ns = p.parse_args(["--synthetic-stub", "--out-dir", "x"])
    assert ns.synthetic_stub is True
    assert ns.out_dir == "x"
