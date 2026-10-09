"""Smoke tests for tools/measure_latency.py.

The script's full pipeline needs the PANNs ckpt and the showcase wavs
(neither committed), so these tests cover only the pure helpers and the
per-tick measurement path — the piece that runs against the real
`MujocoBackend` + `StyleGrooveGenerator` without any ckpts.

What we pin here:
  * `_percentiles` is a well-formed statistics summary (handles empty,
    monotonic ordering of p50 <= p95 <= p99 <= max).
  * `_fmt_row` composes a one-line report row and flags budget overruns.
  * `measure_per_tick` returns the documented stage keys, with the
    sample count matching `ticks`, and `tick_total` dominating the sum
    of its components (it should be >= each component's p50).
"""
from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────── _percentiles

def test_percentiles_empty_input_is_well_defined():
    from tools.measure_latency import _percentiles

    s = _percentiles([])
    assert s["n"] == 0
    # keys match the ones the report consumes — rename-proof.
    assert set(s) == {"n", "mean", "p50", "p95", "p99", "max"}
    assert all(s[k] == 0.0 for k in ("mean", "p50", "p95", "p99", "max"))


def test_percentiles_are_monotonically_ordered():
    from tools.measure_latency import _percentiles

    s = _percentiles([float(x) for x in range(1, 101)])  # 1..100
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["p50"] == pytest.approx(50.5)   # statistics.median
    assert s["p95"] >= s["p50"]
    assert s["p99"] >= s["p95"]
    assert s["max"] >= s["p99"]
    assert s["max"] == 100.0


def test_percentiles_pick_from_sorted_tail():
    """`_percentiles` sorts internally — unsorted input must give the
    same answer as the sorted version. Catches a bug class where the
    helper would read raw indices without sorting."""
    from tools.measure_latency import _percentiles

    sorted_s   = _percentiles(sorted(range(50)))
    unsorted_s = _percentiles(list(reversed(range(50))))
    for k in ("mean", "p50", "p95", "p99", "max"):
        assert sorted_s[k] == unsorted_s[k]


# ─────────────────────────────────────────────────────────── _fmt_row

def test_fmt_row_includes_all_stat_fields():
    from tools.measure_latency import _fmt_row, _percentiles

    row = _fmt_row("step", _percentiles([1.0, 2.0, 3.0]))
    assert "step" in row
    for tag in ("n=", "mean=", "p50=", "p95=", "p99=", "max=", "ms"):
        assert tag in row


def test_fmt_row_flags_budget_overrun():
    from tools.measure_latency import _fmt_row, _percentiles

    under = _fmt_row("x", _percentiles([1.0]), budget_ms=10.0)
    over  = _fmt_row("x", _percentiles([99.0]), budget_ms=10.0)
    assert "OK"   in under and "OVER" not in under
    assert "OVER" in over
    assert "budget" in under and "budget" in over


# ──────────────────────────────────────────── measure_per_tick (real)

def _dummy_style():
    """Build a `GrooveStyle` that drives the per-tick loop at a known
    tempo without needing any trained ckpt. Mirrors the shape the
    selector would produce."""
    from groovebot.style.select import GrooveStyle
    return GrooveStyle(
        move="headbang", intensity=0.7,
        genre="rock", mood="aggressive",
        mood_probs={"aggressive": 1.0}, genre_probs={"rock": 1.0},
        tempo_bpm=120.0, arousal=0.6, arousal_bucket="mid",
    )


def test_measure_per_tick_runs_and_returns_the_stage_table():
    """Needs mujoco (installed). Runs ~50 ticks — fast enough for CI."""
    pytest.importorskip("mujoco")
    from tools.measure_latency import measure_per_tick

    stats = measure_per_tick(_dummy_style(), ticks=50, rate=50.0)

    # The script's report prints exactly these keys — if any is missing
    # or renamed, print_report() would silently miss a row.
    assert set(stats) == {
        "perception", "generate", "clamp", "send", "step", "tick_total",
    }
    for name, s in stats.items():
        assert s["n"] == 50, f"{name}: expected 50 samples, got {s['n']}"
        # non-negative and monotonic
        assert 0 <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"], \
            f"{name}: non-monotonic percentiles {s}"


def test_tick_total_is_at_least_the_dominant_component():
    """`tick_total` wraps all five sub-stages, so at the median it should
    be at least as large as the biggest sub-stage median. If this breaks
    the loop body desynced from what the sum covers."""
    pytest.importorskip("mujoco")
    from tools.measure_latency import measure_per_tick

    stats = measure_per_tick(_dummy_style(), ticks=100, rate=50.0)
    sub_medians = [
        stats[k]["p50"]
        for k in ("perception", "generate", "clamp", "send", "step")
    ]
    assert stats["tick_total"]["p50"] >= max(sub_medians)
