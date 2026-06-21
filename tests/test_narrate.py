"""Tests for groovebot.style.narrate — text narration of v1 bridge state.

The narration layer is observability: it must (a) faithfully report
the `GrooveStyle` it was given, (b) name the same primary joints the
generator actually drives, and (c) emit one entry per beat in the
trace mode. No MuJoCo, no audio — pure dict/string assertions.
"""
from __future__ import annotations
import math

import pytest

from groovebot.groove_style import (
    MOVE_PRIMITIVES,
    StyleGrooveGenerator,
)
from groovebot.orchestrator import MetronomePerception
from groovebot.style.narrate import (
    CYCLE_BEATS,
    MOVE_DESCRIPTOR,
    PRIMARY_JOINTS,
    format_beat_trace,
    format_window_summary,
    narrate,
)
from groovebot.style.select import GrooveStyle
from groovebot.style.table import MOVES


# ---------------------------------------------- helpers

def _style(
    move: str,
    *,
    intensity: float = 0.85,
    bpm: float = 128.0,
    genre: str = "rock",
    mood: str = "aggressive",
    arousal: float = 0.82,
    bucket: str = "high",
    mood_probs: dict[str, float] | None = None,
) -> GrooveStyle:
    if mood_probs is None:
        mood_probs = {"aggressive": 0.55, "happy": 0.20, "epic": 0.15,
                      "calm": 0.05, "sad": 0.03, "dark": 0.02}
    return GrooveStyle(
        move=move,
        intensity=intensity,
        genre=genre,
        mood=mood,
        mood_probs=mood_probs,
        genre_probs={genre: 0.62, "metal": 0.20, "pop": 0.10,
                     "jazz": 0.04, "blues": 0.04},
        tempo_bpm=bpm,
        arousal=arousal,
        arousal_bucket=bucket,
    )


def _run_for_commands(
    style: GrooveStyle,
    *,
    seconds: float,
    rate: float = 50.0,
) -> list[dict[str, float]]:
    """Drive StyleGrooveGenerator headless and capture per-tick targets.

    Uses the same MetronomePerception + StyleGrooveGenerator wiring the
    Orchestrator would use, so the narration is fed the very same numbers
    that hit the (mocked) backend.
    """
    gen = StyleGrooveGenerator(style=style)
    perc = MetronomePerception(bpm=style.tempo_bpm, energy=style.intensity)
    n = int(round(seconds * rate))
    dt = 1.0 / rate
    out = []
    for _ in range(n):
        ctx = perc.tick(dt)
        out.append(dict(gen.generate(ctx).targets))
    return out


# ---------------------------------------------- vocabulary coverage

def test_narration_tables_cover_full_move_vocabulary():
    # If a move is added to the table without a narration entry, the
    # window summary would say "未登録" — catch that drift here.
    assert set(PRIMARY_JOINTS.keys()) == set(MOVES)
    assert set(CYCLE_BEATS.keys()) == set(MOVES)
    assert set(MOVE_DESCRIPTOR.keys()) == set(MOVES)


def test_primary_joints_match_primitive_outputs():
    # Every primary joint declared by the narrator must actually be
    # driven by the corresponding primitive at intensity=1, b=0.25.
    # b=0.25 chosen because every primitive's primary joint is
    # non-zero there (sin(πb) and (1-cos 2πb) both > 0).
    for move, joints in PRIMARY_JOINTS.items():
        out = MOVE_PRIMITIVES[move](0.25, 1.0)
        for joint in joints:
            assert joint in out, (
                f"narrate primary joint {joint} not present in {move} output"
            )


# ---------------------------------------------- window summary

def test_window_summary_reports_style_attributes():
    style = _style("headbang", intensity=0.85, bpm=128.0)
    s = format_window_summary(style, seconds=8.0, rate=50.0)
    assert "headbang" in s
    assert "0.85" in s                  # intensity
    assert "128" in s                   # tempo
    assert "rock" in s                  # genre
    assert "aggressive" in s            # mood label or top prob
    assert "high" in s                  # arousal bucket
    assert "0.82" in s                  # arousal
    assert "neck_pitch" in s            # primary joint for headbang
    assert "8.00s" in s                 # window end
    assert "50Hz" in s
    assert "0.55" in s                  # SOFT_AMP for neck_pitch


def test_window_summary_uses_explicit_reason_when_given():
    style = _style("sway")
    s = format_window_summary(style, seconds=4.0, reason="override-reason")
    assert "override-reason" in s
    # When override given, default genre×bucket×mood string should NOT
    # also appear with the parenthesised mood-prob form.
    assert "rock × high × aggressive(" not in s


def test_window_summary_reports_top_mood_prob_in_default_reason():
    style = _style("sway")
    s = format_window_summary(style, seconds=4.0)
    assert "rock × high × aggressive(0.55)" in s


def test_window_summary_handles_unknown_move():
    style = _style("not_a_real_move")
    s = format_window_summary(style, seconds=4.0)
    assert "not_a_real_move" in s
    assert "未登録" in s
    assert "主関節=—" in s


@pytest.mark.parametrize("move", list(MOVE_PRIMITIVES.keys()))
def test_window_summary_names_primary_joint_for_each_move(move):
    style = _style(move)
    s = format_window_summary(style, seconds=4.0)
    for joint in PRIMARY_JOINTS[move]:
        assert joint in s


# ---------------------------------------------- beat trace

def test_beat_trace_emits_one_line_per_integer_beat_window():
    style = _style("sway", intensity=1.0, bpm=120.0)
    # 2 s at 120 BPM = 4 beats. With rate 50 Hz the last tick lands at
    # t = 1.98 s = 3.96 beats, so we get beat indices {0, 1, 2, 3}.
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    trace = format_beat_trace(style, commands=commands, rate=50.0)
    assert len(trace) == 4
    for line in trace:
        assert "sway" in line
        assert "torso_roll" in line
        assert "rad" in line


def test_beat_trace_respects_max_beats():
    style = _style("sway", intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=4.0, rate=50.0)
    trace = format_beat_trace(
        style, commands=commands, rate=50.0, max_beats=3,
    )
    assert len(trace) == 3


def test_beat_trace_peak_value_inside_urdf_amplitude():
    # sway peak |torso_roll| = 0.35*1.0 = 0.35 — well inside URDF ±0.45.
    style = _style("sway", intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    trace = format_beat_trace(style, commands=commands, rate=50.0)
    for line in trace:
        # Pull the signed peak value out: pattern "peak=+0.350"
        token = next(t for t in line.split() if t.startswith("peak="))
        val = float(token.split("=")[1])
        assert abs(val) <= 0.45 + 1e-9


def test_beat_trace_handles_empty_commands():
    style = _style("headbang")
    assert format_beat_trace(style, commands=[], rate=50.0) == []


def test_beat_trace_handles_zero_bpm():
    style = _style("headbang")
    out = format_beat_trace(
        style, commands=[{"neck_pitch": 0.0}], rate=50.0, bpm=0.0,
    )
    assert out and "省略" in out[0]


def test_beat_trace_uses_override_bpm():
    style = _style("sway", intensity=1.0, bpm=120.0)
    # Generated at 120 BPM but narrated at 60 BPM → half as many
    # beats land inside the same 2 s window.
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    trace_60 = format_beat_trace(style, commands=commands, rate=50.0, bpm=60.0)
    assert len(trace_60) == 2


def test_beat_trace_marks_on_beat_for_sway():
    # sway's primary joint torso_roll = 0.35 * sin(πb). Within beat 0
    # (b ∈ [0, 1)) the peak is at b=0.5 — off-beat by the narrator's
    # 10%-corner rule. This is *correct*: under the v1.2 phase
    # contract, continuous moves (sway/rock/penlight_wave) peak
    # mid-beat by design, and the narrator must say so honestly.
    style = _style("sway", intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=1.0, rate=50.0)
    trace = format_beat_trace(style, commands=commands, rate=50.0)
    assert "off-beat" in trace[0]


@pytest.mark.parametrize("move", ["headbang", "bob_nod", "fist_pump"])
def test_beat_trace_marks_on_beat_for_accent_moves(move):
    # v1.2 phase contract: accent moves peak ON each beat. The
    # narration's 10%-corner rule should label them "on-beat".
    style = _style(move, intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    trace = format_beat_trace(style, commands=commands, rate=50.0)
    assert trace
    # Every beat line should carry the on-beat label.
    for line in trace:
        assert "on-beat" in line, f"{move}: expected on-beat in '{line}'"


def test_beat_trace_marks_on_beat_for_clap_accents():
    # v1.2 phase contract: clap accents land at b=1, 3, 5, ...
    # (musical beats 2 & 4 = backbeat). The clap pulse is continuous
    # over a 2-beat period: pulse=0 at b=0, peaks at b=1, back to 0
    # at b=2. Per-bucket peak phase asymmetry is the signature:
    #   bucket 0 (musical beat 1): pulse rising → peak_phase near 1.0
    #   bucket 1 (musical beat 2): pulse at top → peak_phase near 0.0
    #   bucket 2 (musical beat 3): pulse rising again → near 1.0
    #   bucket 3 (musical beat 4): pulse at top again → near 0.0
    # All four are within the 10% on-beat corner, so labels are all
    # "on-beat" — but the position within the bucket reveals where
    # the accent really sits.
    style = _style("clap", intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    trace = format_beat_trace(style, commands=commands, rate=50.0)
    assert len(trace) == 4

    def _peak_phase(line: str) -> float:
        token = next(t for t in line.split() if t.startswith("b+"))
        return float(token.removeprefix("b+"))

    phases = [_peak_phase(line) for line in trace]
    # Backbeat: musical 2 & 4 (bucket 1, 3) peak at the beat itself
    # (phase ≈ 0), while musical 1 & 3 (bucket 0, 2) are the rising
    # buildup (phase near the right edge).
    assert phases[1] < 0.10
    assert phases[3] < 0.10
    assert phases[0] > 0.90
    assert phases[2] > 0.90
    for line in trace:
        assert "on-beat" in line


# ---------------------------------------------- top-level narrate()

def test_narrate_returns_window_only_by_default():
    style = _style("headbang")
    out = narrate(style, seconds=4.0, rate=50.0)
    assert out.startswith("[0.00-4.00s, 50Hz]")
    assert "beat#" not in out


def test_narrate_verbose_appends_beat_lines():
    style = _style("headbang", intensity=1.0, bpm=120.0)
    commands = _run_for_commands(style, seconds=2.0, rate=50.0)
    out = narrate(style, commands, rate=50.0, seconds=2.0, verbose=True)
    assert out.startswith("[0.00-2.00s, 50Hz]")
    # 2 s @ 120 BPM = 4 beats, so 4 trace lines after the summary.
    body = out.splitlines()
    summary_lines = 4   # bracket + 3 indented lines
    trace_lines = [l for l in body[summary_lines:] if l.startswith("  beat#")]
    assert len(trace_lines) == 4


def test_narrate_verbose_without_commands_emits_skip_note():
    style = _style("bob_nod")
    out = narrate(style, seconds=4.0, verbose=True)
    assert "拍トレース省略" in out
