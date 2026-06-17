"""groovebot.style.narrate — text narration of the v1 bridge state.

Production output stays `JointCommand` (spec §5.1). This module is a
**read-only observability layer**: it converts the same per-tick state
that `experiments/render_groove.py` already dumps to CSV into a human-
readable summary, so the team can sanity-check what the robot just
"decided" before / during / after a sim run.

No new state is invented. Inputs:
  * `GrooveStyle` — the selector's window-level decision (genre / mood
    probs / arousal / tempo / move / intensity).
  * `commands` — an optional sequence of per-tick `JointCommand.targets`
    (joint_name → radians). Same dicts the orchestrator hands to the
    backend, same numbers the render CSV records.

Two granularity levels:
  * `format_window_summary(style, ...)` — one paragraph for the whole
    audio window (perception → decision → action).
  * `format_beat_trace(style, commands, rate, ...)` — per-beat lines
    showing the primary joint's peak angle and beat-phase position.

`narrate(style, commands=..., verbose=True, ...)` returns both joined
with newlines. Designed for stdout / log files, not for the production
control loop.
"""
from __future__ import annotations
from typing import Mapping, Sequence

from groovebot.groove_style import SOFT_AMP
from groovebot.style.select import GrooveStyle


# Per-move static descriptors. These mirror the closed-form primitives
# in `groovebot.groove_style.MOVE_PRIMITIVES`. If a primitive is added
# or its joints change, update both maps together (the test suite has
# a vocabulary-coverage check that catches drift).
PRIMARY_JOINTS: dict[str, list[str]] = {
    "headbang":      ["neck_pitch"],
    "bob_nod":       ["neck_pitch"],
    "sway":          ["torso_roll"],
    "rock":          ["torso_pitch"],
    "fist_pump":     ["l_shoulder_pitch", "r_shoulder_pitch"],
    "clap":          ["l_shoulder_roll", "r_shoulder_roll", "l_elbow", "r_elbow"],
    "penlight_wave": ["l_shoulder_pitch", "l_shoulder_roll"],
    "quiet_listen":  ["neck_pitch"],
}

# Beats per full motion cycle. Informational; used to render the
# "1拍/サイクル" / "2拍/サイクル" hint in the window summary.
CYCLE_BEATS: dict[str, float] = {
    "headbang":      1.0,
    "bob_nod":       1.0,
    "sway":          2.0,
    "rock":          2.0,
    "fist_pump":     1.0,
    "clap":          1.0,
    "penlight_wave": 2.0,
    "quiet_listen":  4.0,
}

MOVE_DESCRIPTOR: dict[str, str] = {
    "headbang":      "首pitch（縦ノリ）",
    "bob_nod":       "首pitch（軽い頷き）",
    "sway":          "体幹roll（左右揺れ）",
    "rock":          "体幹pitch（前後揺れ）",
    "fist_pump":     "両肩pitch+肘屈曲（拳上げ）",
    "clap":          "両肩roll+肘屈曲（手拍子）",
    "penlight_wave": "左肩pitch+roll（ペンライト）",
    "quiet_listen":  "首pitch（呼吸）",
}


# --------------------------------------------------------- helpers

def _top_probs(probs: Mapping[str, float], n: int = 1) -> list[tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _reason_for(style: GrooveStyle) -> str:
    """Build the parenthetical "why this move was picked" string.

    Falls back to the GrooveStyle.mood label when mood_probs is empty
    so the message stays readable even with partial input.
    """
    top = _top_probs(style.mood_probs, n=1)
    if top:
        mood_str = f"{top[0][0]}({top[0][1]:.2f})"
    else:
        mood_str = style.mood
    return f"{style.genre} × {style.arousal_bucket} × {mood_str}"


def _primary_for(move: str) -> list[str]:
    return PRIMARY_JOINTS.get(move, [])


def _beat_phase_label(phase_in_beat: float) -> str:
    """Map a 0..1 phase to "on-beat" / "off-beat".

    Near the integer boundaries (within 10%) we call it on-beat, the
    rest of the beat is off-beat. This is informational only — the v1
    primitives have their peaks at specific phases (e.g. headbang peaks
    at b+0.5 by design), and the narration reports that honestly
    instead of forcing a beat-aligned story.
    """
    if phase_in_beat < 0.10 or phase_in_beat > 0.90:
        return "on-beat"
    return "off-beat"


# --------------------------------------------------------- public API

def format_window_summary(
    style: GrooveStyle,
    *,
    t_start: float = 0.0,
    t_end: float | None = None,
    seconds: float | None = None,
    rate: float | None = None,
    reason: str | None = None,
) -> str:
    """One paragraph of "perception → decision → action" for a window.

    Either pass `t_end` directly, or pass `seconds` and `t_start` is
    used as the origin. If neither is given the window collapses to a
    point (still legal — useful for "current snapshot" prints).
    """
    if t_end is None:
        t_end = t_start + float(seconds) if seconds is not None else t_start
    primary = _primary_for(style.move)
    primary_str = "+".join(primary) if primary else "—"
    descriptor = MOVE_DESCRIPTOR.get(style.move, f"{style.move}（未登録）")
    cycle = CYCLE_BEATS.get(style.move)
    cycle_str = f"、{cycle:g}拍/サイクル" if cycle is not None else ""
    if primary:
        soft = max(SOFT_AMP.get(j, 0.0) for j in primary)
        soft_str = f"、設計上限|θ|≈{soft:.2f} rad"
    else:
        soft_str = ""
    reason_str = reason if reason is not None else _reason_for(style)
    rate_str = f", {rate:.0f}Hz" if rate is not None else ""
    return (
        f"[{t_start:.2f}-{t_end:.2f}s{rate_str}]\n"
        f"  知覚: genre={style.genre}, mood={style.mood}, "
        f"arousal={style.arousal:.2f}/{style.arousal_bucket}, "
        f"tempo={style.tempo_bpm:.0f}BPM\n"
        f"  判断: GrooveStyle={style.move}@{style.intensity:.2f} "
        f"({reason_str})\n"
        f"  動作: {descriptor}{cycle_str}, 主関節={primary_str}{soft_str}"
    )


def format_beat_trace(
    style: GrooveStyle,
    *,
    commands: Sequence[Mapping[str, float]],
    rate: float,
    bpm: float | None = None,
    max_beats: int | None = None,
) -> list[str]:
    """Per-beat trace lines from a captured tick history.

    For each integer beat that has any ticks, report the primary
    joint's peak angle (by absolute value), the beat-phase position
    of that peak, and an on-beat/off-beat label.

    Parameters
    ----------
    commands :
        Per-tick `JointCommand.targets` dicts — same list the CSV
        writer in `render_groove` collects.
    rate :
        Control rate (Hz). Required so we can map tick index → time.
    bpm :
        Beat rate. Defaults to `style.tempo_bpm`.
    max_beats :
        Cap the number of lines returned (useful for very long runs).
    """
    if not commands:
        return []
    bpm_v = float(bpm if bpm is not None else style.tempo_bpm)
    if bpm_v <= 0:
        return ["  （テンポ未確定のためビートトレース省略）"]
    primary = _primary_for(style.move)
    if not primary:
        return [f"  （{style.move} のプライマリ関節未登録）"]
    main = primary[0]

    sec_per_tick = 1.0 / float(rate)
    beats_per_sec = bpm_v / 60.0

    # Bucket ticks by integer beat index; track (beat-phase, value).
    buckets: dict[int, list[tuple[float, float]]] = {}
    for i, cmd in enumerate(commands):
        t = i * sec_per_tick
        b = t * beats_per_sec
        bi = int(b)
        buckets.setdefault(bi, []).append((b - bi, float(cmd.get(main, 0.0))))

    lines: list[str] = []
    for bi in sorted(buckets):
        if max_beats is not None and bi >= max_beats:
            break
        pairs = buckets[bi]
        peak_phase, peak_val = max(pairs, key=lambda pv: abs(pv[1]))
        label = _beat_phase_label(peak_phase)
        t_beat = bi / beats_per_sec
        lines.append(
            f"  beat#{bi} [t={t_beat:.3f}s] {style.move}: "
            f"{main} peak={peak_val:+.3f} rad @ b+{peak_phase:.2f} ({label})"
        )
    return lines


def narrate(
    style: GrooveStyle,
    commands: Sequence[Mapping[str, float]] | None = None,
    *,
    rate: float | None = None,
    t_start: float = 0.0,
    seconds: float | None = None,
    verbose: bool = False,
    reason: str | None = None,
    max_beats: int | None = None,
) -> str:
    """Convenience: window summary, optionally followed by a beat trace.

    `verbose=True` requires both `commands` and `rate`. Without them
    the function appends a single informational line instead of
    crashing, so callers can wire the flag without staging guards
    everywhere.
    """
    parts = [format_window_summary(
        style,
        t_start=t_start, seconds=seconds, rate=rate, reason=reason,
    )]
    if verbose:
        if commands is None or rate is None:
            parts.append("  （拍トレース省略: commands/rate 未指定）")
        else:
            parts.extend(format_beat_trace(
                style, commands=commands, rate=rate,
                max_beats=max_beats,
            ))
    return "\n".join(parts)
