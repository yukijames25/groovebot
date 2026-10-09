"""tools/measure_latency.py — real-time budget check against SYSTEM_SPEC §6.2.

Measures two separate latencies on the current (M1 + v3.1) pipeline:

  1. **One-shot style selection** — `GrooveStyleSelector.select(audio, sr)`
     i.e. PANNs CNN14 embed + StyleHead + StyleRegressionHead + V/A -> mood +
     table.select_move. This runs once per song, not per tick.

  2. **Per-tick control loop** — what the fixed-rate `Orchestrator` actually
     does on every tick: `perception.tick(dt)` + `generator.generate(ctx)` +
     `clamp_command()` + `backend.set_joint_targets()` + `backend.step(dt)`.
     Then a breakdown by component so §6.2's rows can be filled in.

Honest caveat (print it loudly in the report):
  The per-tick path does NOT include continuous audio ingestion. In the
  current pipeline, `MetronomePerception` is a BPM-driven stub — the audio
  was consumed ONCE at song start by the selector, which produced
  `style.tempo_bpm`; thereafter the beat position is advanced by the clock,
  not by what the singer is actually doing. NFR-5's end-to-end voice->body
  budget therefore applies to a path (`ReferenceAligner`) that does not
  exist yet in code. This script measures what IS wired, honestly, and
  flags the gap.

Usage:
    python -m tools.measure_latency                        # default: 4 showcase clips
    python -m tools.measure_latency --ticks 2000           # longer per-tick sample
    python -m tools.measure_latency --clips path1.wav p2.wav
    python -m tools.measure_latency --json out.json        # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse the exact wiring the showcase uses, so the measurement reflects
# the real pipeline (not a hand-rolled approximation).
from experiments.render_groove import (
    DEFAULT_AROUSAL_HEAD,
    DEFAULT_GENRE_HEAD,
    DEFAULT_PANNS_CKPT,
    _build_selector,
)
from groovebot.backend import JOINT_NAMES, MujocoBackend
from groovebot.groove_style import StyleGrooveGenerator, metronome_from_style
from groovebot.limits import make_clamp
from groovebot.orchestrator import Orchestrator
from groovebot.types import GrooveContext


# ─── timing helpers ────────────────────────────────────────────────────────

def _now_ms() -> float:
    """Monotonic high-res timer in milliseconds (perf_counter_ns / 1e6)."""
    return time.perf_counter_ns() / 1e6


def _percentiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    if not xs:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    n = len(xs)
    pick = lambda q: xs[min(n - 1, int(q * n))]
    return {
        "n":    n,
        "mean": statistics.fmean(xs),
        "p50":  statistics.median(xs),
        "p95":  pick(0.95),
        "p99":  pick(0.99),
        "max":  xs[-1],
    }


def _fmt_row(label: str, s: dict, *, budget_ms: float | None = None) -> str:
    """One row of the latency table: label | mean | p50 | p95 | p99 | max | budget?."""
    bud = f"  (budget {budget_ms:.0f} ms)" if budget_ms is not None else ""
    verdict = ""
    if budget_ms is not None:
        verdict = "  OK" if s["p95"] <= budget_ms else "  OVER"
    return (
        f"  {label:<32} "
        f"n={s['n']:>5}  "
        f"mean={s['mean']:>7.3f}  "
        f"p50={s['p50']:>7.3f}  "
        f"p95={s['p95']:>7.3f}  "
        f"p99={s['p99']:>7.3f}  "
        f"max={s['max']:>7.3f} ms"
        f"{bud}{verdict}"
    )


# ─── (1) one-shot style selection latency ──────────────────────────────────

def measure_selector(
    selector,
    clip_paths: list[Path],
    *,
    warmup: int = 1,
) -> dict:
    """Return per-stage latency stats for the one-time song-start cost.

    Stages measured per clip (strictly sequential, same objects the showcase
    reuses):

      - `audio_load`   : librosa.load (wav decode + resample)
      - `embed`        : PannsBackbone.embed  — PANNs CNN14 forward
      - `heads`        : StyleHead + StyleRegressionHead on the same embed
      - `va_mood`      : mood_probs_from_va  — pure NumPy
      - `tempo`        : attributes.estimate_tempo (librosa.beat.beat_track)
      - `select_move`  : table lookup (dict ops)
      - `select_total` : selector.select(audio, sr) end-to-end
    """
    import librosa

    from groovebot.style.attributes import estimate_tempo
    from groovebot.style.mood_from_va import mood_probs_from_va
    from groovebot.style.deam import sam_to_unit
    from groovebot.style.select import GENRES, MOODS
    from groovebot.style.table import select_move
    import torch

    stages: dict[str, list[float]] = {
        "audio_load":   [],
        "embed":        [],
        "heads":        [],
        "va_mood":      [],
        "tempo":        [],
        "select_move":  [],
        "select_total": [],
    }

    # Warm up PyTorch / PANNs so the first run doesn't skew `p99`.
    if clip_paths and warmup:
        y, sr = librosa.load(str(clip_paths[0]), sr=None, mono=True)
        audio = np.asarray(y, dtype=np.float32)
        for _ in range(warmup):
            selector.select(audio, int(sr))

    for clip in clip_paths:
        # audio_load
        t = _now_ms()
        y, sr = librosa.load(str(clip), sr=None, mono=True)
        stages["audio_load"].append(_now_ms() - t)
        audio = np.asarray(y, dtype=np.float32)

        # embed (PANNs CNN14 forward on CPU)
        t = _now_ms()
        emb = selector.backbone.embed(audio, int(sr))
        stages["embed"].append(_now_ms() - t)

        # heads: StyleHead (genre+mood) + StyleRegressionHead (arousal+valence)
        t = _now_ms()
        x = torch.from_numpy(emb).unsqueeze(0).to(selector.device)
        probs = selector.head.predict_probs(x)
        gp = probs["genre"].squeeze(0).cpu().numpy()
        _ = probs["mood"].squeeze(0).cpu().numpy()
        with torch.no_grad():
            out = selector.regression_head(x)
            a_raw = float(out["arousal"].item())
            v_raw = float(out["valence"].item())
        arousal_val = sam_to_unit(a_raw)
        valence_val = sam_to_unit(v_raw)
        genre_probs = {g: float(p) for g, p in zip(GENRES, gp)}
        stages["heads"].append(_now_ms() - t)

        # va -> mood (pure numpy, prototype distances)
        t = _now_ms()
        mood_probs = mood_probs_from_va(
            arousal_val, valence_val, prototypes=selector.mood_va_prototypes,
        )
        stages["va_mood"].append(_now_ms() - t)

        # tempo (librosa.beat.beat_track — the single biggest cost after PANNs)
        t = _now_ms()
        tempo = estimate_tempo(audio, int(sr))
        stages["tempo"].append(_now_ms() - t)

        # table lookup
        t = _now_ms()
        from groovebot.style.attributes import arousal_bucket
        genre = max(genre_probs, key=genre_probs.get)
        bucket = arousal_bucket(arousal_val)
        _ = select_move(genre, bucket, mood_probs)
        stages["select_move"].append(_now_ms() - t)

        # end-to-end: what the pipeline actually calls
        t = _now_ms()
        style = selector.select(audio, int(sr))
        stages["select_total"].append(_now_ms() - t)

    return {name: _percentiles(xs) for name, xs in stages.items()}


# ─── (2) per-tick control-loop latency ─────────────────────────────────────

def measure_per_tick(
    style,
    ticks: int,
    rate: float,
) -> dict:
    """Return per-component latency stats for a single Orchestrator tick.

    Runs the real `MujocoBackend` with the real `StyleGrooveGenerator`.
    No audio is consumed inside the loop (that is the honest NFR-5 gap).

    Measured per tick:
      - `perception`   : MetronomePerception.tick(dt)  — BPM clock step
      - `generate`     : StyleGrooveGenerator.generate(ctx)
      - `clamp`        : clamp_command (URDF range check per joint, NFR-4)
      - `send`         : MujocoBackend.set_joint_targets (dict update)
      - `step`         : MujocoBackend.step(dt) (mj_forward kinematics)
      - `tick_total`   : the five above, back-to-back
    """
    from groovebot.limits import make_clamp
    urdf = str(_REPO_ROOT / "robot" / "groovebot.urdf")

    backend = MujocoBackend()
    backend.load(urdf)
    perception = metronome_from_style(style)
    generator = StyleGrooveGenerator(style=style, use_ctx_arousal=False)
    clamp = make_clamp(urdf, joint_names=JOINT_NAMES)
    dt = 1.0 / float(rate)

    stages: dict[str, list[float]] = {
        "perception":  [],
        "generate":    [],
        "clamp":       [],
        "send":        [],
        "step":        [],
        "tick_total":  [],
    }

    # Warm up a few ticks so the first MuJoCo kinematic eval doesn't skew p99.
    for _ in range(20):
        ctx = perception.tick(dt)
        cmd = generator.generate(ctx)
        cmd = clamp(cmd)
        backend.set_joint_targets(cmd.targets)
        backend.step(dt)

    for _ in range(ticks):
        tick_start = _now_ms()

        t = _now_ms()
        ctx = perception.tick(dt)
        stages["perception"].append(_now_ms() - t)

        t = _now_ms()
        cmd = generator.generate(ctx)
        stages["generate"].append(_now_ms() - t)

        t = _now_ms()
        cmd = clamp(cmd)
        stages["clamp"].append(_now_ms() - t)

        t = _now_ms()
        backend.set_joint_targets(cmd.targets)
        stages["send"].append(_now_ms() - t)

        t = _now_ms()
        backend.step(dt)
        stages["step"].append(_now_ms() - t)

        stages["tick_total"].append(_now_ms() - tick_start)

    backend.close()
    return {name: _percentiles(xs) for name, xs in stages.items()}


# ─── report ────────────────────────────────────────────────────────────────

def print_report(
    *,
    clip_paths: list[Path],
    rate: float,
    sel_stats: dict,
    tick_stats: dict,
) -> None:
    period_ms = 1000.0 / rate

    print()
    print("=" * 90)
    print("GrooveBot latency measurement — SYSTEM_SPEC §6.2 (NFR-5 + NFR-7)")
    print("=" * 90)
    print(f"  pipeline:        PANNs CNN14 + StyleHead + StyleRegressionHead -> V/A -> table")
    print(f"  control rate:    {rate:.0f} Hz  (period = {period_ms:.2f} ms)")
    print(f"  clips:           {[p.name for p in clip_paths]}")
    print()
    print("-- (1) ONE-SHOT STYLE SELECTION  (runs once per song, at song start) --")
    print()
    print(_fmt_row("audio_load  (librosa.load)",        sel_stats["audio_load"]))
    print(_fmt_row("embed       (PANNs CNN14 forward)", sel_stats["embed"]))
    print(_fmt_row("heads       (Style{,Regression}Head)", sel_stats["heads"]))
    print(_fmt_row("va_mood     (mood_probs_from_va)",  sel_stats["va_mood"]))
    print(_fmt_row("tempo       (librosa.beat.beat_track)", sel_stats["tempo"]))
    print(_fmt_row("select_move (table lookup)",         sel_stats["select_move"]))
    print(_fmt_row("TOTAL       selector.select()",      sel_stats["select_total"]))
    print()
    print("  NOTE: this is a ONE-TIME cost at song start, not a per-tick cost.")
    print("        NFR-5 (<=150 ms end-to-end) is a per-event budget; this does")
    print("        not fit that budget on CPU because PANNs CNN14 was never")
    print("        meant to be a live classifier. The showcase runs select()")
    print("        once, caches `style`, and then runs the control loop.")
    print()
    print("-- (2) PER-TICK CONTROL LOOP  (what the fixed-rate Orchestrator runs) --")
    print()
    print(_fmt_row("perception  (MetronomePerception)",  tick_stats["perception"]))
    print(_fmt_row("generate    (StyleGrooveGenerator)", tick_stats["generate"]))
    print(_fmt_row("clamp       (URDF range, NFR-4)",    tick_stats["clamp"]))
    print(_fmt_row("send        (MujocoBackend.set)",    tick_stats["send"]))
    print(_fmt_row("step        (MujocoBackend.step)",   tick_stats["step"]))
    print(_fmt_row("TICK TOTAL  (sum of above)", tick_stats["tick_total"],
                   budget_ms=period_ms))
    print()
    budget_7 = period_ms
    overruns = sum(1 for t in [tick_stats["tick_total"]["p95"]] if t > budget_7)
    print(f"  NFR-7 check (fixed {rate:.0f} Hz, period {period_ms:.2f} ms):")
    print(f"    tick p95 = {tick_stats['tick_total']['p95']:.3f} ms, "
          f"p99 = {tick_stats['tick_total']['p99']:.3f} ms, "
          f"max = {tick_stats['tick_total']['max']:.3f} ms "
          f"{'(within period)' if tick_stats['tick_total']['max'] < period_ms else '(! OVERRUN at max)'}")
    print()
    print("-- NFR-5 end-to-end voice->body (<=150 ms) --")
    print()
    print("  Honest finding: the current pipeline does NOT have a continuous")
    print("  voice->beat path. MetronomePerception is a BPM-driven stub whose")
    print("  BPM was set once by the selector at song start. There is no")
    print("  per-event voice->body latency to measure yet. The §6.2 budget")
    print("  rows that WOULD cover this (ReferenceAligner, ArousalEstimator)")
    print("  belong to M2, which is not implemented. Closing this gap requires")
    print("  M2.")
    print()
    print("  What IS measurable today, with the per-tick numbers above:")
    print(f"    - jitter: tick p99 - p50 = "
          f"{tick_stats['tick_total']['p99'] - tick_stats['tick_total']['p50']:.3f} ms")
    print(f"    - headroom under NFR-7 period: "
          f"{period_ms - tick_stats['tick_total']['p95']:.2f} ms ({100*(period_ms - tick_stats['tick_total']['p95'])/period_ms:.1f}%)")
    print(f"    - one-shot select() cost: p95 = "
          f"{sel_stats['select_total']['p95']:.1f} ms "
          f"(acceptable for song-start; unrelated to NFR-5 per-event)")
    print()
    print("=" * 90)


# ─── entry point ───────────────────────────────────────────────────────────

def _default_clips() -> list[Path]:
    showcase = _REPO_ROOT / "data" / "raw" / "showcase"
    return sorted(showcase.glob("*.wav"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", nargs="*", default=None,
                    help="Audio clip paths. Default: data/raw/showcase/*.wav")
    ap.add_argument("--ticks", type=int, default=1000,
                    help="Per-tick samples for the control loop (default 1000).")
    ap.add_argument("--rate", type=float, default=50.0,
                    help="Control rate in Hz (default 50; NFR-7 band 30-50).")
    ap.add_argument("--panns-ckpt",   default=DEFAULT_PANNS_CKPT)
    ap.add_argument("--genre-head",   default=DEFAULT_GENRE_HEAD)
    ap.add_argument("--arousal-head", default=DEFAULT_AROUSAL_HEAD)
    ap.add_argument("--mood-source",  default="va", choices=["head", "va"])
    ap.add_argument("--json", default=None,
                    help="Also write a machine-readable JSON report here.")
    args = ap.parse_args(argv)

    clip_paths = (
        [Path(c) for c in args.clips] if args.clips else _default_clips()
    )
    clip_paths = [p for p in clip_paths if p.is_file()]
    if not clip_paths:
        print("ERROR: no clip files found. Pass --clips or populate "
              "data/raw/showcase/.", file=sys.stderr)
        return 2

    print(f"[measure_latency] building selector (CPU, PANNs CNN14 "
          f"+ TL heads)...", flush=True)
    selector = _build_selector(
        panns_ckpt=args.panns_ckpt,
        genre_head=args.genre_head,
        arousal_head=args.arousal_head,
        mood_source=args.mood_source,
        allow_untrained=False,
    )

    print(f"[measure_latency] (1) one-shot selection on "
          f"{len(clip_paths)} clips (warmup=1)...", flush=True)
    sel_stats = measure_selector(selector, clip_paths, warmup=1)

    # Pick the first clip's style to drive the per-tick measurement.
    import librosa
    y, sr = librosa.load(str(clip_paths[0]), sr=None, mono=True)
    style = selector.select(np.asarray(y, dtype=np.float32), int(sr))
    print(f"[measure_latency] driving per-tick loop with "
          f"{clip_paths[0].name}: {style.as_text()}", flush=True)

    print(f"[measure_latency] (2) per-tick control loop: "
          f"{args.ticks} ticks @ {args.rate:.0f} Hz...", flush=True)
    tick_stats = measure_per_tick(style, ticks=args.ticks, rate=args.rate)

    print_report(
        clip_paths=clip_paths,
        rate=args.rate,
        sel_stats=sel_stats,
        tick_stats=tick_stats,
    )

    if args.json:
        out = {
            "pipeline":         "PANNs CNN14 + StyleHead + StyleRegressionHead -> V/A -> table",
            "rate_hz":          args.rate,
            "clips":            [str(p) for p in clip_paths],
            "selector":         sel_stats,
            "per_tick":         tick_stats,
            "nfr_7_period_ms":  1000.0 / args.rate,
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[measure_latency] json report -> {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
