"""experiments/run_noise_robustness.py — how robust is the tagger to dirty audio?

Procedure (no training; uses existing checkpoints):

  1. Build the v3.1 selector (`GrooveStyleSelector.from_panns`) with the
     same checkpoints the showcase uses.
  2. Collect an evaluation set: the 4 showcase clips + N GTZAN clips per
     genre (default 5 × 10 = 50 → 54 clips total).
  3. For each clip:
       - run the selector on the CLEAN audio → baseline tags
       - for each degradation condition in `default_conditions()`:
            - degrade the audio
            - run the selector again
            - record the delta from baseline on genre / arousal / valence
              / arousal_bucket / final move
  4. Aggregate per-condition stats and dump a report (stdout + JSON).

The question is **not** "did the tagger pick the correct label"; it is
**"did the tagger pick the same label as it did on the clean recording"**.
Consistency is the metric that matters for a live mic pipeline where
you don't have ground truth, only the dry-run reference.

Honest limitations are called out in the report: a condition can also
look "robust" simply because the tagger was already saturated or stuck
— that distinction has to be read off the baseline confidence, not
inferred from the match rate alone.

Usage:
    python -m experiments.run_noise_robustness
    python -m experiments.run_noise_robustness --per-genre 3
    python -m experiments.run_noise_robustness --json data/noise_robustness.json
    python -m experiments.run_noise_robustness --quick     # 1 clip/genre, smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


# Reuse the exact wiring the showcase uses so the measurement reflects
# the real production pipeline, not a hand-rolled approximation.
from experiments.render_groove import (
    DEFAULT_AROUSAL_HEAD,
    DEFAULT_GENRE_HEAD,
    DEFAULT_PANNS_CKPT,
    _build_selector,
)
from groovebot.style.select import GrooveStyle
from tools.degrade_audio import Condition, default_conditions


# ─── eval set discovery ──────────────────────────────────────────────────

def default_showcase_clips() -> list[Path]:
    """The 4 showcase clips (if present)."""
    root = _REPO_ROOT / "data" / "raw" / "showcase"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.wav"))


def default_gtzan_clips(per_genre: int = 5) -> list[Path]:
    """`per_genre` clips from each GTZAN genre, picked deterministically.

    Walks `data/raw/gtzan_full/Data/genres_original/<genre>/*.wav`. Picks
    the first `per_genre` wavs in sorted order from each genre. Returns
    an empty list if GTZAN is absent.
    """
    root = _REPO_ROOT / "data" / "raw" / "gtzan_full" / "Data" / "genres_original"
    if not root.is_dir():
        return []
    out: list[Path] = []
    for genre_dir in sorted(root.iterdir()):
        if not genre_dir.is_dir():
            continue
        wavs = sorted(genre_dir.glob("*.wav"))
        out.extend(wavs[: per_genre])
    return out


# ─── baseline + degraded tagging ──────────────────────────────────────────

@dataclass
class ClipTag:
    """Snapshot of what the selector said about one (clip, condition) pair."""
    clip: str
    condition: str          # "clean" for the baseline
    family: str             # "baseline" / "noise" / "reverb" / "volume" / "band" / "codec"
    genre_top: str
    genre_top_prob: float
    arousal: float
    arousal_bucket: str
    valence: float | None
    move: str
    intensity: float


def _affect_valence(selector, audio: np.ndarray, sr: int) -> float | None:
    """Pull the regression-head valence directly, like run_showcase does."""
    if getattr(selector, "regression_head", None) is None:
        return None
    emb = selector.backbone.embed(audio, int(sr))
    _, v_unit = selector._affect_from_emb(emb)
    return float(v_unit)


def tag_one(selector, audio: np.ndarray, sr: int,
            *, clip: str, condition: str, family: str) -> ClipTag:
    """Run the selector, package the subset of fields we compare on."""
    style: GrooveStyle = selector.select(audio, int(sr))
    gtop = max(style.genre_probs, key=style.genre_probs.get)
    return ClipTag(
        clip=clip,
        condition=condition,
        family=family,
        genre_top=gtop,
        genre_top_prob=float(style.genre_probs[gtop]),
        arousal=float(style.arousal),
        arousal_bucket=style.arousal_bucket,
        valence=_affect_valence(selector, audio, sr),
        move=style.move,
        intensity=float(style.intensity),
    )


# ─── aggregation ──────────────────────────────────────────────────────────

@dataclass
class ConditionStats:
    """What happened across all clips for one degradation condition."""
    condition: str
    family: str
    n: int
    genre_match_rate: float
    genre_top_prob_delta_mean: float   # mean(baseline_top_prob - degraded_top_prob)
    arousal_mae: float
    arousal_pearson: float
    valence_mae: float
    valence_pearson: float
    bucket_match_rate: float
    move_match_rate: float


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) != len(a):
        return float("nan")
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.std() < 1e-9 or bv.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def aggregate(
    baselines: dict[str, ClipTag],
    degraded:  dict[str, dict[str, ClipTag]],   # condition -> {clip -> tag}
) -> list[ConditionStats]:
    """Compute per-condition stats from the raw tags."""
    out: list[ConditionStats] = []
    for cond, clips in degraded.items():
        g_match, b_match, m_match = 0, 0, 0
        g_prob_deltas, a_ds, v_ds = [], [], []
        a_base, a_deg, v_base, v_deg = [], [], [], []
        family = ""
        n = 0
        for clip_name, tag in clips.items():
            base = baselines[clip_name]
            family = tag.family
            n += 1
            g_match += int(tag.genre_top == base.genre_top)
            b_match += int(tag.arousal_bucket == base.arousal_bucket)
            m_match += int(tag.move == base.move)
            g_prob_deltas.append(base.genre_top_prob - tag.genre_top_prob)
            a_ds.append(abs(tag.arousal - base.arousal))
            a_base.append(base.arousal); a_deg.append(tag.arousal)
            if base.valence is not None and tag.valence is not None:
                v_ds.append(abs(tag.valence - base.valence))
                v_base.append(base.valence); v_deg.append(tag.valence)
        out.append(ConditionStats(
            condition=cond,
            family=family,
            n=n,
            genre_match_rate=g_match / n if n else 0.0,
            genre_top_prob_delta_mean=statistics.fmean(g_prob_deltas) if g_prob_deltas else 0.0,
            arousal_mae=statistics.fmean(a_ds) if a_ds else 0.0,
            arousal_pearson=_pearson(a_base, a_deg),
            valence_mae=statistics.fmean(v_ds) if v_ds else 0.0,
            valence_pearson=_pearson(v_base, v_deg),
            bucket_match_rate=b_match / n if n else 0.0,
            move_match_rate=m_match / n if n else 0.0,
        ))
    return out


# ─── baseline-dispersion diagnostics (to call out "robust vs saturated") ──

@dataclass
class BaselineDispersion:
    """Spread of the baseline tags across the eval set — used to interpret
    the aggregate stats honestly."""
    genre_top_prob_mean: float
    genre_top_prob_std: float
    arousal_std: float
    arousal_bucket_counts: dict[str, int]
    move_counts: dict[str, int]
    n_unique_genres: int
    n_unique_moves: int
    n_clips: int


def baseline_dispersion(baselines: dict[str, ClipTag]) -> BaselineDispersion:
    tops  = [b.genre_top_prob for b in baselines.values()]
    arous = [b.arousal        for b in baselines.values()]
    bkts: dict[str, int] = {}
    mvs:  dict[str, int] = {}
    gs = set()
    for b in baselines.values():
        bkts[b.arousal_bucket] = bkts.get(b.arousal_bucket, 0) + 1
        mvs[b.move] = mvs.get(b.move, 0) + 1
        gs.add(b.genre_top)
    return BaselineDispersion(
        genre_top_prob_mean=statistics.fmean(tops) if tops else 0.0,
        genre_top_prob_std=statistics.pstdev(tops) if len(tops) > 1 else 0.0,
        arousal_std=statistics.pstdev(arous) if len(arous) > 1 else 0.0,
        arousal_bucket_counts=bkts,
        move_counts=mvs,
        n_unique_genres=len(gs),
        n_unique_moves=len(mvs),
        n_clips=len(baselines),
    )


# ─── reporting ────────────────────────────────────────────────────────────

def print_report(
    stats: list[ConditionStats],
    disp: BaselineDispersion,
) -> None:
    print()
    print("=" * 118)
    print(f"Noise robustness — v3.1 selector (PANNs CNN14 + heads), "
          f"n={disp.n_clips} clips")
    print("=" * 118)

    print(f"  BASELINE dispersion (what the clean set looked like):")
    print(f"    genre top prob: mean={disp.genre_top_prob_mean:.3f}  "
          f"std={disp.genre_top_prob_std:.3f}  "
          f"({disp.n_unique_genres} distinct top-1 genres)")
    print(f"    arousal std: {disp.arousal_std:.3f}  "
          f"buckets: {disp.arousal_bucket_counts}")
    print(f"    move counts: {disp.move_counts}  "
          f"({disp.n_unique_moves} distinct moves)")
    print()

    print(f"  {'condition':<28} {'n':>3}  "
          f"{'genre':>7} {'Δprob':>8}  "
          f"{'arousal':>9} {'r':>6}  "
          f"{'valence':>9} {'r':>6}  "
          f"{'bucket':>7} {'move':>7}")
    print(f"  {'':<28} {'':>3}  "
          f"{'match':>7} {'(mean)':>8}  "
          f"{'MAE':>9} {'(ρ)':>6}  "
          f"{'MAE':>9} {'(ρ)':>6}  "
          f"{'match':>7} {'match':>7}")
    print("  " + "-" * 112)

    # Preserve the order of default_conditions() so families stay grouped.
    for cs in stats:
        print(f"  {cs.condition:<28} {cs.n:>3}  "
              f"{100*cs.genre_match_rate:>6.1f}% {cs.genre_top_prob_delta_mean:>+8.3f}  "
              f"{cs.arousal_mae:>9.4f} {cs.arousal_pearson:>6.2f}  "
              f"{cs.valence_mae:>9.4f} {cs.valence_pearson:>6.2f}  "
              f"{100*cs.bucket_match_rate:>6.1f}% {100*cs.move_match_rate:>6.1f}%")
    print("=" * 118)
    print(
        "  'match' columns compare the degraded top-1 label to the clean baseline.\n"
        "  Δprob is mean(baseline_top_prob − degraded_top_prob): positive = lost confidence.\n"
        "  Pearson r below each MAE is the correlation between baseline and degraded values.\n"
        "  n is clips that completed the condition (identical for all rows unless an\n"
        "  individual codec encode errored).\n"
    )


# ─── runner ───────────────────────────────────────────────────────────────

def run(
    clips: list[Path],
    *,
    selector,
    conditions: list[Condition],
    seed: int = 42,
    progress: bool = True,
) -> tuple[dict[str, ClipTag], dict[str, dict[str, ClipTag]], list[ConditionStats], BaselineDispersion]:
    """Drive the whole experiment and return (baselines, degraded, stats, dispersion)."""
    import librosa

    baselines: dict[str, ClipTag] = {}
    degraded:  dict[str, dict[str, ClipTag]] = {c.label: {} for c in conditions}

    n_clips = len(clips)
    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        try:
            y, sr = librosa.load(str(clip), sr=None, mono=True)
        except Exception as exc:
            print(f"  [skip] {clip.name}: load failed ({exc})", flush=True)
            continue
        audio = np.asarray(y, dtype=np.float32)

        name = clip.stem
        base_tag = tag_one(selector, audio, int(sr),
                           clip=name, condition="clean", family="baseline")
        baselines[name] = base_tag

        for ci, cond in enumerate(conditions):
            try:
                deg = cond.apply(audio, int(sr), seed=seed + ci)
                tag = tag_one(selector, deg, int(sr),
                              clip=name, condition=cond.label, family=cond.family)
                degraded[cond.label][name] = tag
            except Exception as exc:
                print(f"  [skip] {clip.name} / {cond.label}: {exc}", flush=True)

        if progress:
            dt = time.time() - t0
            avg = dt / i
            eta = avg * (n_clips - i)
            print(f"  [{i}/{n_clips}] {name:<24} base={base_tag.genre_top}/"
                  f"{base_tag.move:<13} ETA {eta:5.0f}s", flush=True)

    stats = aggregate(baselines, degraded)
    disp  = baseline_dispersion(baselines)
    return baselines, degraded, stats, disp


# ─── entry point ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-genre", type=int, default=5,
                    help="GTZAN clips per genre (default 5 = 50 clips).")
    ap.add_argument("--no-showcase", action="store_true",
                    help="Skip the 4 showcase clips (GTZAN only).")
    ap.add_argument("--no-gtzan", action="store_true",
                    help="Skip the GTZAN set (showcase only).")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke run: 1 clip per genre + no showcase.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for stochastic degradations (default 42).")
    ap.add_argument("--panns-ckpt",   default=DEFAULT_PANNS_CKPT)
    ap.add_argument("--genre-head",   default=DEFAULT_GENRE_HEAD)
    ap.add_argument("--arousal-head", default=DEFAULT_AROUSAL_HEAD)
    ap.add_argument("--mood-source",  default="va", choices=["head", "va"])
    ap.add_argument("--json", default=None,
                    help="Also write the full per-tag + stats dump here.")
    args = ap.parse_args(argv)

    if args.quick:
        args.per_genre = 1
        args.no_showcase = True

    clips: list[Path] = []
    if not args.no_showcase:
        clips.extend(default_showcase_clips())
    if not args.no_gtzan:
        clips.extend(default_gtzan_clips(args.per_genre))
    clips = [c for c in clips if c.is_file()]
    if not clips:
        print("ERROR: no clips found. Populate data/raw/showcase and/or "
              "data/raw/gtzan_full/Data/genres_original.", file=sys.stderr)
        return 2

    print(f"[robustness] building selector...", flush=True)
    selector = _build_selector(
        panns_ckpt=args.panns_ckpt,
        genre_head=args.genre_head,
        arousal_head=args.arousal_head,
        mood_source=args.mood_source,
        allow_untrained=False,
    )

    conditions = default_conditions()
    print(f"[robustness] {len(clips)} clips × {len(conditions)} conditions "
          f"= {len(clips) * (len(conditions) + 1)} tagger calls", flush=True)

    baselines, degraded, stats, disp = run(
        clips, selector=selector, conditions=conditions, seed=args.seed,
    )

    print_report(stats, disp)

    if args.json:
        payload = {
            "n_clips": len(baselines),
            "conditions": [c.label for c in conditions],
            "baselines": {k: asdict(v) for k, v in baselines.items()},
            "degraded": {
                cond: {k: asdict(v) for k, v in clips.items()}
                for cond, clips in degraded.items()
            },
            "stats": [asdict(s) for s in stats],
            "dispersion": asdict(disp),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2),
                                   encoding="utf-8")
        print(f"[robustness] json → {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
