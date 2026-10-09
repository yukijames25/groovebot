"""experiments/compare_bucket_vs_continuous.py — hard buckets vs smooth routing.

Three slices of the question "did going continuous actually buy us something?":

  (a) 54-clip inventory — showcase + GTZAN 5-per-genre. For each clip,
      run `GrooveStyleSelector` in BOTH modes and record (move, intensity)
      plus the raw arousal. Reports move-match rate and intensity delta.

  (b) ★ Boundary-stability perturbation. For each clip, synthesise ±ε
      perturbations to the arousal (ε ∈ {0.01, 0.02, 0.05}) and resolve
      (move, intensity) in each mode. Counts how many of the 7 jittered
      calls (base, ±ε for three ε values) agree with the base move, and
      measures the std of intensity across them. This is where the
      "hard-threshold brittleness" from the robustness experiment
      should show up most sharply.

  (c) ★ Noise re-measurement. Reuses `tools.degrade_audio` to apply SNR
      30/20/10/5 dB + telephone band to the same 54 clips, runs the
      tagger, and compares move / bucket / intensity in BOTH modes.
      Direct comparison with the earlier robustness run: did continuous
      recover the mid-bucket clips that bucket mode lost?

  (d) Intensity distribution. Dumps the sorted intensities so we can
      see whether continuous mode populates the gaps between 0.30 /
      0.60 / 0.90 or if everything just clusters back around them.

The script runs ALL four passes end-to-end. Pass `--pass b` (etc.) to
restrict to one slice if iterating.

Honest limit: the arousal values driving (b) are perturbations of the
baseline arousal estimate, not of real audio. For a real-audio
perturbation study, (c) is the measurement — but it conflates the
noise's effect on the backbone with the mode's response. (b) is the
clean, isolated test of the routing itself.

Usage:
    python -m experiments.compare_bucket_vs_continuous
    python -m experiments.compare_bucket_vs_continuous --per-genre 3 --json out.json
    python -m experiments.compare_bucket_vs_continuous --pass b   # perturbation only
    python -m experiments.compare_bucket_vs_continuous --quick    # smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

from experiments.render_groove import (
    DEFAULT_AROUSAL_HEAD,
    DEFAULT_GENRE_HEAD,
    DEFAULT_PANNS_CKPT,
    _build_selector,
)
from experiments.run_noise_robustness import (
    default_gtzan_clips,
    default_showcase_clips,
)
from groovebot.style.attributes import arousal_bucket
from groovebot.style.table import (
    _HIGH_ANCHOR,
    _LOW_ANCHOR,
    _MID_ANCHOR,
    select_move,
)


# ─── per-clip base tagging ────────────────────────────────────────────────

@dataclass
class BaseTag:
    """Clean-audio tag plus the ingredients downstream select_move needs."""
    clip: str
    genre: str
    arousal: float
    valence: float | None
    arousal_bucket: str
    mood_probs: dict[str, float]
    genre_probs: dict[str, float]
    # per-mode outputs on CLEAN audio
    move_bucket: str
    inten_bucket: float
    move_cont: str
    inten_cont: float


def _mood_probs_from_style(style) -> dict[str, float]:
    # Return a dict copy, since the GrooveStyle instance keeps the live view.
    return {k: float(v) for k, v in style.mood_probs.items()}


def _tag_clean(selector, audio: np.ndarray, sr: int, name: str) -> BaseTag:
    """Run the selector on clean audio, record EVERY ingredient so later
    passes can replay select_move with perturbed arousal without
    re-running the heavy backbone."""
    # Selector drives select_move internally in whatever mode it was
    # built with; here we build with bucket mode and additionally
    # evaluate select_move manually in continuous mode on the same
    # (genre, mood_probs, arousal).
    style = selector.select(audio, int(sr))
    g_top = max(style.genre_probs, key=style.genre_probs.get)
    mood = _mood_probs_from_style(style)
    gp   = {k: float(v) for k, v in style.genre_probs.items()}
    # The selector was built in bucket mode — its (move, intensity) IS
    # the bucket result, which saves a redundant table call here.
    move_b, i_b = style.move, float(style.intensity)
    move_c, i_c = select_move(
        g_top, style.arousal_bucket, mood,
        arousal=float(style.arousal), mode="continuous",
    )
    return BaseTag(
        clip=name,
        genre=g_top,
        arousal=float(style.arousal),
        valence=None,
        arousal_bucket=style.arousal_bucket,
        mood_probs=mood,
        genre_probs=gp,
        move_bucket=move_b,
        inten_bucket=i_b,
        move_cont=move_c,
        inten_cont=i_c,
    )


# ─── (a) 54-clip inventory ────────────────────────────────────────────────

@dataclass
class ModeDeltaStats:
    """Aggregate of 'how do the two modes differ on the same 54 clips?'"""
    n: int
    move_match_rate: float
    intensity_mae: float
    intensity_pearson: float
    disagreement_clips: list[str]
    intensity_hist_bucket:    dict[str, int]
    intensity_hist_continuous: dict[str, int]


def _histogram(xs: Sequence[float]) -> dict[str, int]:
    """Bin a list of intensities into 10 bins over [0, 1]."""
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    counts = {f"[{bins[i]:.1f},{bins[i+1]:.2f})": 0 for i in range(len(bins) - 1)}
    for x in xs:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                key = f"[{bins[i]:.1f},{bins[i+1]:.2f})"
                counts[key] += 1
                break
    return counts


def inventory(bases: list[BaseTag]) -> ModeDeltaStats:
    n = len(bases)
    match = sum(int(b.move_bucket == b.move_cont) for b in bases)
    diffs = [abs(b.inten_bucket - b.inten_cont) for b in bases]
    disagree = [b.clip for b in bases if b.move_bucket != b.move_cont]
    i_b = np.array([b.inten_bucket for b in bases], dtype=np.float64)
    i_c = np.array([b.inten_cont   for b in bases], dtype=np.float64)
    pear = float(np.corrcoef(i_b, i_c)[0, 1]) if n >= 3 else float("nan")
    return ModeDeltaStats(
        n=n,
        move_match_rate=match / n if n else 0.0,
        intensity_mae=statistics.fmean(diffs) if diffs else 0.0,
        intensity_pearson=pear,
        disagreement_clips=disagree,
        intensity_hist_bucket=_histogram(i_b.tolist()),
        intensity_hist_continuous=_histogram(i_c.tolist()),
    )


# ─── (b) perturbation stability ───────────────────────────────────────────

@dataclass
class PerturbStats:
    """Per-mode stability under small arousal perturbations."""
    epsilon: float
    n_clips: int
    move_match_rate: float          # fraction of (clip, sign) pairs where
                                    # the perturbed move == base move
    intensity_std_mean: float       # mean per-clip std of intensity across
                                    # {-ε, 0, +ε}
    intensity_jump_max: float       # worst-case per-clip |intensity(+ε) - intensity(-ε)|
    worst_clip: str                 # the clip whose intensity jumped the most


def perturbation_pass(bases: list[BaseTag],
                      epsilons: Sequence[float]) -> dict[str, list[PerturbStats]]:
    """For each epsilon, for each clip, replay select_move at
    (base_arousal + sign·eps) for sign ∈ {-1, 0, +1} in each mode."""
    out: dict[str, list[PerturbStats]] = {"bucket": [], "continuous": []}
    for mode in ("bucket", "continuous"):
        for eps in epsilons:
            total_pairs = 0
            match_pairs = 0
            inten_stds: list[float] = []
            worst_jump = 0.0
            worst_name = ""
            for b in bases:
                triples_m: list[str] = []
                triples_i: list[float] = []
                for sign in (-1, 0, +1):
                    a_perturbed = max(0.0, min(1.0, b.arousal + sign * eps))
                    bk = arousal_bucket(a_perturbed)
                    if mode == "bucket":
                        m, i = select_move(b.genre, bk, b.mood_probs, mode="bucket")
                    else:
                        m, i = select_move(b.genre, bk, b.mood_probs,
                                           arousal=a_perturbed, mode="continuous")
                    triples_m.append(m)
                    triples_i.append(i)
                base_move = triples_m[1]
                for s_idx in (0, 2):
                    total_pairs += 1
                    match_pairs += int(triples_m[s_idx] == base_move)
                inten_stds.append(statistics.pstdev(triples_i))
                jump = abs(triples_i[2] - triples_i[0])
                if jump > worst_jump:
                    worst_jump, worst_name = jump, b.clip
            out[mode].append(PerturbStats(
                epsilon=eps,
                n_clips=len(bases),
                move_match_rate=match_pairs / total_pairs if total_pairs else 0.0,
                intensity_std_mean=statistics.fmean(inten_stds) if inten_stds else 0.0,
                intensity_jump_max=worst_jump,
                worst_clip=worst_name,
            ))
    return out


# ─── (c) noise re-measurement ─────────────────────────────────────────────

@dataclass
class NoiseModeStats:
    condition: str
    family: str
    n: int
    bucket_move_match: float
    continuous_move_match: float
    bucket_intensity_mae: float
    continuous_intensity_mae: float


def noise_pass(selector, clips: list[Path],
               bases: dict[str, BaseTag],
               seed: int = 42) -> list[NoiseModeStats]:
    """Re-run the four worst-degradation conditions from the noise exp,
    tagging each in both modes."""
    import librosa
    from tools.degrade_audio import (
        NoiseCondition, BandCondition, default_conditions,
    )
    # Pick the subset of conditions the user flagged as most interesting
    # (noise sweep + telephone). Pulling by label keeps the comparable
    # names identical to the earlier run's report.
    picked_labels = {"SNR 30 dB", "SNR 20 dB", "SNR 10 dB", "SNR  5 dB",
                     "telephone (300-3400 Hz)"}
    conds = [c for c in default_conditions() if c.label in picked_labels]

    out: list[NoiseModeStats] = []
    for ci, cond in enumerate(conds):
        n = 0
        b_match = 0
        c_match = 0
        b_err: list[float] = []
        c_err: list[float] = []
        for p in clips:
            name = p.stem
            if name not in bases:
                continue
            try:
                y, sr = librosa.load(str(p), sr=None, mono=True)
            except Exception:
                continue
            audio = np.asarray(y, dtype=np.float32)
            try:
                deg = cond.apply(audio, int(sr), seed=seed + ci)
            except Exception:
                continue

            # Re-run tagger on degraded audio. Selector is in bucket
            # mode, so its (move, intensity) is bucket's answer. For
            # continuous mode we replay select_move on the same
            # (genre, arousal, mood).
            style = selector.select(deg, int(sr))
            g_top = max(style.genre_probs, key=style.genre_probs.get)
            mood  = _mood_probs_from_style(style)
            move_b_deg, i_b_deg = style.move, float(style.intensity)
            move_c_deg, i_c_deg = select_move(
                g_top, style.arousal_bucket, mood,
                arousal=float(style.arousal), mode="continuous",
            )

            base = bases[name]
            n += 1
            b_match += int(move_b_deg == base.move_bucket)
            c_match += int(move_c_deg == base.move_cont)
            b_err.append(abs(i_b_deg - base.inten_bucket))
            c_err.append(abs(i_c_deg - base.inten_cont))

        out.append(NoiseModeStats(
            condition=cond.label,
            family=cond.family,
            n=n,
            bucket_move_match=b_match / n if n else 0.0,
            continuous_move_match=c_match / n if n else 0.0,
            bucket_intensity_mae=statistics.fmean(b_err) if b_err else 0.0,
            continuous_intensity_mae=statistics.fmean(c_err) if c_err else 0.0,
        ))
    return out


# ─── pretty print ─────────────────────────────────────────────────────────

def print_report(
    bases: list[BaseTag],
    inv: ModeDeltaStats,
    perturbs: dict[str, list[PerturbStats]] | None,
    noise: list[NoiseModeStats] | None,
) -> None:
    print()
    print("=" * 118)
    print(f"Bucket vs continuous arousal routing — v3.1 selector, n={len(bases)} clips")
    print("=" * 118)
    print(f"  anchors: low={_LOW_ANCHOR}  mid={_MID_ANCHOR}  high={_HIGH_ANCHOR}")
    print()

    print("-- (a) 54-clip inventory: do the two modes pick the same move on CLEAN audio? --")
    print(f"    move match rate     : {100*inv.move_match_rate:5.1f}%")
    print(f"    intensity MAE       : {inv.intensity_mae:.3f}")
    print(f"    intensity Pearson ρ : {inv.intensity_pearson:.3f}")
    print(f"    disagreement clips  : {len(inv.disagreement_clips)}")
    for c in inv.disagreement_clips[:10]:
        b = next(b for b in bases if b.clip == c)
        print(f"      {c:<24} a={b.arousal:.2f} bucket={b.move_bucket:<14} cont={b.move_cont}")
    print()
    print("    intensity bin counts (left = bucket, right = continuous):")
    bins = list(inv.intensity_hist_bucket.keys())
    for k in bins:
        bb = inv.intensity_hist_bucket[k]
        cc = inv.intensity_hist_continuous[k]
        bar_b = "#" * bb; bar_c = "#" * cc
        print(f"      {k:<16} bucket {bb:>3} {bar_b:<18}  |  continuous {cc:>3} {bar_c}")
    print()

    if perturbs is not None:
        print("-- (b) ★ perturbation stability: arousal jitter ±ε, measured on both modes --")
        print(f"    {'ε':>7}  {'mode':<10}  {'move match':>12}  {'inten std (mean)':>18}  "
              f"{'inten jump max':>16}  worst clip")
        print(f"    {'':>7}  {'':<10}  {'(perturbed vs base)':>22}")
        for eps_i in range(len(perturbs["bucket"])):
            for mode in ("bucket", "continuous"):
                p = perturbs[mode][eps_i]
                print(f"    ±{p.epsilon:>5.2f}  {mode:<10}  {100*p.move_match_rate:>10.1f}%  "
                      f"{p.intensity_std_mean:>18.4f}  "
                      f"{p.intensity_jump_max:>16.3f}  {p.worst_clip}")
            print()

    if noise is not None:
        print("-- (c) ★ noise re-measurement: same 5 conditions, both modes --")
        print(f"    {'condition':<28} {'n':>3}  "
              f"{'bucket move':>12} {'cont. move':>12}  "
              f"{'bucket iMAE':>12} {'cont. iMAE':>11}")
        print("    " + "-" * 102)
        for s in noise:
            print(f"    {s.condition:<28} {s.n:>3}  "
                  f"{100*s.bucket_move_match:>11.1f}% {100*s.continuous_move_match:>11.1f}%  "
                  f"{s.bucket_intensity_mae:>12.3f} {s.continuous_intensity_mae:>11.3f}")
        print()
    print("=" * 118)


# ─── runner ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-genre", type=int, default=5,
                    help="GTZAN clips per genre (default 5 = 50 clips).")
    ap.add_argument("--no-showcase", action="store_true")
    ap.add_argument("--no-gtzan",    action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke: 1 clip/genre, no showcase, skip noise pass.")
    ap.add_argument("--pass", dest="passes", default="abcd",
                    help="Which passes to run (letters from a/b/c/d). Default: all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panns-ckpt",   default=DEFAULT_PANNS_CKPT)
    ap.add_argument("--genre-head",   default=DEFAULT_GENRE_HEAD)
    ap.add_argument("--arousal-head", default=DEFAULT_AROUSAL_HEAD)
    ap.add_argument("--mood-source",  default="va", choices=["head", "va"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.per_genre = 1
        args.no_showcase = True
        args.passes = "ab"   # noise pass is the slow one; skip for smoke

    clips: list[Path] = []
    if not args.no_showcase:
        clips.extend(default_showcase_clips())
    if not args.no_gtzan:
        clips.extend(default_gtzan_clips(args.per_genre))
    clips = [c for c in clips if c.is_file()]
    if not clips:
        print("ERROR: no clips found.", file=sys.stderr)
        return 2

    print(f"[compare] building selector (bucket mode)...", flush=True)
    selector = _build_selector(
        panns_ckpt=args.panns_ckpt,
        genre_head=args.genre_head,
        arousal_head=args.arousal_head,
        mood_source=args.mood_source,
        allow_untrained=False,
    )

    import librosa
    bases: list[BaseTag] = []
    print(f"[compare] (a) baseline-tagging {len(clips)} clips...", flush=True)
    t0 = time.time()
    for i, c in enumerate(clips, 1):
        try:
            y, sr = librosa.load(str(c), sr=None, mono=True)
        except Exception as exc:
            print(f"  [skip] {c.name}: {exc}", flush=True)
            continue
        audio = np.asarray(y, dtype=np.float32)
        tag = _tag_clean(selector, audio, int(sr), c.stem)
        bases.append(tag)
        if i % 10 == 0 or i == len(clips):
            dt = time.time() - t0
            print(f"  [{i}/{len(clips)}] ETA {(dt/i)*(len(clips)-i):.0f}s", flush=True)

    inv = inventory(bases)

    perturbs = None
    if "b" in args.passes:
        print(f"[compare] (b) perturbation stability pass...", flush=True)
        perturbs = perturbation_pass(bases, epsilons=(0.01, 0.02, 0.05))

    noise_stats = None
    if "c" in args.passes:
        print(f"[compare] (c) noise re-measurement pass...", flush=True)
        base_by_name = {b.clip: b for b in bases}
        noise_stats = noise_pass(
            selector, clips, base_by_name, seed=args.seed,
        )

    print_report(bases, inv, perturbs, noise_stats)

    if args.json:
        payload = {
            "n_clips": len(bases),
            "anchors": {
                "low":  _LOW_ANCHOR, "mid": _MID_ANCHOR, "high": _HIGH_ANCHOR,
            },
            "bases":      [asdict(b) for b in bases],
            "inventory":  asdict(inv),
            "perturbation": {
                m: [asdict(p) for p in perturbs[m]]
                for m in (perturbs or {})
            },
            "noise": [asdict(s) for s in (noise_stats or [])],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2),
                                   encoding="utf-8")
        print(f"[compare] json → {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
