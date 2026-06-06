"""
tools/eval_beat.py — M0 beat-tracking evaluation harness.

Compares the beats produced by BeatTrackerPerception (BeatNet, online/causal)
against a known click-grid ground truth, on the same WAV. The recording protocol
is: sing/hum into a mic while listening to a metronome on earphones, so the
metronome BPM is the ground truth.

Outputs:
    F-measure          mir_eval.beat.f_measure
    CMLt / AMLt        mir_eval.beat.continuity (CMLt = correct tempo & phase,
                                                  AMLt = allows tempo doubling/halving)
    RT-factor          process_sec / audio_sec  (<=1.0 means real-time-capable)
    waveform PNG       audio + ground-truth clicks + detected beats overlay

Subcommands:
    eval   --wav PATH (--bpm BPM | --beats FILE) [--offset SEC]
           [--beats-per-bar 4] [--out PNG] [--json]
    synth  --out PATH --bpm BPM [--seconds SEC] [--with-vocal]
                                          # generate a synthetic click(+vocal) WAV
                                          # for harness smoke tests only.

Ground-truth modes (mutually exclusive):
    --bpm 120        constant-BPM click grid (synthetic / click-sync recording)
    --beats FILE     one beat time (seconds) per line; supports comments and
                     tab-separated multi-column annotations (first column is
                     used). For public datasets with beat annotations
                     (GTZAN-Rhythm, Ballroom, Isophonics/Beatles, RWC) after
                     vocal-separation by tools/prep_dataset.py.

The real M0 evaluation runs on Colab/Kaggle (BeatNet stack is heavy). This
script is the same code path; locally it raises a clear error if BeatNet is
absent.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Ground-truth + scoring
# --------------------------------------------------------------------------- #
def click_grid(bpm: float, duration_sec: float,
               offset_sec: float = 0.0) -> np.ndarray:
    """Beat times for a constant-BPM click track, in seconds."""
    period = 60.0 / bpm
    n = int((duration_sec - offset_sec) / period) + 1
    return offset_sec + np.arange(n) * period


def load_beat_annotation(path: str) -> np.ndarray:
    """Read a beat annotation file: one beat time (seconds) per line.

    Tolerated formats — what `tools/prep_dataset.py` emits and what most
    public dataset annotations look like after passing through it:
      - lines starting with '#' or ';' are comments (skipped)
      - blank lines are skipped
      - lines may contain tab- or whitespace-separated columns; the FIRST
        column is read as the beat time
      - times must parse as float and be non-decreasing (sorted on read)
    """
    times: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            first = line.split()[0]
            try:
                times.append(float(first))
            except ValueError:
                raise ValueError(
                    f"{path}: cannot parse beat time from line: {raw!r}"
                ) from None
    return np.sort(np.asarray(times, dtype=float))


@dataclass
class BeatScores:
    """Per-track evaluation result. Keep flat so it serialises to one CSV row."""
    track: str
    bpm: Optional[float]    # None when GT came from --beats (variable tempo)
    f_measure: float
    cmlc: float
    cmlt: float
    amlc: float
    amlt: float
    n_gt: int
    n_est: int
    audio_sec: float
    proc_sec: float
    rt_factor: float        # proc_sec / audio_sec; <=1.0 means realtime-capable


def score_beats(track: str, bpm: Optional[float],
                gt: np.ndarray, est: np.ndarray,
                audio_sec: float, proc_sec: float) -> BeatScores:
    """Compute the standard mir_eval beat metrics + a latency proxy."""
    import mir_eval
    gt = mir_eval.beat.trim_beats(np.asarray(gt, dtype=float))
    est = mir_eval.beat.trim_beats(np.asarray(est, dtype=float))
    f = mir_eval.beat.f_measure(gt, est) if len(est) and len(gt) else 0.0
    if len(est) and len(gt):
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(gt, est)
    else:
        cmlc = cmlt = amlc = amlt = 0.0
    return BeatScores(
        track=track,
        bpm=bpm,
        f_measure=float(f),
        cmlc=float(cmlc),
        cmlt=float(cmlt),
        amlc=float(amlc),
        amlt=float(amlt),
        n_gt=int(len(gt)),
        n_est=int(len(est)),
        audio_sec=float(audio_sec),
        proc_sec=float(proc_sec),
        rt_factor=float(proc_sec / audio_sec) if audio_sec > 0 else float("inf"),
    )


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_overlay(wav: np.ndarray, sr: int, gt: np.ndarray, est: np.ndarray,
                 out_path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(wav)) / sr
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, wav, linewidth=0.4, color="#888888", label="audio")
    for x in gt:
        ax.axvline(x, color="#1f77b4", linewidth=0.8, alpha=0.7)
    for x in est:
        ax.axvline(x, color="#d62728", linewidth=0.8, alpha=0.7, linestyle="--")
    # Synthesize legend handles (axvline doesn't show up nicely otherwise).
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#888888", label="audio"),
        Line2D([0], [0], color="#1f77b4", label="ground-truth click"),
        Line2D([0], [0], color="#d62728", linestyle="--", label="detected beat"),
    ], loc="upper right")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1] if len(t) else 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthetic WAV (smoke-test fixture)
# --------------------------------------------------------------------------- #
def synth_click_wav(out_path: str, bpm: float, seconds: float,
                    sr: int = 22050, with_vocal: bool = False,
                    seed: int = 0) -> None:
    """A click track at `bpm`, optionally mixed with a wobbly tone meant to
    stand in for "vocal" energy. Smoke-test fixture only: lets us verify the
    eval harness end-to-end without recording anything.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float32)

    period = 60.0 / bpm
    click_len = int(0.020 * sr)
    click_env = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)
    click_tone = (np.sin(2 * np.pi * 1000.0 * np.arange(click_len) / sr)
                  .astype(np.float32) * click_env * 0.6)

    t = 0.0
    while t < seconds:
        i = int(t * sr)
        end = min(i + click_len, n)
        out[i:end] += click_tone[: end - i]
        t += period

    if with_vocal:
        # A wobbly low tone (~220 Hz) at modest volume to approximate vocal energy.
        time_axis = np.arange(n) / sr
        vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * time_axis)
        vocal = 0.15 * np.sin(2 * np.pi * 220.0 * vibrato * time_axis)
        vocal += 0.02 * rng.standard_normal(n)  # breath noise
        out += vocal.astype(np.float32)

    out = np.clip(out, -1.0, 1.0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, out, sr)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    audio_sec = len(wav) / sr

    # Ground truth: either a constant-BPM click grid or an annotation file.
    if args.beats is not None:
        gt = load_beat_annotation(args.beats)
        bpm_for_report: Optional[float] = None
    else:
        gt = click_grid(args.bpm, audio_sec, offset_sec=args.offset)
        bpm_for_report = args.bpm

    # Run the tracker. Import is lazy so the rest of the harness is testable
    # without BeatNet.
    from groovebot.perception import BeatTrackerPerception
    tracker = BeatTrackerPerception(sample_rate=sr,
                                    beats_per_bar=args.beats_per_bar)
    t0 = time.perf_counter()
    events = tracker.process_wav(args.wav)
    proc_sec = time.perf_counter() - t0
    est = np.array([e.time for e in events], dtype=float)

    scores = score_beats(track=os.path.basename(args.wav), bpm=bpm_for_report,
                         gt=gt, est=est,
                         audio_sec=audio_sec, proc_sec=proc_sec)

    if args.out:
        plot_overlay(wav, sr, gt, est, args.out,
                     title=f"{scores.track}  F={scores.f_measure:.3f}  "
                           f"CMLt={scores.cmlt:.3f}  AMLt={scores.amlt:.3f}  "
                           f"RT={scores.rt_factor:.2f}x")

    if args.json:
        print(json.dumps(asdict(scores), indent=2))
    else:
        print(_format_row_header())
        print(_format_row(scores))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    synth_click_wav(args.out, args.bpm, args.seconds,
                    with_vocal=args.with_vocal)
    print(f"wrote {args.out}  ({args.seconds:.1f}s @ {args.bpm} BPM"
          f"{', +vocal' if args.with_vocal else ''})")
    return 0


def _format_row_header() -> str:
    return (f"{'track':<32}  {'BPM':>6}  {'F':>6}  {'CMLt':>6}  {'AMLt':>6}  "
            f"{'RT-fac':>6}  {'n_est':>5}  {'n_gt':>5}")


def _format_row(s: BeatScores) -> str:
    bpm = f"{s.bpm:>6.1f}" if s.bpm is not None else f"{'-':>6}"
    return (f"{s.track:<32}  {bpm}  {s.f_measure:>6.3f}  "
            f"{s.cmlt:>6.3f}  {s.amlt:>6.3f}  {s.rt_factor:>6.2f}  "
            f"{s.n_est:>5d}  {s.n_gt:>5d}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate a WAV against click-grid or "
                                    "beat-annotation ground truth")
    e.add_argument("--wav", required=True, help="input WAV file")
    gt_group = e.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--bpm", type=float, default=None,
                          help="BPM of the click track the singer heard "
                               "(constant-BPM ground truth)")
    gt_group.add_argument("--beats", type=str, default=None,
                          help="annotation file: one beat time (sec) per line "
                               "(public-dataset ground truth, see spec §10.2)")
    e.add_argument("--offset", type=float, default=0.0,
                   help="seconds before the first click (only with --bpm)")
    e.add_argument("--beats-per-bar", type=int, default=4)
    e.add_argument("--out", type=str, default=None,
                   help="PNG path for waveform + beats overlay")
    e.add_argument("--json", action="store_true",
                   help="emit a JSON line instead of the text row")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("synth", help="generate a synthetic click(+vocal) WAV "
                                     "for harness smoke testing")
    s.add_argument("--out", required=True)
    s.add_argument("--bpm", type=float, required=True)
    s.add_argument("--seconds", type=float, default=8.0)
    s.add_argument("--with-vocal", action="store_true")
    s.set_defaults(func=cmd_synth)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as e:
        # Expected user-facing condition (e.g. BeatNet missing locally).
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
