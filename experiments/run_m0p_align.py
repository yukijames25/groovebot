"""experiments/run_m0p_align.py — M0' Tier 1 alignment evaluation pipeline.

For each reference (audio, beats) pair under `--root`, for each time-stretch
rate, we:

  1. synthesize a warped query (`tools.synth_warp.synth_warp_audio`),
  2. compute reference + query alignment features
     (`groovebot.align.features.extract_align_features`),
  3. run offline DTW (`groovebot.align.dtw_align.OfflineDTWAligner.align`),
  4. map reference beats through the warp path -> recovered query beats,
  5. score recovered vs. true warped beats with `tools.eval_beat.score_beats`
     (same `mir_eval` harness used by the blind / fallback path).

Emits a per-track CSV, per-rate + overall aggregate CSVs, and per-track
overlay PNGs (warp path + audio with GT vs. recovered beats).

Designed for CPU on the Windows dev laptop. Only librosa is required — no
madmom, torch, or Demucs needed for Tier 1.

CLI:
    python -m experiments.run_m0p_align \\
        --root data/m0p_refs \\
        --out-dir data/m0p_work \\
        --feature chroma \\
        [--rates 0.9 0.95 1.0 1.05 1.1]
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import extract_align_features
from tools.eval_beat import load_beat_annotation, score_beats
from tools.synth_warp import DEFAULT_RATES, synth_warp_audio, warped_beat_times


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Discover `<stem>.wav` + `<stem>.beats` neighbours under `root`."""
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(root.rglob("*.wav")):
        beats = wav.with_suffix(".beats")
        if beats.exists():
            pairs.append((wav, beats))
    return pairs


def run_one(
    wav_path: Path,
    beats_path: Path,
    rate: float,
    feature_kind: str,
    aligner: OfflineDTWAligner,
    out_dir: Path,
    make_png: bool = True,
) -> dict:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != aligner.sample_rate:
        raise ValueError(
            f"{wav_path.name}: sr={sr}, aligner.sample_rate="
            f"{aligner.sample_rate}; resample upstream or rebuild the aligner"
        )
    ref_beats = load_beat_annotation(str(beats_path))

    query = synth_warp_audio(audio, sr, rate)
    true_query_beats = warped_beat_times(ref_beats, rate)

    t0 = time.perf_counter()
    ref_feats = extract_align_features(
        audio, sr, kind=feature_kind, hop_length=aligner.hop_length,
    )
    qry_feats = extract_align_features(
        query, sr, kind=feature_kind, hop_length=aligner.hop_length,
    )
    wp = aligner.align(qry_feats, ref_feats)
    recovered = aligner.map_reference_beats(wp, ref_beats)
    proc_sec = time.perf_counter() - t0

    audio_sec = len(query) / sr
    scores = score_beats(
        track=f"{wav_path.stem}_r{rate}",
        bpm=None,
        gt=true_query_beats,
        est=recovered,
        audio_sec=audio_sec,
        proc_sec=proc_sec,
    )

    if make_png:
        png_path = out_dir / f"{wav_path.stem}_r{rate}.png"
        _save_overlay_png(query, sr, true_query_beats, recovered, wp,
                          str(png_path), title=scores.track)

    row = asdict(scores)
    row["rate"] = rate
    row["feature_kind"] = feature_kind
    return row


def _save_overlay_png(
    query: np.ndarray, sr: int,
    gt: np.ndarray, est: np.ndarray, wp: np.ndarray,
    path: str, title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax_wav, ax_wp) = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(len(query)) / sr
    ax_wav.plot(t, query, linewidth=0.4, color="#888888")
    for x in gt:
        ax_wav.axvline(x, color="#1f77b4", linewidth=0.8, alpha=0.7)
    for x in est:
        ax_wav.axvline(x, color="#d62728", linewidth=0.8, alpha=0.7,
                       linestyle="--")
    ax_wav.legend(handles=[
        Line2D([0], [0], color="#888888", label="query audio"),
        Line2D([0], [0], color="#1f77b4", label="GT warped beat"),
        Line2D([0], [0], color="#d62728", linestyle="--",
               label="recovered beat"),
    ], loc="upper right")
    ax_wav.set_xlabel("time [s]")
    ax_wav.set_title(title)
    ax_wav.set_xlim(0, t[-1] if len(t) else 1.0)

    if wp.size:
        ax_wp.plot(wp[:, 1], wp[:, 0], color="#2ca02c", linewidth=0.7)
    ax_wp.set_xlabel("reference frame")
    ax_wp.set_ylabel("query frame")
    ax_wp.set_title("DTW warp path")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def aggregate(rows: list[dict]) -> tuple[list[dict], dict]:
    """Per-rate means + an overall mean. Keeps the row schema tiny so it's
    easy to display inline."""
    by_rate: dict[float, list[dict]] = {}
    for r in rows:
        by_rate.setdefault(r["rate"], []).append(r)
    per_rate: list[dict] = []
    for rate, group in sorted(by_rate.items()):
        per_rate.append({
            "rate": rate,
            "n": len(group),
            "f_mean": float(np.mean([g["f_measure"] for g in group])),
            "cmlt_mean": float(np.mean([g["cmlt"] for g in group])),
            "amlt_mean": float(np.mean([g["amlt"] for g in group])),
            "rt_mean": float(np.mean([g["rt_factor"] for g in group])),
        })
    overall = {
        "n": len(rows),
        "f_mean": float(np.mean([r["f_measure"] for r in rows])) if rows else 0.0,
        "cmlt_mean": float(np.mean([r["cmlt"] for r in rows])) if rows else 0.0,
        "amlt_mean": float(np.mean([r["amlt"] for r in rows])) if rows else 0.0,
        "rt_mean": float(np.mean([r["rt_factor"] for r in rows])) if rows else 0.0,
    }
    return per_rate, overall


def save_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run_pipeline(
    root: Path,
    out_dir: Path,
    feature_kind: str = "chroma",
    rates: Iterable[float] = DEFAULT_RATES,
    sample_rate: int = 22050,
    hop_length: int = 512,
    make_png: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    aligner = OfflineDTWAligner(sample_rate=sample_rate, hop_length=hop_length)
    rows: list[dict] = []
    for wav, beats in find_pairs(root):
        info = sf.info(str(wav))
        if info.samplerate != sample_rate:
            if verbose:
                print(f"skip {wav.name}: sr={info.samplerate}, "
                      f"need {sample_rate}", file=sys.stderr)
            continue
        for r in rates:
            row = run_one(wav, beats, r, feature_kind, aligner, out_dir,
                          make_png=make_png)
            rows.append(row)
            if verbose:
                print(f"  {row['track']}: F={row['f_measure']:.3f}  "
                      f"CMLt={row['cmlt']:.3f}  AMLt={row['amlt']:.3f}  "
                      f"RT={row['rt_factor']:.2f}x")
    per_rate, overall = aggregate(rows)
    save_csv(rows, out_dir / "m0p_per_track.csv")
    save_csv(per_rate, out_dir / "m0p_per_rate.csv")
    save_csv([overall], out_dir / "m0p_overall.csv")
    return rows, per_rate, overall


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True,
                    help="directory containing <stem>.wav + <stem>.beats pairs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature", choices=("chroma", "pitch"), default="chroma")
    ap.add_argument("--rates", type=float, nargs="*",
                    default=list(DEFAULT_RATES))
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--no-png", action="store_true")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_pipeline(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        feature_kind=args.feature,
        rates=args.rates,
        sample_rate=args.sr,
        hop_length=args.hop,
        make_png=not args.no_png,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
