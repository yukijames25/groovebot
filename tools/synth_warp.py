"""tools/synth_warp.py — apply known time-stretch ratios to a reference audio
to generate query audio with a ground-truth warp.

For M0' Tier 1 (spec §9.x). We already know reference beat times in seconds;
applying a constant time-stretch ratio r to the audio re-maps each beat:
    t_query = t_ref / r
(r > 1 = faster, shorter; r < 1 = slower, longer.)
The synthesized query is fed to DTW; reference beats mapped through the
recovered warp should land within tolerance of the known t_query.

CLI:
    python -m tools.synth_warp \\
        --wav song.wav --beats song.beats \\
        --out-dir data/m0p_warped/song \\
        [--rates 0.9 0.95 1.05 1.1]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


DEFAULT_RATES: tuple[float, ...] = (0.9, 0.95, 1.05, 1.1)


def synth_warp_audio(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch `audio` by `rate` (>1 faster/shorter). Mono output."""
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0 if a.shape[0] < a.shape[-1] else -1)
    a = a.astype(np.float32, copy=False)
    return librosa.effects.time_stretch(y=a, rate=float(rate))


def warped_beat_times(ref_beats_sec: np.ndarray, rate: float) -> np.ndarray:
    """Ground-truth query-side beat times under constant rate `rate`."""
    return np.asarray(ref_beats_sec, dtype=float) / float(rate)


def _rate_tag(rate: float) -> str:
    """Make a filesystem-safe tag for a rate (e.g. 0.95 -> '0.95')."""
    s = f"{rate:.4f}".rstrip("0").rstrip(".")
    return s if s else "1"


def synth_one(
    audio: np.ndarray,
    sr: int,
    ref_beats: np.ndarray,
    rate: float,
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write <stem>_r<tag>.wav and <stem>_r<tag>.beats for `rate`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y = synth_warp_audio(audio, sr, rate)
    b = warped_beat_times(ref_beats, rate)
    tag = _rate_tag(rate)
    wav_path = out_dir / f"{stem}_r{tag}.wav"
    beats_path = out_dir / f"{stem}_r{tag}.beats"
    sf.write(str(wav_path), y, sr)
    np.savetxt(str(beats_path), b, fmt="%.6f")
    return wav_path, beats_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--wav", required=True, help="reference WAV")
    ap.add_argument("--beats", required=True,
                    help="reference beat annotation (eval_beat --beats format)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for warped WAVs + warped beats")
    ap.add_argument("--rates", type=float, nargs="*",
                    default=list(DEFAULT_RATES),
                    help="time-stretch rates (default: 0.9 0.95 1.05 1.1)")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Reuse eval_beat's loader so the on-disk format stays in lockstep.
    from tools.eval_beat import load_beat_annotation
    ref_beats = load_beat_annotation(args.beats)
    stem = Path(args.wav).stem
    out_dir = Path(args.out_dir)
    for r in args.rates:
        wav_path, beats_path = synth_one(audio, sr, ref_beats, r, out_dir, stem)
        print(f"  rate={r}: {wav_path.name} "
              f"({sf.info(str(wav_path)).duration:.2f}s) "
              f"+ {beats_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
