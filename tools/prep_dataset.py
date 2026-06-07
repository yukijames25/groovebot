"""
tools/prep_dataset.py — public beat-annotated datasets -> our --beats format.

Pipeline (spec §10.2 "公開データ方式"):

  raw dataset/                       # downloaded from each dataset's source
    audio/<track>.wav
    annotations/<track>.{beats,onsets,txt,lab,...}
            │
            ▼  prep_dataset.py convert-ann
  out/
    beats/<track>.beats              # 1 beat time (sec) per line; --beats input
            │
            ▼  prep_dataset.py separate    (Colab/Kaggle: needs Demucs+torch)
    vocal/<track>.wav                # vocal-only stem; --wav input

Then on Colab/Kaggle:

    python -m tools.eval_beat eval --wav out/vocal/<track>.wav \
                                   --beats out/beats/<track>.beats \
                                   --out  out/eval/<track>.png

Why split convert-ann from separate?
- convert-ann is pure-Python parsing. It runs on the Windows dev box and is
  covered by unit tests (`tests/test_prep_dataset.py`).
- separate calls Demucs (heavy: torch + ~1 GB models). Lives behind a lazy
  import and raises a clear error locally; you run it on Colab/Kaggle.

Supported annotation formats (extend as datasets are added):
- Ballroom (Gouyon 2006): `<time>\\t<beat_in_bar>` per line.
- GTZAN-Rhythm (Marchand 2015): `<time>\\t<beat_in_bar>` (same shape).
- Isophonics/Beatles (Mauch 2009): `<start>\\t<end>\\t<label>` with beats per
  line; we read column 0.
- RWC Popular AIST annotations: `<time> <beat_in_bar>` (whitespace).
- Dagstuhl ChoirSet: a cappella control — beat annotations as `<time>` per line.

All converters output the same one-column `*.beats` text — that is the input
format `tools/eval_beat.py --beats FILE` expects (and `load_beat_annotation`
parses).
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Callable


_DEMUCS_INSTALL_HINT = (
    "demucs is not installed. Vocal separation runs on Colab/Kaggle:\n"
    "  !pip install -r requirements-experiments.txt\n"
    "Locally, use `convert-ann` (annotation conversion only) — it has no heavy deps."
)


# --------------------------------------------------------------------------- #
# Annotation converters (pure Python, locally testable).
# Each parser returns a sorted list[float] of beat times in seconds. The CLI
# then writes one time per line to `<out>.beats`.
# --------------------------------------------------------------------------- #
def parse_ballroom(path: str) -> list[float]:
    """Ballroom (Gouyon 2006) — `<time>\\t<beat_in_bar>` per line.

    The same shape covers GTZAN-Rhythm (Marchand 2015) and RWC Popular AIST
    annotations (whitespace instead of tab, also handled).
    """
    times: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            cols = line.split()
            try:
                times.append(float(cols[0]))
            except (ValueError, IndexError):
                raise ValueError(
                    f"{path}: cannot parse beat time from: {raw!r}"
                ) from None
    times.sort()
    return times


def parse_isophonics_beats(path: str) -> list[float]:
    """Isophonics/Beatles — beat files have one beat per line; first column is
    the beat time. Some variants include `<time> <bar.beat>`; the first
    whitespace-separated token is the time.
    """
    return parse_ballroom(path)   # same first-column rule


def parse_single_column(path: str) -> list[float]:
    """Generic single-column `<time>` per line. Use for Dagstuhl-style beat
    annotations and any dataset already exported to one-column floats.
    """
    return parse_ballroom(path)


# Stubs for datasets whose distribution we have not yet wired up — declare the
# expected format so the next person knows what to implement.
def parse_gtzan_rhythm(path: str) -> list[float]:
    """GTZAN-Rhythm (Marchand & Peeters, 2015): `<time>\\t<beat_in_bar>` per
    line. Same shape as Ballroom."""
    return parse_ballroom(path)


def parse_rwc_popular(path: str) -> list[float]:
    """RWC Popular AIST annotations: whitespace-separated, beats in column 0
    (other columns: beat-in-bar, measure index, …). Verify with the AIST
    README at https://staff.aist.go.jp/m.goto/RWC-MDB/."""
    return parse_ballroom(path)


_PARSERS: dict[str, Callable[[str], list[float]]] = {
    "ballroom": parse_ballroom,
    "gtzan": parse_gtzan_rhythm,
    "rwc": parse_rwc_popular,
    "isophonics": parse_isophonics_beats,
    "dagstuhl": parse_single_column,
    "generic": parse_single_column,
}


def write_beats_file(times: list[float], out_path: str) -> None:
    """Emit the one-column `--beats` format (see `eval_beat.load_beat_annotation`)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# beat times in seconds, one per line\n")
        for t in times:
            f.write(f"{t:.6f}\n")


# --------------------------------------------------------------------------- #
# Demucs (vocal separation) — heavy; experiments side only.
# --------------------------------------------------------------------------- #
def _require_demucs():
    try:
        import demucs.separate  # type: ignore
        return demucs.separate
    except Exception as e:
        raise RuntimeError(_DEMUCS_INSTALL_HINT) from e


def _find_vocal_output(out_root: Path, model: str, in_stem: str) -> Path:
    """Locate Demucs's vocals.wav for an input named `<in_stem>.wav`.

    Canonical layout (Demucs 4.x): `<out_root>/<model>/<in_stem>/vocals.wav`.

    Fallback: GTZAN audio stems contain a dot (`blues.00000`) which Demucs
    has been observed to either preserve or rewrite (e.g. to `blues_00000`).
    So if the canonical path is missing, scan `<out_root>/<model>` (or the
    whole `out_root` if the model dir is absent) for any `vocals.wav` whose
    parent directory name matches the stem or its dot-to-underscore variant.
    """
    canonical = out_root / model / in_stem / "vocals.wav"
    if canonical.exists():
        return canonical
    search_root = out_root / model if (out_root / model).exists() else out_root
    variants = {in_stem, in_stem.replace(".", "_")}
    for cand in search_root.rglob("vocals.wav"):
        if cand.parent.name in variants:
            return cand
    raise RuntimeError(
        f"Demucs ran but vocals.wav was not found.\n"
        f"  expected: {canonical}\n"
        f"  searched: {search_root} (recursive)\n"
        f"  variants tried for parent-dir match: {sorted(variants)}\n"
        f"Check the model name ({model!r}) and the output layout."
    )


def separate_vocal(in_wav: str, out_dir: str, model: str = "htdemucs") -> str:
    """Run Demucs (two-stems: vocals) and return the path to vocals.wav.

    Locally this raises RuntimeError because demucs is not installed. The
    Colab/Kaggle notebook calls it after `pip install -r requirements-
    experiments.txt`.

    Demucs canonically writes to `<out_dir>/<model>/<track_stem>/vocals.wav`
    (with `other.wav` for the rest). When the input stem contains a dot
    (GTZAN's `blues.00000`) some Demucs versions rewrite the subdir name, so
    we use `_find_vocal_output` to locate it robustly.
    """
    sep = _require_demucs()       # local: RuntimeError with install hint.
    in_path = Path(in_wav)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sep.main([
        "--two-stems=vocals",
        "-n", model,
        "-o", str(out_root),
        str(in_path),
    ])
    return str(_find_vocal_output(out_root, model, in_path.stem))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_convert_ann(args: argparse.Namespace) -> int:
    if args.dataset not in _PARSERS:
        print(f"unknown dataset: {args.dataset}. "
              f"choices: {sorted(_PARSERS.keys())}", file=sys.stderr)
        return 2
    parser_fn = _PARSERS[args.dataset]
    times = parser_fn(args.input)
    write_beats_file(times, args.output)
    print(f"wrote {args.output}  ({len(times)} beats)")
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    try:
        out = separate_vocal(args.wav, args.out_dir, model=args.model)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote vocal stem: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert-ann",
                       help="convert a public-dataset beat annotation to "
                            "our --beats format (one time per line)")
    c.add_argument("--dataset", required=True,
                   choices=sorted(_PARSERS.keys()))
    c.add_argument("--input", required=True, help="source annotation file")
    c.add_argument("--output", required=True, help="output .beats file")
    c.set_defaults(func=cmd_convert_ann)

    s = sub.add_parser("separate",
                       help="run Demucs vocal separation (Colab/Kaggle only)")
    s.add_argument("--wav", required=True, help="input mixed-audio WAV")
    s.add_argument("--out-dir", required=True,
                   help="directory where the vocal stem will be written")
    s.add_argument("--model", default="htdemucs",
                   help="Demucs model name (default htdemucs)")
    s.set_defaults(func=cmd_separate)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
