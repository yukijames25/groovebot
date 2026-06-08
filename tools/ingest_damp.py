"""tools/ingest_damp.py — DAMP-VSEP / DAMP-S-AG adapter for the M0' Tier 2
DAMP route (spec §9.x).

DAMP datasets are licensed (Smule Research Data License: non-commercial,
no redistribution). The raw files therefore live only under `data/` (which
is gitignored). This module does NOT download them — it only walks an
on-disk normalized layout and groups the files into "arrangements" so the
evaluation runner can score multiple renditions of the same song.

**Normalized layout** (what `discover_arrangements` expects):

    <root>/<arrangement_id>/
        backing.wav
        vocal_<rendition_id>.wav   (one or more)

For DAMP-VSEP this is one segment per arrangement (the segment id becomes
the arrangement id, and the segment's stems become backing + a single
vocal). For DAMP-S-AG all renditions share one arrangement ("amazing_grace")
with one backing track and many `vocal_*.wav` files.

If your local DAMP dump uses a different file layout, write a small one-off
script that produces the normalized layout via copy or symlink; the runner
only depends on the structure above.
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DampRendition:
    rendition_id: str
    vocal_wav: Path


@dataclass(frozen=True)
class DampArrangement:
    arrangement_id: str
    arrangement_dir: Path
    backing_wav: Path
    renditions: tuple[DampRendition, ...]


def discover_arrangements(
    root: Path,
    *,
    backing_name: str = "backing.wav",
    vocal_glob: str = "vocal_*.wav",
) -> list[DampArrangement]:
    """List arrangement subdirectories under `root` that have a backing track
    plus one or more matching vocal files.

    Arrangement IDs are taken from the subdirectory name. Rendition IDs are
    derived from each vocal filename by stripping the `vocal_` prefix and
    the `.wav` suffix (so `vocal_singer42.wav` -> `singer42`).

    Dirs missing `backing.wav` or with zero matching vocals are silently
    skipped — they're noise for the runner, not errors.
    """
    arrangements: list[DampArrangement] = []
    if not root.exists():
        return arrangements
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        backing = d / backing_name
        if not backing.exists():
            continue
        vocals = sorted(d.glob(vocal_glob))
        if not vocals:
            continue
        renditions = tuple(
            DampRendition(
                rendition_id=_rendition_id_from_path(v),
                vocal_wav=v,
            )
            for v in vocals
        )
        arrangements.append(DampArrangement(
            arrangement_id=d.name,
            arrangement_dir=d,
            backing_wav=backing,
            renditions=renditions,
        ))
    return arrangements


def _rendition_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("vocal_"):
        return stem[len("vocal_"):]
    return stem


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True,
                    help="root dir containing arrangement subdirectories")
    ap.add_argument("--backing-name", default="backing.wav")
    ap.add_argument("--vocal-glob", default="vocal_*.wav")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    arrangements = discover_arrangements(
        Path(args.root),
        backing_name=args.backing_name,
        vocal_glob=args.vocal_glob,
    )
    if not arrangements:
        print(f"no arrangements found under {args.root}", file=sys.stderr)
        return 1
    for a in arrangements:
        print(f"{a.arrangement_id}: backing={a.backing_wav.name}  "
              f"renditions={len(a.renditions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
