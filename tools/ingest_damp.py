"""tools/ingest_damp.py — DAMP-VSEP / DAMP-S-AG adapter for the M0' Tier 2
DAMP route (spec §9.x).

DAMP datasets are licensed (Smule Research Data License: non-commercial,
no redistribution). The raw files therefore live only under `data/` (which
is gitignored).

This module is responsible for two things:

1. **Discovery** (`discover_arrangements`): scan a normalized on-disk
   layout and group files into arrangements. Each arrangement subdir
   must contain at least one vocal AND at least one reference (a backing
   WAV, a reference MIDI, or both).

       <root>/<arrangement_id>/
           backing.wav        (optional)
           reference.midi     (optional; DAMP-S-AG MIDI route)
           vocal_<rendition_id>.<ext>   (one or more; libsndfile-readable)

2. **Extraction** (`extract_damp_s_ag`): stream a chosen subset of
   renditions out of the raw DAMP-S-AG tarball directly into the
   normalized layout above, without unpacking the rest of the archive
   and without modifying the source tarball.

CLI subcommands:

    list       List arrangements under <root> (the discovery output).
    damp-s-ag  Extract a subset of DAMP-S-AG into the normalized layout.

Run with `--help` for argument details.
"""
from __future__ import annotations
import argparse
import csv
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DampRendition:
    rendition_id: str
    vocal_wav: Path


@dataclass(frozen=True)
class DampArrangement:
    arrangement_id: str
    arrangement_dir: Path
    backing_wav: Path | None
    renditions: tuple[DampRendition, ...]
    reference_midi: Path | None = None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_arrangements(
    root: Path,
    *,
    backing_name: str = "backing.wav",
    midi_name: str = "reference.midi",
    vocal_glob: str = "vocal_*.wav",
) -> list[DampArrangement]:
    """List arrangement subdirectories under `root`.

    A subdir is included if it has at least one vocal matching `vocal_glob`
    AND at least one reference — either `backing_name` or `midi_name`. Both
    references may be present; the runner picks which to use.

    Rendition IDs are derived from each vocal filename by stripping the
    `vocal_` prefix and the extension (so `vocal_singer42.m4a` -> `singer42`).

    Dirs missing all references or all vocals are silently skipped.
    """
    arrangements: list[DampArrangement] = []
    if not root.exists():
        return arrangements
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        backing = d / backing_name
        midi = d / midi_name
        backing_p = backing if backing.exists() else None
        midi_p = midi if midi.exists() else None
        if backing_p is None and midi_p is None:
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
            backing_wav=backing_p,
            renditions=renditions,
            reference_midi=midi_p,
        ))
    return arrangements


def _rendition_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("vocal_"):
        return stem[len("vocal_"):]
    return stem


# --------------------------------------------------------------------------- #
# DAMP-S-AG subset extraction (no full unpack, source tarball untouched)
# --------------------------------------------------------------------------- #
def extract_damp_s_ag(
    tarball: Path,
    out_root: Path,
    *,
    arrangement_id: str = "amazing_grace",
    tsv_in_tar: str = "amazing_grace.tsv",
    midi_in_tar: str = "amazing_grace.midi",
    rendition_dir_in_tar: str = "amazing_grace",
    max_n: int | None = 100,
    headphones_only: bool = False,
    country: str | None = None,
) -> tuple[Path, list[str]]:
    """Stream-extract a subset of DAMP-S-AG into the normalized layout.

    Writes:
        <out_root>/<arrangement_id>/reference.midi
        <out_root>/<arrangement_id>/vocal_<perf_id>.m4a   (×N)

    Selection: walks the tarball in archive order; for each rendition
    encountered, looks up its performance_id in the TSV (read earlier in
    the same pass) and applies `--headphones-only` / `--country`. Stops
    after `max_n` accepted renditions. Reproducible (archive order is
    stable) and fast (no second pass).

    Returns `(arrangement_dir, [performance_ids in extraction order])`.

    Does NOT modify the source tarball or extract anything outside the
    normalized layout.
    """
    out_arr = Path(out_root) / arrangement_id
    out_arr.mkdir(parents=True, exist_ok=True)

    tsv_rows: dict[str, dict[str, str]] = {}
    extracted_midi = False
    selected_ids: list[str] = []
    rendition_prefix = f"{rendition_dir_in_tar}/"

    with tarfile.open(str(tarball), "r:gz") as tar:
        for member in tar:
            name = member.name
            # Skip macOS resource forks ("./._...") and any tar-internal
            # directory entries.
            if name.startswith("./") or member.isdir():
                continue
            if name == tsv_in_tar and not tsv_rows:
                with tar.extractfile(member) as f:    # type: ignore[union-attr]
                    text = f.read().decode("utf-8")
                reader = csv.DictReader(text.splitlines(), delimiter="\t")
                for row in reader:
                    tsv_rows[row["performance_id"]] = row
                continue
            if name == midi_in_tar and not extracted_midi:
                out_midi = out_arr / "reference.midi"
                with tar.extractfile(member) as src, open(out_midi, "wb") as dst:   # type: ignore[union-attr]
                    dst.write(src.read())
                extracted_midi = True
                continue
            if (name.startswith(rendition_prefix)
                    and name.endswith(".m4a")):
                if max_n is not None and len(selected_ids) >= max_n:
                    # We're done with renditions; we already have the TSV
                    # and MIDI from earlier in the archive.
                    break
                perf_id = name[len(rendition_prefix):-len(".m4a")]
                row = tsv_rows.get(perf_id)
                if row is None:
                    continue
                if headphones_only and row.get("headphones") != "1":
                    continue
                if country is not None and row.get("country") != country:
                    continue
                out_file = out_arr / f"vocal_{perf_id}.m4a"
                with tar.extractfile(member) as src, open(out_file, "wb") as dst:   # type: ignore[union-attr]
                    dst.write(src.read())
                selected_ids.append(perf_id)
    return out_arr, selected_ids


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    arrangements = discover_arrangements(
        Path(args.root),
        backing_name=args.backing_name,
        midi_name=args.midi_name,
        vocal_glob=args.vocal_glob,
    )
    if not arrangements:
        print(f"no arrangements found under {args.root}", file=sys.stderr)
        return 1
    for a in arrangements:
        refs = []
        if a.backing_wav is not None:
            refs.append("backing")
        if a.reference_midi is not None:
            refs.append("midi")
        print(f"{a.arrangement_id}: refs={','.join(refs)}  "
              f"renditions={len(a.renditions)}")
    return 0


def cmd_extract_damp_s_ag(args: argparse.Namespace) -> int:
    out_arr, ids = extract_damp_s_ag(
        Path(args.tarball),
        Path(args.out),
        arrangement_id=args.arrangement_id,
        tsv_in_tar=args.tsv_in_tar,
        midi_in_tar=args.midi_in_tar,
        rendition_dir_in_tar=args.rendition_dir_in_tar,
        max_n=args.max_n,
        headphones_only=args.headphones_only,
        country=args.country,
    )
    print(f"wrote {out_arr}  ({len(ids)} renditions + reference.midi)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    list_sp = sub.add_parser("list", help="list arrangements under root")
    list_sp.add_argument("--root", required=True)
    list_sp.add_argument("--backing-name", default="backing.wav")
    list_sp.add_argument("--midi-name", default="reference.midi")
    list_sp.add_argument("--vocal-glob", default="vocal_*.wav")
    list_sp.set_defaults(func=cmd_list)

    ext_sp = sub.add_parser(
        "damp-s-ag",
        help="extract a subset of DAMP-S-AG (Amazing Grace) into the "
             "normalized arrangement layout (does not modify tarball)",
    )
    ext_sp.add_argument("--tarball", required=True,
                        help="path to amazing_grace.tar.gz")
    ext_sp.add_argument("--out", required=True,
                        help="output root (e.g. data/m0p_t2_damp)")
    ext_sp.add_argument("--arrangement-id", default="amazing_grace")
    ext_sp.add_argument("--tsv-in-tar", default="amazing_grace.tsv")
    ext_sp.add_argument("--midi-in-tar", default="amazing_grace.midi")
    ext_sp.add_argument("--rendition-dir-in-tar", default="amazing_grace")
    ext_sp.add_argument("--max-n", type=int, default=100,
                        help="cap the number of renditions extracted "
                             "(default 100; use 0 or negative for all)")
    ext_sp.add_argument("--headphones-only", action="store_true",
                        help="keep only TSV rows with headphones == 1")
    ext_sp.add_argument("--country", default=None,
                        help="keep only TSV rows with this 2-letter country")
    ext_sp.set_defaults(func=cmd_extract_damp_s_ag)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    # `--max-n 0` / negative -> unlimited.
    if getattr(args, "max_n", None) is not None and args.max_n <= 0:
        args.max_n = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
