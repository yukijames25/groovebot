"""tools/ingest_mtg_moodtheme.py — MTG-Jamendo moodtheme manifest builder.

This script does NOT download anything. It walks an *already-downloaded*
MTG-Jamendo `autotagging_moodtheme` tree, reads the official TSV, joins
the per-track tag list to `groovebot.style.mood_mapping`, and writes a
small CSV manifest the v3 mood trainer can read:

    path,mtg_track_id,artist_id,mood_class

Use the **official** MTG download script (it is the only sanctioned way
to fetch the audio; the dataset is CC but redistribution from us would
be impolite):

  git clone https://github.com/MTG/mtg-jamendo-dataset
  cd mtg-jamendo-dataset
  python scripts/download/download.py \\
      --dataset autotagging_moodtheme \\
      --type audio-low \\
      --output-dir <YOUR>/data/raw/mtg_moodtheme \\
      --from 00 --to 02     # bound to 3 archives -> ~3000 clips, ~1.8 GB

The bounding flags `--from / --to` are the project-policy guardrails:
we want a few thousand clips for the v3 head, not the full 18k-clip
moodtheme subset. The trainer is happy with whatever subset lands.

Output manifest schema is intentionally minimal so it stays stable when
the MTG side changes (they revise the tag list / TSV columns
occasionally). For the trainer, only `path` and `mood_class` matter;
`mtg_track_id` is kept for traceability and `artist_id` is what the
artist-non-overlap split groups by.
"""
from __future__ import annotations
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from groovebot.style.mood_mapping import (
    MTG_MOODTHEME_TAGS,
    coverage_check,
    resolve_clip_moods,
)


@dataclass
class MtgClip:
    track_id: str
    artist_id: str
    rel_path: str
    tags: list[str]      # raw "mood/theme---<tag>" stripped to just the tag
    mood_class: str | None  # resolved by the mapping; None to skip


def _strip_tag_prefix(t: str) -> str:
    """MTG tags ship as `mood/theme---epic` etc; we keep just the tag."""
    return t.rsplit("---", 1)[-1].strip().lower()


def parse_tsv(tsv_path: Path) -> list[tuple[str, str, str, list[str]]]:
    """Read MTG's autotagging_moodtheme TSV.

    Schema as of the master branch (subject to upstream change):

        TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS

    All columns tab-separated. The TAGS column is space-separated raw
    `mood/theme---<tag>` strings.

    Returns `[(track_id, artist_id, rel_path, [tag, ...]), ...]`.
    """
    rows: list[tuple[str, str, str, list[str]]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return rows
        # Locate the columns we need by header name, since MTG has
        # historically reordered them.
        try:
            i_track = header.index("TRACK_ID")
            i_artist = header.index("ARTIST_ID")
            i_path = header.index("PATH")
            i_tags = header.index("TAGS")
        except ValueError as e:
            raise ValueError(
                f"unexpected MTG TSV header: {header!r} — schema changed?"
            ) from e
        for row in reader:
            if not row or len(row) <= max(i_track, i_artist, i_path, i_tags):
                continue
            tags = [
                _strip_tag_prefix(t) for t in row[i_tags].split()
                if t.strip()
            ]
            rows.append((row[i_track], row[i_artist], row[i_path], tags))
    return rows


def build_manifest(
    audio_root: Path,
    tsv_path: Path,
    *,
    conflict_rule: str = "drop_on_disagreement",
    require_audio_present: bool = True,
) -> tuple[list[MtgClip], dict[str, int]]:
    """Apply the mood mapping to every TSV row and return the kept
    clips plus a small reason histogram for the dropped ones.

    Reasons:
      - `no_mood_tag`     : every tag is a theme / ambiguous
      - `conflict`        : `drop_on_disagreement` and the moods disagreed
      - `audio_missing`   : the .mp3 / .ogg is not in `audio_root`
    """
    rows = parse_tsv(tsv_path)
    kept: list[MtgClip] = []
    reasons: dict[str, int] = {"no_mood_tag": 0, "conflict": 0, "audio_missing": 0}
    for track_id, artist_id, rel_path, tags in rows:
        if require_audio_present:
            apath = audio_root / rel_path
            if not apath.exists():
                # MTG paths sometimes use leading directories like
                # "01/01234.mp3" — accept that as-is.
                reasons["audio_missing"] += 1
                continue
        mood = resolve_clip_moods(tags, rule=conflict_rule)  # type: ignore[arg-type]
        if mood is None:
            mapped = [t for t in tags if t in _MAPPED_SET]
            if not mapped:
                reasons["no_mood_tag"] += 1
            else:
                reasons["conflict"] += 1
            continue
        kept.append(MtgClip(
            track_id=track_id, artist_id=artist_id, rel_path=rel_path,
            tags=tags, mood_class=mood,
        ))
    return kept, reasons


# Lazy-evaluated set of tags that have a mood entry, used to distinguish
# "no mood tag" from "conflict" in the dropped-reason histogram.
from groovebot.style.mood_mapping import TAG_TO_MOOD as _TAG_TO_MOOD
_MAPPED_SET = frozenset(_TAG_TO_MOOD)


def write_manifest(clips: list[MtgClip], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("path", "mtg_track_id", "artist_id", "mood_class",
                    "raw_tags"))
        for c in clips:
            w.writerow((c.rel_path, c.track_id, c.artist_id, c.mood_class,
                        " ".join(c.tags)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio-root", required=True,
                    help="dir containing MTG audio (the --output-dir you "
                         "passed to MTG's download.py)")
    ap.add_argument("--tsv", required=True,
                    help="autotagging_moodtheme.tsv from the MTG repo")
    ap.add_argument("--out-csv", required=True,
                    help="where to write the (path, mood_class, artist_id) "
                         "manifest")
    ap.add_argument("--conflict-rule",
                    choices=("drop_on_disagreement", "first_match"),
                    default="drop_on_disagreement")
    ap.add_argument("--ignore-missing", action="store_true",
                    help="keep TSV rows even if their audio is not on disk; "
                         "useful for a manifest-only dry run")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    audio_root = Path(args.audio_root)
    tsv_path = Path(args.tsv)
    out_csv = Path(args.out_csv)

    # Sanity-check the mapping is in sync with the upstream tag list.
    uncovered, unknown = coverage_check()
    if uncovered:
        print(f"WARNING: {len(uncovered)} MTG tags are neither mapped nor "
              f"dropped: {sorted(uncovered)}", file=sys.stderr)
    if unknown:
        print(f"WARNING: {len(unknown)} mapped/dropped tags are not in the "
              f"upstream MTG list (typo?): {sorted(unknown)}", file=sys.stderr)

    kept, reasons = build_manifest(
        audio_root, tsv_path,
        conflict_rule=args.conflict_rule,
        require_audio_present=not args.ignore_missing,
    )
    write_manifest(kept, out_csv)

    print(f"manifest: {len(kept)} clips kept -> {out_csv}")
    print(f"dropped reasons: {reasons}")
    print(f"upstream tags: {len(MTG_MOODTHEME_TAGS)}, "
          f"mapped: {len(_TAG_TO_MOOD)}, conflict_rule: {args.conflict_rule}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
