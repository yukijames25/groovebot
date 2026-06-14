"""tools/gtzan_split.py — train / val / test split for the full GTZAN audio.

Two modes:

  * **naive** — per-genre stratified random split with a fixed seed.
    Documents the *optimistic* baseline; useful only as the lower bound on
    leakage. Same-artist tracks land in train and test together (GTZAN's
    famous bias), so the resulting accuracy is inflated.

  * **fault** — fault-filtered + artist-aware split published by
    jongpillee/music_dataset_split (Kereliuk 2015, following Sturm 2013):
    `train_filtered.txt`, `valid_filtered.txt`, `test_filtered.txt`. 70
    tracks are dropped (repetitions, mislabels, distortions). No artist
    appears across the train/val/test boundary.

Both modes additionally drop files that fail an `sf.info` probe
(canonical example: `jazz.00054.wav` is corrupted in the public Marsyas
upload and the Kaggle mirror) and log the skipped names.

The functions return plain Python lists of `GTZANClip` dataclasses, so
`experiments/train_style.py` can hand them to any `Dataset` constructor
that expects `(path, genre)` pairs.

References:
  - Sturm 2013: "The GTZAN dataset: Its contents, its faults, their
    effects on evaluation, and its future use", arXiv:1306.1461.
  - Kereliuk et al. 2015: "Deep Learning and Music Adversaries",
    IEEE TMM.
  - jongpillee/music_dataset_split, master branch
    (https://github.com/jongpillee/music_dataset_split, GTZAN_split/).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import random

import soundfile as sf


# Genre order matches groovebot.style.model.GENRES; not imported to keep
# this tool free of the style package's torch dependency.
_GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)


@dataclass
class GTZANClip:
    """One GTZAN track on disk + its genre label."""
    path: Path
    genre: str
    rel: str   # e.g. "blues/blues.00000.wav" — matches jongpillee paths


@dataclass
class SplitReport:
    """What ended up in train / val / test, plus the dropped manifest."""
    train: list[GTZANClip] = field(default_factory=list)
    val: list[GTZANClip] = field(default_factory=list)
    test: list[GTZANClip] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (rel, reason)
    mode: str = ""
    sources: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "skipped": len(self.skipped),
        }


def discover_full_gtzan(root: Path) -> tuple[list[GTZANClip], list[tuple[str, str]]]:
    """Walk `root/<genre>/<genre>.NNNNN.wav` and probe each file with
    `sf.info`. Files that fail the probe are dropped and returned in the
    `skipped` list with a reason. Genre folder names that aren't in
    `_GENRES` are silently ignored (some Kaggle mirrors ship extra
    metadata folders alongside the audio)."""
    clips: list[GTZANClip] = []
    skipped: list[tuple[str, str]] = []
    for genre in _GENRES:
        gdir = root / genre
        if not gdir.is_dir():
            continue
        for wav in sorted(gdir.glob("*.wav")):
            rel = f"{genre}/{wav.name}"
            try:
                info = sf.info(str(wav))
            except Exception as e:
                skipped.append((rel, f"unreadable:{type(e).__name__}"))
                continue
            # Sanity: drop anything shorter than 5 s (the style window
            # is 5-10 s; a sub-5 s clip cannot center-crop).
            min_frames = max(1, int(info.samplerate * 5))
            if info.frames < min_frames:
                skipped.append((rel, f"too_short:{info.frames}f"))
                continue
            clips.append(GTZANClip(path=wav, genre=genre, rel=rel))
    return clips, skipped


def naive_stratified_split(
    clips: list[GTZANClip],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> SplitReport:
    """Per-genre random split. Same seed → same partition.

    Same artist will appear across the boundary; this is the leakage
    baseline. Use only alongside `fault_filtered_split` for comparison."""
    rng = random.Random(seed)
    by_genre: dict[str, list[GTZANClip]] = {}
    for c in clips:
        by_genre.setdefault(c.genre, []).append(c)
    report = SplitReport(
        mode="naive",
        sources={"description": "per-genre random stratified split, fixed seed"},
    )
    for genre, group in by_genre.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if (n - n_test) > 1 else 0
        report.test.extend(group[:n_test])
        report.val.extend(group[n_test: n_test + n_val])
        report.train.extend(group[n_test + n_val:])
    return report


_DEFAULT_SOURCE = {
    "split_repo": "https://github.com/jongpillee/music_dataset_split",
    "split_path": "GTZAN_split/{train,valid,test}_filtered.txt",
    "citation": (
        "Kereliuk et al. 2015 (Deep Learning and Music Adversaries) "
        "following Sturm 2013 fault analysis (arXiv:1306.1461)"
    ),
    "caveat": (
        "GTZAN artist labels are only partially known; "
        "artist-non-overlap is best-effort per the source repo."
    ),
}


def fault_filtered_split(
    clips: list[GTZANClip],
    splits_dir: Path,
) -> SplitReport:
    """Read the jongpillee filtered split text files and partition
    `clips` accordingly.

    Files referenced by the split lists but absent from `clips` (e.g.
    `jazz.00054.wav` after the broken-file probe) are silently dropped
    on the assumption they belong to `skipped` already. Files present in
    `clips` but absent from all three split lists are also dropped (the
    fault filter removed 70 tracks).
    """
    train_set = _read_split(splits_dir / "train_filtered.txt")
    val_set = _read_split(splits_dir / "valid_filtered.txt")
    test_set = _read_split(splits_dir / "test_filtered.txt")
    report = SplitReport(
        mode="fault",
        sources=dict(_DEFAULT_SOURCE),
    )
    seen: set[str] = set()
    for c in clips:
        if c.rel in train_set:
            report.train.append(c)
            seen.add(c.rel)
        elif c.rel in val_set:
            report.val.append(c)
            seen.add(c.rel)
        elif c.rel in test_set:
            report.test.append(c)
            seen.add(c.rel)
    listed_total = len(train_set | val_set | test_set)
    matched = len(seen)
    report.sources["listed_in_repo"] = str(listed_total)
    report.sources["matched_in_clips"] = str(matched)
    return report


def _read_split(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"split file not found: {path}. Download from {_DEFAULT_SOURCE['split_repo']} "
            f"({_DEFAULT_SOURCE['split_path']})"
        )
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.add(s)
    return out


def build_split(
    gtzan_root: Path,
    mode: str,
    *,
    splits_dir: Path | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> SplitReport:
    """High-level entry: discover + probe + split, log skipped files.

    Mode is `"naive"` or `"fault"`. For `"fault"`, `splits_dir` must
    contain the three jongpillee filter files.
    """
    clips, skipped = discover_full_gtzan(gtzan_root)
    if mode == "naive":
        report = naive_stratified_split(
            clips, val_frac=val_frac, test_frac=test_frac, seed=seed,
        )
    elif mode == "fault":
        if splits_dir is None:
            raise ValueError("mode='fault' requires splits_dir")
        report = fault_filtered_split(clips, splits_dir)
    else:
        raise ValueError(f"unknown split mode: {mode!r}")
    report.skipped.extend(skipped)
    return report


__all__ = [
    "GTZANClip",
    "SplitReport",
    "build_split",
    "discover_full_gtzan",
    "fault_filtered_split",
    "naive_stratified_split",
]
