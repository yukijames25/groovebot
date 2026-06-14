"""Tests for tools.gtzan_split (discovery + naive / fault splits).

Uses a tiny synthetic GTZAN-shaped tree (3 genres × 4 files) and a
hand-written `*_filtered.txt` set so tests do NOT need the real 1.2 GB
dataset on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.gtzan_split import (
    GTZANClip,
    build_split,
    discover_full_gtzan,
    fault_filtered_split,
    naive_stratified_split,
)


SR = 22050
GENRES_IN_FIXTURE = ("blues", "classical", "rock")


def _make_gtzan_tree(root: Path, broken: set[str] = frozenset()) -> None:
    """3 genres × 4 files × 6 s synthetic audio. Files in `broken` are
    written as truncated headers so sf.info raises."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for genre in GENRES_IN_FIXTURE:
        gdir = root / genre
        gdir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            rel = f"{genre}.{i:05d}.wav"
            path = gdir / rel
            if rel in broken:
                path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEbroken")
                continue
            sig = (0.1 * rng.standard_normal(SR * 6)).astype(np.float32)
            sf.write(str(path), sig, SR)


def test_discover_finds_all_files(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    clips, skipped = discover_full_gtzan(root)
    assert len(clips) == 3 * 4
    assert not skipped
    for c in clips:
        assert c.genre in GENRES_IN_FIXTURE
        assert c.rel.startswith(c.genre + "/")


def test_discover_skips_unreadable(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root, broken={"blues.00002.wav"})
    clips, skipped = discover_full_gtzan(root)
    assert len(clips) == 3 * 4 - 1
    assert skipped == [("blues/blues.00002.wav",
                        "unreadable:LibsndfileError")] or (
        len(skipped) == 1 and skipped[0][0] == "blues/blues.00002.wav"
    )


def test_naive_split_is_stratified_and_disjoint(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    clips, _ = discover_full_gtzan(root)
    report = naive_stratified_split(clips, val_frac=0.25, test_frac=0.25, seed=0)
    train, val, test = report.train, report.val, report.test
    # disjoint
    s_train = {c.rel for c in train}
    s_val = {c.rel for c in val}
    s_test = {c.rel for c in test}
    assert not (s_train & s_val)
    assert not (s_train & s_test)
    assert not (s_val & s_test)
    # every genre represented in train
    train_genres = {c.genre for c in train}
    assert train_genres == set(GENRES_IN_FIXTURE)


def test_naive_split_is_deterministic_with_seed(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    clips, _ = discover_full_gtzan(root)
    r1 = naive_stratified_split(clips, seed=42)
    r2 = naive_stratified_split(clips, seed=42)
    assert [c.rel for c in r1.train] == [c.rel for c in r2.train]
    assert [c.rel for c in r1.test] == [c.rel for c in r2.test]


def test_fault_split_partitions_by_text_files(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    clips, _ = discover_full_gtzan(root)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train_filtered.txt").write_text(
        "blues/blues.00000.wav\nblues/blues.00001.wav\n"
        "classical/classical.00000.wav\n"
        "rock/rock.00000.wav\n"
    )
    (splits_dir / "valid_filtered.txt").write_text(
        "blues/blues.00002.wav\nclassical/classical.00001.wav\nrock/rock.00001.wav\n"
    )
    (splits_dir / "test_filtered.txt").write_text(
        "blues/blues.00003.wav\nclassical/classical.00002.wav\nrock/rock.00002.wav\n"
    )

    report = fault_filtered_split(clips, splits_dir)
    assert {c.rel for c in report.train} == {
        "blues/blues.00000.wav", "blues/blues.00001.wav",
        "classical/classical.00000.wav", "rock/rock.00000.wav",
    }
    assert len(report.val) == 3
    assert len(report.test) == 3
    # Files not in any split list (classical.00003, rock.00003) are dropped.
    all_after = (report.train + report.val + report.test)
    assert len(all_after) == 10
    assert "github.com/jongpillee/music_dataset_split" in (
        report.sources["split_repo"]
    )


def test_fault_split_missing_dir_raises(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    clips, _ = discover_full_gtzan(root)
    with pytest.raises(FileNotFoundError, match="split file not found"):
        fault_filtered_split(clips, tmp_path / "does-not-exist")


def test_build_split_high_level_naive(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root, broken={"rock.00003.wav"})
    report = build_split(root, mode="naive", seed=0)
    assert report.mode == "naive"
    # broken file becomes a skipped record
    assert any("rock.00003" in rel for rel, _ in report.skipped)


def test_build_split_unknown_mode_raises(tmp_path):
    root = tmp_path / "gtzan"
    _make_gtzan_tree(root)
    with pytest.raises(ValueError, match="unknown split mode"):
        build_split(root, mode="invalid")
