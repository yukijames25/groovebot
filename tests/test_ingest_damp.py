"""tools.ingest_damp — arrangement discovery / rendition id parsing."""
from __future__ import annotations
from pathlib import Path

from tools.ingest_damp import (
    DampArrangement,
    DampRendition,
    discover_arrangements,
)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0")
    return p


def test_discover_arrangements_finds_complete_dirs(tmp_path):
    arr_dir = tmp_path / "amazing_grace"
    _touch(arr_dir / "backing.wav")
    _touch(arr_dir / "vocal_alice.wav")
    _touch(arr_dir / "vocal_bob.wav")

    arrangements = discover_arrangements(tmp_path)
    assert len(arrangements) == 1
    a = arrangements[0]
    assert isinstance(a, DampArrangement)
    assert a.arrangement_id == "amazing_grace"
    assert a.backing_wav.name == "backing.wav"
    assert {r.rendition_id for r in a.renditions} == {"alice", "bob"}


def test_discover_arrangements_skips_dirs_missing_backing(tmp_path):
    bad = tmp_path / "missing_backing"
    _touch(bad / "vocal_alice.wav")
    assert discover_arrangements(tmp_path) == []


def test_discover_arrangements_skips_dirs_without_vocals(tmp_path):
    bad = tmp_path / "no_vocals"
    _touch(bad / "backing.wav")
    assert discover_arrangements(tmp_path) == []


def test_discover_arrangements_empty_root_returns_empty(tmp_path):
    assert discover_arrangements(tmp_path / "missing") == []
    assert discover_arrangements(tmp_path) == []


def test_rendition_id_strips_vocal_prefix(tmp_path):
    arr_dir = tmp_path / "song"
    _touch(arr_dir / "backing.wav")
    _touch(arr_dir / "vocal_singer42.wav")
    a = discover_arrangements(tmp_path)[0]
    assert a.renditions[0].rendition_id == "singer42"


def test_discover_arrangements_custom_globs(tmp_path):
    arr_dir = tmp_path / "song"
    _touch(arr_dir / "bg.wav")
    _touch(arr_dir / "voice_a.wav")
    _touch(arr_dir / "voice_b.wav")
    arr = discover_arrangements(
        tmp_path, backing_name="bg.wav", vocal_glob="voice_*.wav",
    )
    assert len(arr) == 1
    assert {r.rendition_id for r in arr[0].renditions} == {
        "voice_a", "voice_b",     # `_rendition_id_from_path` only strips
                                  # `vocal_`, so custom-glob ids keep the
                                  # filename stem as-is.
    }


def test_rendition_dataclasses_are_hashable():
    """Both dataclasses are frozen so they can sit in sets / dict keys."""
    r1 = DampRendition(rendition_id="alice", vocal_wav=Path("a.wav"))
    r2 = DampRendition(rendition_id="alice", vocal_wav=Path("a.wav"))
    assert hash(r1) == hash(r2)
    assert r1 == r2
