"""Tests for groovebot.style.deam (DEAM static-annotation loader)."""
from __future__ import annotations
from pathlib import Path

import pytest

from groovebot.style.deam import (
    DEAM_SAM_HI, DEAM_SAM_LO,
    DeamRecord, read_static_annotations, read_static_annotations_many,
    sam_to_unit, song_disjoint_split, unit_to_sam,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def _touch_audio(audio_root: Path, song_ids: list[int], use_memd: bool = False) -> None:
    base = audio_root / "MEMD_audio" if use_memd else audio_root
    base.mkdir(parents=True, exist_ok=True)
    for sid in song_ids:
        (base / f"{sid}.mp3").write_bytes(b"")


def test_read_static_annotations_flat_layout(tmp_path: Path):
    csv_path = tmp_path / "static.csv"
    audio_root = tmp_path / "audio"
    _write_csv(csv_path, [
        {"song_id": 1, "valence_mean": 5.5, "arousal_mean": 6.2},
        {"song_id": 2, "valence_mean": 3.1, "arousal_mean": 2.0},
    ])
    _touch_audio(audio_root, [1, 2])

    records = read_static_annotations(csv_path, audio_root)
    assert len(records) == 2
    by_id = {r.song_id: r for r in records}
    assert by_id[1].arousal == pytest.approx(6.2)
    assert by_id[1].valence == pytest.approx(5.5)
    assert by_id[1].audio_path.name == "1.mp3"


def test_read_static_annotations_memd_layout(tmp_path: Path):
    csv_path = tmp_path / "static.csv"
    audio_root = tmp_path / "audio"
    _write_csv(csv_path, [
        {"song_id": 42, "valence_mean": 4.0, "arousal_mean": 7.0},
    ])
    _touch_audio(audio_root, [42], use_memd=True)
    records = read_static_annotations(csv_path, audio_root)
    assert len(records) == 1
    assert records[0].audio_path.parent.name == "MEMD_audio"


def test_read_static_annotations_drops_missing_audio(tmp_path: Path):
    csv_path = tmp_path / "static.csv"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_csv(csv_path, [
        {"song_id": 1, "valence_mean": 5.0, "arousal_mean": 5.0},
        {"song_id": 2, "valence_mean": 5.0, "arousal_mean": 5.0},
    ])
    _touch_audio(audio_root, [1])
    records = read_static_annotations(csv_path, audio_root)
    assert {r.song_id for r in records} == {1}


def test_read_static_annotations_keep_missing_audio(tmp_path: Path):
    csv_path = tmp_path / "static.csv"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_csv(csv_path, [
        {"song_id": 9, "valence_mean": 5.0, "arousal_mean": 5.0},
    ])
    records = read_static_annotations(
        csv_path, audio_root, require_audio_present=False,
    )
    assert len(records) == 1
    assert records[0].song_id == 9


def test_read_static_annotations_normalises_leading_space(tmp_path: Path):
    """Upstream DEAM csv ships with `' valence_mean'` (leading space)
    in some releases — the loader must normalise."""
    csv_path = tmp_path / "static.csv"
    csv_path.write_text(
        "song_id, valence_mean , arousal_mean\n1,5.0,5.0\n",
        encoding="utf-8",
    )
    audio_root = tmp_path / "audio"
    _touch_audio(audio_root, [1])
    records = read_static_annotations(csv_path, audio_root)
    assert len(records) == 1


def test_read_static_annotations_bad_header_raises(tmp_path: Path):
    csv_path = tmp_path / "static.csv"
    csv_path.write_text("song_id,other\n1,2\n", encoding="utf-8")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    with pytest.raises(ValueError, match="unexpected DEAM header"):
        read_static_annotations(csv_path, audio_root)


def test_read_static_annotations_many_dedup(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    audio_root = tmp_path / "audio"
    _write_csv(a, [{"song_id": 1, "valence_mean": 5.0, "arousal_mean": 5.0}])
    _write_csv(b, [
        {"song_id": 1, "valence_mean": 6.0, "arousal_mean": 6.0},  # overrides
        {"song_id": 2, "valence_mean": 7.0, "arousal_mean": 7.0},
    ])
    _touch_audio(audio_root, [1, 2])
    records = read_static_annotations_many([a, b], audio_root)
    by_id = {r.song_id: r for r in records}
    assert set(by_id) == {1, 2}
    # last file wins on song 1
    assert by_id[1].arousal == pytest.approx(6.0)


def test_song_disjoint_split_no_id_crosses(tmp_path: Path):
    records = [
        DeamRecord(audio_path=Path("dummy"), song_id=i, arousal=5.0, valence=5.0)
        for i in range(20)
    ]
    train, val, test = song_disjoint_split(records, val_frac=0.2, test_frac=0.2, seed=42)
    train_ids = {r.song_id for r in train}
    val_ids = {r.song_id for r in val}
    test_ids = {r.song_id for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train) + len(val) + len(test) == 20


def test_song_disjoint_split_deterministic_with_seed():
    records = [
        DeamRecord(audio_path=Path("d"), song_id=i, arousal=5.0, valence=5.0)
        for i in range(30)
    ]
    a = song_disjoint_split(records, seed=7)
    b = song_disjoint_split(records, seed=7)
    assert [r.song_id for r in a[2]] == [r.song_id for r in b[2]]


def test_sam_to_unit_endpoints():
    assert sam_to_unit(DEAM_SAM_LO) == pytest.approx(0.0)
    assert sam_to_unit(DEAM_SAM_HI) == pytest.approx(1.0)
    assert sam_to_unit(5.0) == pytest.approx(0.5)


def test_sam_to_unit_clamps_out_of_range():
    assert sam_to_unit(-3.0) == 0.0
    assert sam_to_unit(20.0) == 1.0


def test_unit_to_sam_roundtrip():
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert sam_to_unit(unit_to_sam(v)) == pytest.approx(v)
