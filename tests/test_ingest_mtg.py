"""Tests for tools/ingest_mtg_moodtheme (TSV parsing + manifest)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.ingest_mtg_moodtheme import (
    _strip_tag_prefix,
    build_manifest,
    parse_tsv,
    write_manifest,
)


def _write_audio(path: Path, dur_sec: float = 1.0, sr: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path),
             np.zeros(int(sr * dur_sec), dtype=np.float32), sr)


def test_strip_tag_prefix():
    assert _strip_tag_prefix("mood/theme---epic") == "epic"
    assert _strip_tag_prefix("happy") == "happy"
    assert _strip_tag_prefix("genre---rock") == "rock"


def test_parse_tsv(tmp_path):
    tsv = tmp_path / "tags.tsv"
    tsv.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "1\tA\tX\t01/0001.mp3\t30.0\tmood/theme---happy mood/theme---fun\n"
        "2\tB\tY\t02/0002.mp3\t45.0\tmood/theme---sad\n"
        "3\tC\tZ\t03/0003.mp3\t12.0\t\n",
        encoding="utf-8",
    )
    rows = parse_tsv(tsv)
    assert len(rows) == 3
    assert rows[0] == ("1", "A", "01/0001.mp3", ["happy", "fun"])
    assert rows[2][3] == []


def test_build_manifest_kept_and_dropped(tmp_path):
    audio_root = tmp_path / "audio"
    _write_audio(audio_root / "01" / "0001.mp3")     # happy clip
    _write_audio(audio_root / "02" / "0002.mp3")     # sad clip
    _write_audio(audio_root / "03" / "0003.mp3")     # conflict clip
    # 0004 has no audio file -> audio_missing
    tsv = tmp_path / "tags.tsv"
    tsv.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "1\tA\tX\t01/0001.mp3\t30.0\tmood/theme---happy\n"
        "2\tB\tY\t02/0002.mp3\t30.0\tmood/theme---sad\n"
        "3\tC\tZ\t03/0003.mp3\t30.0\tmood/theme---happy mood/theme---sad\n"
        "4\tD\tW\t04/0004.mp3\t30.0\tmood/theme---fun\n",
        encoding="utf-8",
    )
    kept, reasons = build_manifest(audio_root, tsv)
    assert reasons["audio_missing"] == 1
    assert reasons["conflict"] == 1
    assert {c.mood_class for c in kept} == {"happy", "sad"}
    assert len(kept) == 2


def test_build_manifest_first_match_keeps_conflicts(tmp_path):
    audio_root = tmp_path / "audio"
    _write_audio(audio_root / "01" / "0001.mp3")
    tsv = tmp_path / "tags.tsv"
    tsv.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "1\tA\tX\t01/0001.mp3\t30.0\tmood/theme---happy mood/theme---sad\n",
        encoding="utf-8",
    )
    kept, _ = build_manifest(audio_root, tsv, conflict_rule="first_match")
    assert len(kept) == 1
    assert kept[0].mood_class in {"happy", "sad"}


def test_build_manifest_theme_only_drops(tmp_path):
    audio_root = tmp_path / "audio"
    _write_audio(audio_root / "01" / "0001.mp3")
    tsv = tmp_path / "tags.tsv"
    tsv.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "1\tA\tX\t01/0001.mp3\t30.0\tmood/theme---christmas mood/theme---advertising\n",
        encoding="utf-8",
    )
    kept, reasons = build_manifest(audio_root, tsv)
    assert kept == []
    assert reasons["no_mood_tag"] == 1


def test_write_manifest_roundtrip(tmp_path):
    from tools.ingest_mtg_moodtheme import MtgClip
    clips = [
        MtgClip(track_id="1", artist_id="A", rel_path="01/0001.mp3",
                tags=["happy"], mood_class="happy"),
        MtgClip(track_id="2", artist_id="B", rel_path="02/0002.mp3",
                tags=["sad", "melancholic"], mood_class="sad"),
    ]
    out = tmp_path / "manifest.csv"
    write_manifest(clips, out)
    text = out.read_text(encoding="utf-8")
    assert "path,mtg_track_id,artist_id,mood_class,raw_tags" in text
    assert "01/0001.mp3" in text and "happy" in text
    assert "02/0002.mp3" in text and "melancholic" in text
