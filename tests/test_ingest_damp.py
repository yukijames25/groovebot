"""tools.ingest_damp — arrangement discovery / extraction."""
from __future__ import annotations
import io
import tarfile
from pathlib import Path

from tools.ingest_damp import (
    DampArrangement,
    DampRendition,
    discover_arrangements,
    extract_damp_s_ag,
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
    assert a.backing_wav is not None
    assert a.backing_wav.name == "backing.wav"
    assert a.reference_midi is None
    assert {r.rendition_id for r in a.renditions} == {"alice", "bob"}


def test_discover_arrangements_finds_midi_only_dirs(tmp_path):
    arr = tmp_path / "song_only_midi"
    _touch(arr / "reference.midi")
    _touch(arr / "vocal_a.wav")
    arrangements = discover_arrangements(tmp_path)
    assert len(arrangements) == 1
    assert arrangements[0].backing_wav is None
    assert arrangements[0].reference_midi is not None
    assert arrangements[0].reference_midi.name == "reference.midi"


def test_discover_arrangements_finds_dirs_with_both_refs(tmp_path):
    arr = tmp_path / "song_both"
    _touch(arr / "backing.wav")
    _touch(arr / "reference.midi")
    _touch(arr / "vocal_a.wav")
    arrangements = discover_arrangements(tmp_path)
    assert len(arrangements) == 1
    assert arrangements[0].backing_wav is not None
    assert arrangements[0].reference_midi is not None


def test_discover_arrangements_skips_dirs_with_no_reference(tmp_path):
    """No backing AND no MIDI -> skipped, even if vocals exist."""
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


# --------------------------------------------------------------------------- #
# extract_damp_s_ag — synthetic tarball, no real DAMP needed
# --------------------------------------------------------------------------- #
def _build_synth_damp_tarball(path: Path,
                              rendition_ids,
                              tsv_rows,
                              midi_bytes: bytes = b"MThd fake midi",
                              ) -> Path:
    """Make a tar.gz that mimics DAMP-S-AG's layout for extraction tests."""
    tsv_lines = [
        "performance_id\taccount_id\tcountry\theadphones",
    ]
    for pid, (acct, country, headphones) in tsv_rows.items():
        tsv_lines.append(f"{pid}\t{acct}\t{country}\t{headphones}")
    tsv_bytes = ("\n".join(tsv_lines) + "\n").encode("utf-8")

    with tarfile.open(str(path), "w:gz") as tar:
        def _add_bytes(name, data):
            buf = io.BytesIO(data)
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, buf)
        _add_bytes("amazing_grace.tsv", tsv_bytes)
        _add_bytes("amazing_grace.midi", midi_bytes)
        # Place renditions in the SAME order they appear in tsv_rows / list.
        for pid in rendition_ids:
            _add_bytes(f"amazing_grace/{pid}.m4a", f"FAKEAUDIO_{pid}".encode())
    return path


def test_extract_damp_s_ag_extracts_subset(tmp_path):
    rendition_ids = ["111_222", "333_444", "555_666"]
    tsv_rows = {
        "111_222": ("acct_a", "US", "1"),
        "333_444": ("acct_b", "GB", "0"),
        "555_666": ("acct_c", "US", "1"),
    }
    tarball = _build_synth_damp_tarball(
        tmp_path / "ag.tar.gz", rendition_ids, tsv_rows,
    )
    out_arr, ids = extract_damp_s_ag(
        tarball, tmp_path / "out",
        arrangement_id="amazing_grace",
        max_n=2,
    )
    assert ids == ["111_222", "333_444"]   # archive order, first 2
    assert (out_arr / "reference.midi").exists()
    assert (out_arr / "vocal_111_222.m4a").exists()
    assert (out_arr / "vocal_333_444.m4a").exists()
    assert not (out_arr / "vocal_555_666.m4a").exists()


def test_extract_damp_s_ag_filters_by_headphones(tmp_path):
    rendition_ids = ["111_222", "333_444", "555_666"]
    tsv_rows = {
        "111_222": ("acct_a", "US", "1"),
        "333_444": ("acct_b", "GB", "0"),
        "555_666": ("acct_c", "US", "1"),
    }
    tarball = _build_synth_damp_tarball(
        tmp_path / "ag.tar.gz", rendition_ids, tsv_rows,
    )
    _out, ids = extract_damp_s_ag(
        tarball, tmp_path / "out",
        max_n=None, headphones_only=True,
    )
    assert ids == ["111_222", "555_666"]


def test_extract_damp_s_ag_filters_by_country(tmp_path):
    rendition_ids = ["111_222", "333_444", "555_666"]
    tsv_rows = {
        "111_222": ("acct_a", "US", "1"),
        "333_444": ("acct_b", "GB", "0"),
        "555_666": ("acct_c", "US", "1"),
    }
    tarball = _build_synth_damp_tarball(
        tmp_path / "ag.tar.gz", rendition_ids, tsv_rows,
    )
    _out, ids = extract_damp_s_ag(
        tarball, tmp_path / "out",
        max_n=None, country="GB",
    )
    assert ids == ["333_444"]


def test_extract_damp_s_ag_writes_layout_compatible_with_discovery(tmp_path):
    """The output layout must be `discover_arrangements`-readable in MIDI
    mode (vocal_*.m4a, reference.midi, no backing.wav)."""
    rendition_ids = ["111_222"]
    tsv_rows = {"111_222": ("acct_a", "US", "1")}
    tarball = _build_synth_damp_tarball(
        tmp_path / "ag.tar.gz", rendition_ids, tsv_rows,
    )
    extract_damp_s_ag(tarball, tmp_path / "out", max_n=1)
    arrangements = discover_arrangements(
        tmp_path / "out", vocal_glob="vocal_*.m4a",
    )
    assert len(arrangements) == 1
    a = arrangements[0]
    assert a.backing_wav is None
    assert a.reference_midi is not None
    assert a.reference_midi.name == "reference.midi"
    assert [r.rendition_id for r in a.renditions] == ["111_222"]
