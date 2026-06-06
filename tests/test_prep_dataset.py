"""tools/prep_dataset.py — annotation conversion is locally testable. Demucs
is not (heavy + experiments-only); we just verify it fails with a clear hint
when missing.
"""
from __future__ import annotations
import os
import tempfile

import pytest

from tools.prep_dataset import (
    parse_ballroom,
    parse_isophonics_beats,
    parse_single_column,
    separate_vocal,
    write_beats_file,
)
from tools.eval_beat import load_beat_annotation


# --------------------------------------------------------------------------- #
# Annotation parsers.
# --------------------------------------------------------------------------- #
def test_parse_ballroom_picks_first_column_and_skips_noise():
    src = (
        "# Ballroom-format demo\n"
        "0.500\t1\n"
        "1.000\t2\n"
        "\n"
        "; another comment\n"
        "1.500\t3\n"
        "2.000\t4\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write(src)
        times = parse_ballroom(p)
    assert times == [0.5, 1.0, 1.5, 2.0]


def test_parse_isophonics_uses_first_column():
    """Isophonics beat files: `<time> <bar.beat>` — first whitespace token."""
    src = "0.100 1.1\n0.600 1.2\n1.100 1.3\n1.600 1.4\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write(src)
        times = parse_isophonics_beats(p)
    assert times == [0.1, 0.6, 1.1, 1.6]


def test_parse_single_column_handles_unsorted_input():
    src = "1.0\n0.5\n2.0\n0.0\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "track.beats")
        with open(p, "w") as f:
            f.write(src)
        times = parse_single_column(p)
    assert times == [0.0, 0.5, 1.0, 2.0]


def test_parse_ballroom_raises_on_malformed():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "bad.beats")
        with open(p, "w") as f:
            f.write("nope\n")
        with pytest.raises(ValueError):
            parse_ballroom(p)


# --------------------------------------------------------------------------- #
# Roundtrip: ann -> write_beats_file -> eval_beat.load_beat_annotation.
# This pins the contract between prep_dataset and eval_beat (one source of
# truth for our `--beats` format).
# --------------------------------------------------------------------------- #
def test_writer_format_is_loadable_by_eval_beat():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.beats")
        write_beats_file([0.0, 0.5, 1.0, 1.5, 2.0], out)
        arr = load_beat_annotation(out)
        assert arr.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


# --------------------------------------------------------------------------- #
# Demucs path — local: must raise a RuntimeError with the install hint.
# (If Demucs happens to be installed on CI, the call below still raises
# NotImplementedError as documented in the stub, so the test logic stays one
# branch.)
# --------------------------------------------------------------------------- #
def test_separate_vocal_without_demucs_raises_with_install_hint():
    try:
        import demucs  # noqa: F401
        pytest.skip("demucs is installed; this case verifies the missing-dep path")
    except Exception:
        pass

    with pytest.raises(RuntimeError, match="demucs"):
        separate_vocal("nonexistent.wav", "/tmp/out")
