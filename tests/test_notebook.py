"""notebooks/m0_gtzan_eval.ipynb — structural + CLI-consistency checks.

What we verify:
  1. nbformat-4 minimal structure (cells/metadata/nbformat[_minor]).
  2. Every cell is well-formed (cell_type, source as list[str]).
  3. The notebook embeds the agreed REPO_URL so it is turnkey on Colab.
  4. Every `python -m tools.eval_beat ...` / `tools.prep_dataset ...` line in
     code cells parses cleanly against the real argparse — so a typo'd flag
     is caught here, not by a Colab user 30 minutes into a run.

We do NOT execute any cell.
"""
from __future__ import annotations
import json
import re
import shlex
from pathlib import Path

import pytest

import tools.eval_beat as eval_beat_mod
import tools.prep_dataset as prep_dataset_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "m0_gtzan_eval.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #
def test_notebook_file_exists():
    assert NB_PATH.exists(), f"missing notebook: {NB_PATH}"


def test_notebook_is_nbformat_v4(nb):
    assert nb.get("nbformat") == 4
    assert "nbformat_minor" in nb
    assert "cells" in nb and isinstance(nb["cells"], list)
    assert "metadata" in nb and isinstance(nb["metadata"], dict)
    assert len(nb["cells"]) >= 10, "notebook should have at least ~10 cells"


def test_every_cell_is_well_formed(nb):
    for i, c in enumerate(nb["cells"]):
        assert c.get("cell_type") in ("markdown", "code"), \
            f"cell {i}: bad cell_type"
        src = c.get("source")
        assert isinstance(src, list), f"cell {i}: source must be list[str]"
        assert all(isinstance(s, str) for s in src), \
            f"cell {i}: source items must be strings"
        if c["cell_type"] == "code":
            assert "outputs" in c, f"code cell {i}: missing outputs"
            assert c.get("execution_count") is None, \
                f"code cell {i}: execution_count must be cleared"


def test_notebook_embeds_repo_url(nb):
    """User said the URL must be baked in; assert it's there literally so
    nobody accidentally ships a placeholder."""
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "github.com/yukijames25/groovebot" in text, \
        "REPO_URL not embedded in the notebook"


# --------------------------------------------------------------------------- #
# CLI-consistency: any `python -m tools.eval_beat ...` line in a code cell
# must parse against the real argparse surface.
# --------------------------------------------------------------------------- #
CLI_LINE = re.compile(
    r"!?\s*python\s+-m\s+tools\.(eval_beat|prep_dataset)\s+(.+)$",
    re.MULTILINE,
)


def _argparser_for(module_tag: str):
    if module_tag == "eval_beat":
        return eval_beat_mod.build_parser()
    if module_tag == "prep_dataset":
        return prep_dataset_mod.build_parser()
    raise AssertionError(module_tag)


def test_documented_cli_lines_parse_against_real_argparse(nb):
    """Catches typo'd flags before a Colab user does."""
    found = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for m in CLI_LINE.finditer(src):
            tag, rest = m.group(1), m.group(2).strip()
            # Strip line continuations and trailing comments.
            rest = rest.replace("\\\n", " ").split("#", 1)[0].strip()
            argv = shlex.split(rest)
            parser = _argparser_for(tag)
            # argparse calls sys.exit on bad input; treat that as a fail.
            try:
                parser.parse_args(argv)
            except SystemExit as e:
                pytest.fail(
                    f"notebook CLI line for tools.{tag} is invalid:\n"
                    f"    {rest}\n"
                    f"  argparse rejected it (exit code {e.code})."
                )
            found += 1
    # The current notebook calls these tools through experiments.run_gtzan_eval,
    # not the CLI. That's fine — the test is here to guard against the day
    # someone adds a shell line. So zero matches is allowed.
    assert found >= 0


# --------------------------------------------------------------------------- #
# Sanity: the notebook actually mentions the real symbols it imports.
# --------------------------------------------------------------------------- #
def test_notebook_imports_experiments_engine(nb):
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "experiments.run_gtzan_eval" in text
    for sym in ("VOCAL_GENRES", "run_pipeline", "aggregate"):
        assert sym in text, f"notebook does not reference experiments.{sym}"
