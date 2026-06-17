"""Tiny smoke test for the render_groove dev script.

We don't exercise the MuJoCo renderer from pytest — that wants a GL
backend and dumps files. Instead: confirm the module imports, the
forced-style builder produces a valid `GrooveStyle`, and the moves it
iterates over match the primitive library.
"""
from __future__ import annotations
import argparse

import pytest


def test_render_groove_imports_and_exposes_expected_api():
    rg = pytest.importorskip("experiments.render_groove")
    assert callable(getattr(rg, "render_one", None))
    assert callable(getattr(rg, "_forced_style", None))


def test_forced_style_builds_a_legal_grovestyle():
    from experiments.render_groove import _forced_style
    from groovebot.style.select import GrooveStyle
    from groovebot.style.table import MOVES

    style = _forced_style("headbang", bpm=140.0, intensity=0.7)
    assert isinstance(style, GrooveStyle)
    assert style.move == "headbang"
    assert style.move in MOVES
    assert style.tempo_bpm == pytest.approx(140.0)
    assert 0.0 <= style.intensity <= 1.0


def test_moves_to_render_all_moves_covers_primitives():
    from experiments.render_groove import _moves_to_render
    from groovebot.groove_style import MOVE_PRIMITIVES

    args = argparse.Namespace(
        all_moves=True, audio=None, style="bob_nod",
        bpm=120.0, intensity=0.85,
    )
    seen = [tag for tag, _ in _moves_to_render(args)]
    assert set(seen) == set(MOVE_PRIMITIVES.keys())


def test_render_groove_exposes_narrate_module():
    # Light coupling check: the v1.1 narration layer is imported by the
    # render script (so --narrate / --verbose work without further wiring).
    from experiments import render_groove
    from groovebot.style import narrate as narrate_mod

    assert render_groove.narrate is narrate_mod.narrate
