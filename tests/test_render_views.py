"""Camera-view option for the render_groove / run_showcase dev scripts.

The view is a *rendering-only* knob: it must change which pixels get
drawn and which filename they land in, and nothing else. These tests
pin both halves of that contract:

  * the view table is well formed, and `side` still resolves to MuJoCo's
    `camera=-1` sentinel so the historical output is reproduced exactly;
  * the filename suffixes keep non-default views from overwriting the
    side render;
  * the joint targets produced under two different views are identical.

Rendering itself needs a GL backend, so the pixel-level tests skip when
MuJoCo (or its renderer) is unavailable.
"""
from __future__ import annotations

import pytest


# ------------------------------------------------------------- view table

def test_views_table_has_the_three_documented_poses():
    from experiments.render_groove import VIEWS

    assert set(VIEWS) == {"side", "front", "threequarter"}


def test_default_view_is_side_and_keeps_the_bare_filename():
    from experiments.render_groove import DEFAULT_VIEW, view_suffix

    assert DEFAULT_VIEW == "side"
    # A bare suffix is what stops the default run from being renamed.
    assert view_suffix("side") == ""


@pytest.mark.parametrize("view,suffix", [
    ("side", ""),
    ("front", "_front"),
    ("threequarter", "_3q"),
])
def test_view_suffixes_are_the_documented_ones(view, suffix):
    from experiments.render_groove import view_suffix

    assert view_suffix(view) == suffix


def test_suffixes_are_unique_so_views_cannot_overwrite_each_other():
    from experiments.render_groove import VIEWS, view_suffix

    suffixes = [view_suffix(v) for v in VIEWS]
    assert len(set(suffixes)) == len(suffixes)


def test_only_side_uses_the_mujoco_default_camera():
    # `use_model_default` is the flag that routes a view through
    # camera=-1. Exactly one view may claim it, or the "side output is
    # unchanged" guarantee stops meaning anything.
    from experiments.render_groove import VIEWS

    defaults = [name for name, spec in VIEWS.items() if spec.use_model_default]
    assert defaults == ["side"]


def test_non_default_views_are_distinct_camera_poses():
    from experiments.render_groove import VIEWS

    front, tq = VIEWS["front"], VIEWS["threequarter"]
    assert (front.azimuth, front.elevation) != (tq.azimuth, tq.elevation)
    # Front means "camera on +X looking back down -X" in the URDF frame
    # (Z up, X forward, Y left), i.e. azimuth 180.
    assert front.azimuth == pytest.approx(180.0)
    # Three-quarter sits strictly between side (90) and front (180).
    assert 90.0 < tq.azimuth < 180.0


def test_view_specs_are_frozen():
    # The table is module-level shared state; freezing it keeps a caller
    # from mutating one render's framing into every later render's.
    import dataclasses

    from experiments.render_groove import VIEWS

    spec = VIEWS["front"]
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.azimuth = 0.0


# ------------------------------------------------------- camera construction

def test_side_resolves_to_the_camera_minus_one_sentinel():
    mujoco = pytest.importorskip("mujoco")
    from experiments.render_groove import URDF, _make_camera

    model = mujoco.MjModel.from_xml_path(URDF)
    # None is the signal render_one turns into `camera=-1` — the exact
    # call the script made before --view existed.
    assert _make_camera(model, "side") is None


@pytest.mark.parametrize("view", ["front", "threequarter"])
def test_explicit_views_build_a_camera_with_the_table_pose(view):
    mujoco = pytest.importorskip("mujoco")
    from experiments.render_groove import URDF, VIEWS, _make_camera

    model = mujoco.MjModel.from_xml_path(URDF)
    spec = VIEWS[view]
    cam = _make_camera(model, view)

    assert cam is not None
    assert cam.azimuth == pytest.approx(spec.azimuth)
    assert cam.elevation == pytest.approx(spec.elevation)
    assert cam.lookat[2] == pytest.approx(spec.lookat_z)
    # Distance is scaled off the model-derived default, not hardcoded,
    # so the framing survives edits to the URDF.
    ref = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, ref)
    assert cam.distance == pytest.approx(ref.distance * spec.dist_scale)


def test_unknown_view_is_rejected_by_render_one():
    from experiments.render_groove import render_one
    from groovebot.style.select import GrooveStyle

    style = GrooveStyle(
        move="headbang", intensity=0.5, genre="rock", mood="aggressive",
        mood_probs={"aggressive": 1.0}, genre_probs={"rock": 1.0},
        tempo_bpm=120.0, arousal=0.5, arousal_bucket="mid",
    )
    with pytest.raises(ValueError, match="unknown view"):
        render_one(
            style=style, seconds=0.1, rate=50.0, outdir=None, tag="x",
            width=64, height=48, skip_render=True, use_ctx_arousal=False,
            view="from_above",
        )


# ------------------------------------------- the view must not move a joint

@pytest.mark.parametrize("view", ["side", "front", "threequarter"])
def test_joint_targets_are_identical_across_views(tmp_path, view):
    """The camera must not reach the brain.

    Renders the same style twice (side vs `view`) with the GIF disabled
    and compares the per-tick CSV byte for byte. This is the property
    the showcase claim "attributes / style / JointCommand are unchanged
    by --view" rests on.
    """
    pytest.importorskip("mujoco")
    from experiments.render_groove import render_one
    from groovebot.style.select import GrooveStyle

    style = GrooveStyle(
        move="clap", intensity=0.8, genre="pop", mood="happy",
        mood_probs={"happy": 1.0}, genre_probs={"pop": 1.0},
        tempo_bpm=117.0, arousal=0.64, arousal_bucket="high",
    )
    kwargs = dict(
        style=style, seconds=0.4, rate=50.0, tag="clip",
        width=64, height=48, skip_render=True, use_ctx_arousal=False,
    )
    base = render_one(outdir=tmp_path / "a", view="side", **kwargs)
    other = render_one(outdir=tmp_path / "b", view=view, **kwargs)

    assert base.csv_path.read_bytes() == other.csv_path.read_bytes()
    assert base.final_pose == other.final_pose


def test_render_result_records_the_view_and_suffixes_the_paths(tmp_path):
    pytest.importorskip("mujoco")
    from experiments.render_groove import render_one
    from groovebot.style.select import GrooveStyle

    style = GrooveStyle(
        move="bob_nod", intensity=0.5, genre="rock", mood="happy",
        mood_probs={"happy": 1.0}, genre_probs={"rock": 1.0},
        tempo_bpm=120.0, arousal=0.5, arousal_bucket="mid",
    )
    kwargs = dict(
        style=style, seconds=0.2, rate=50.0, outdir=tmp_path, tag="clip",
        width=64, height=48, skip_render=True, use_ctx_arousal=False,
    )
    side = render_one(view="side", **kwargs)
    front = render_one(view="front", **kwargs)

    assert side.view == "side"
    assert front.view == "front"
    assert side.csv_path.name == "clip.csv"
    assert front.csv_path.name == "clip_front.csv"
    # The whole point: the front run left the side artefact in place.
    assert side.csv_path.exists() and front.csv_path.exists()
    assert side.csv_path != front.csv_path


# --------------------------------------------------- views really differ

def test_front_and_side_render_different_pixels():
    """Guards against the camera being silently ignored."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from experiments.render_groove import URDF, _make_camera

    model = mujoco.MjModel.from_xml_path(URDF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, height=64, width=64)
    except Exception:                                     # pragma: no cover
        pytest.skip("no GL backend for MuJoCo Renderer")

    frames = {}
    for view in ("side", "front", "threequarter"):
        cam = _make_camera(model, view)
        renderer.update_scene(data, camera=-1 if cam is None else cam)
        frames[view] = renderer.render().copy()

    assert not np.array_equal(frames["side"], frames["front"])
    assert not np.array_equal(frames["front"], frames["threequarter"])


# ------------------------------------------------------ run_showcase wiring

def test_run_showcase_exposes_view_and_force_summary_flags():
    rs = pytest.importorskip("experiments.run_showcase")
    import inspect

    src = inspect.getsource(rs.main)
    assert '"--view"' in src
    assert '"--force-summary"' in src


def test_run_showcase_run_one_takes_a_view():
    import inspect

    rs = pytest.importorskip("experiments.run_showcase")
    params = inspect.signature(rs.run_one).parameters
    assert "view" in params
    assert params["view"].default == "side"


def test_summary_writer_refuses_to_clobber_by_default(tmp_path):
    """showcase_summary.md holds hand-written analysis below the table."""
    rs = pytest.importorskip("experiments.run_showcase")

    rows = [{
        "clip": "c", "lang": "en", "genre": "rock (0.90)", "arousal": "0.60",
        "valence": "0.50", "mood": "happy (0.40)", "style_move": "headbang",
        "intensity": "0.90", "dominant_move": "headbang",
        "accent_phase": "on-beat", "tempo_bpm": "120", "arousal_bucket": "high",
    }]
    path = tmp_path / "showcase_summary.md"
    hand_written = "# mine\n\nhand-written analysis that must survive\n"
    path.write_text(hand_written, encoding="utf-8")

    assert rs._write_summary_md(tmp_path, rows) is None
    assert path.read_text(encoding="utf-8") == hand_written

    # ...but an explicit force still regenerates it.
    assert rs._write_summary_md(tmp_path, rows, force=True) == path
    assert "hand-written analysis" not in path.read_text(encoding="utf-8")


def test_summary_writer_creates_the_file_when_absent(tmp_path):
    rs = pytest.importorskip("experiments.run_showcase")

    rows = [{
        "clip": "c", "lang": "ja", "genre": "pop (0.40)", "arousal": "0.42",
        "valence": "0.42", "mood": "sad (0.33)", "style_move": "clap",
        "intensity": "0.57", "dominant_move": "clap",
        "accent_phase": "backbeat (2 & 4)", "tempo_bpm": "78",
        "arousal_bucket": "mid",
    }]
    out = rs._write_summary_md(tmp_path, rows)
    assert out == tmp_path / "showcase_summary.md"
    assert "clap" in out.read_text(encoding="utf-8")
