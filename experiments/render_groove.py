"""
experiments/render_groove.py — visualise the JointCommand bridge v1.

Drives the MuJoCo backend with `StyleGrooveGenerator` and writes a short
GIF of the resulting motion. Two modes:

  * `--style MOVE` (default if no audio): force a single move primitive
    from the `table.MOVES` vocabulary. Bypasses the selector — handy
    for inspecting one primitive at a time without needing a clip.
  * `--audio PATH`: run `GrooveStyleSelector` (v3 PANNs path, **trained
    checkpoints required**) on the clip and visualise whatever style
    the table picks. Missing ckpts → loud INVALID stop unless
    `--allow-untrained` is set.
    Defaults:
      - PANNs backbone: `data/raw/Cnn14_mAP=0.431.pth`
      - genre head:     `data/style_v3_fault/style_head.pt`
      - arousal head:   `data/style_v3_arousal/style_head_arousal.pt`
    Mood defaults to `va` (derived from learned arousal+valence via the
    circumplex map); pass `--mood-source head` to use the MTG-trained
    mood head instead.

`--all-moves` renders every primitive in `MOVE_PRIMITIVES` to its own
GIF — the canonical "primitive library at a glance" output.

Outputs (default `data/renders/`):
  - `<tag>.gif` — animated frames at the control rate
  - `<tag>.csv` — per-tick joint targets (always produced; safe even
    when the GL renderer fails)

This script is a developer tool; pytest does not exercise rendering.
The honest limit: MuJoCo renders kinematic playback. Real sim dynamics
and real servo dynamics will look different — see
SYSTEM_SPEC.md §14 v1 bridge "honest limits".

Usage:
    python -m experiments.render_groove --all-moves --seconds 4
    python -m experiments.render_groove --style headbang --bpm 140 --seconds 6
    python -m experiments.render_groove --audio data/raw/clip.wav --seconds 8
"""
from __future__ import annotations
import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from groovebot.backend import JOINT_NAMES, MujocoBackend
from groovebot.groove_style import (
    MOVE_PRIMITIVES,
    StyleGrooveGenerator,
    metronome_from_style,
    neutral_pose,
)
from groovebot.limits import make_clamp
from groovebot.orchestrator import MetronomePerception, Orchestrator
from groovebot.style.narrate import narrate
from groovebot.style.select import GrooveStyle


URDF = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "robot", "groovebot.urdf"
)
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "renders"
)

# Default trained checkpoint locations. The selector path requires *all*
# four files (PANNs backbone + genre head + arousal regression head).
# Missing files trigger a loud INVALID stop unless --allow-untrained is set.
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PANNS_CKPT = os.path.join(_REPO_ROOT, "data", "raw", "Cnn14_mAP=0.431.pth")
DEFAULT_GENRE_HEAD = os.path.join(_REPO_ROOT, "data", "style_v3_fault", "style_head.pt")
DEFAULT_AROUSAL_HEAD = os.path.join(
    _REPO_ROOT, "data", "style_v3_arousal", "style_head_arousal.pt"
)


# ----------------------------------------------------------- core runner

@dataclass
class RenderResult:
    tag: str
    csv_path: Path
    gif_path: Path | None
    final_pose: dict[str, float]


class _RecordingBackend:
    """Wraps a real backend and snapshots commands + frames each tick.

    Frames are RGB ndarrays from MuJoCo's Renderer if available; if the
    GL backend is missing or the render call fails, frames stay empty
    and only the CSV log is produced.
    """

    def __init__(self, inner: MujocoBackend, *, render: bool,
                 width: int, height: int, every: int = 1):
        self.inner = inner
        self.render = render
        self.width = width
        self.height = height
        self.every = max(1, int(every))
        self.commands: list[dict[str, float]] = []
        self.frames: list[np.ndarray] = []
        self._tick = 0
        self._renderer = None

    def load(self, urdf_path: str) -> None:
        self.inner.load(urdf_path)
        if self.render:
            try:
                import mujoco
                self._renderer = mujoco.Renderer(
                    self.inner._m, height=self.height, width=self.width,
                )
            except Exception as exc:                              # pragma: no cover
                print(f"[render_groove] disabling GIF: renderer init failed: {exc}")
                self.render = False
                self._renderer = None

    def set_joint_targets(self, targets):
        self.commands.append(dict(targets))
        self.inner.set_joint_targets(targets)

    def step(self, dt):
        self.inner.step(dt)
        if self.render and self._renderer is not None:
            if self._tick % self.every == 0:
                try:
                    self._renderer.update_scene(self.inner._d, camera=-1)
                    self.frames.append(self._renderer.render().copy())
                except Exception as exc:                          # pragma: no cover
                    print(f"[render_groove] disabling GIF mid-run: {exc}")
                    self.render = False
        self._tick += 1

    def get_joint_states(self):
        return self.inner.get_joint_states()

    def close(self):
        self.inner.close()


def _save_csv(commands: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tick", *JOINT_NAMES])
        for i, cmd in enumerate(commands):
            w.writerow([i, *(f"{cmd.get(n, 0.0):.6f}" for n in JOINT_NAMES)])


def _save_gif(frames: list[np.ndarray], path: Path, *, fps: float) -> bool:
    if not frames:
        return False
    try:
        from PIL import Image
    except Exception as exc:                                       # pragma: no cover
        print(f"[render_groove] cannot save GIF (no Pillow): {exc}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / fps)))
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        str(path),
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return True


def render_one(
    *,
    style: GrooveStyle,
    seconds: float,
    rate: float,
    outdir: Path,
    tag: str,
    width: int,
    height: int,
    skip_render: bool,
    use_ctx_arousal: bool,
    narrate_mode: str = "off",
    narrate_max_beats: int | None = None,
) -> RenderResult:
    backend = _RecordingBackend(
        MujocoBackend(),
        render=not skip_render,
        width=width, height=height,
    )
    backend.load(URDF)

    perception = metronome_from_style(style)
    generator = StyleGrooveGenerator(style=style, use_ctx_arousal=use_ctx_arousal)
    clamp = make_clamp(URDF, joint_names=JOINT_NAMES)

    orch = Orchestrator(
        perception=perception,
        generator=generator,
        backend=backend,
        rate=rate,
        clamp=clamp,
        realtime=False,
    )
    orch.run(seconds)

    csv_path = outdir / f"{tag}.csv"
    gif_path = outdir / f"{tag}.gif"
    _save_csv(backend.commands, csv_path)
    ok = _save_gif(backend.frames, gif_path, fps=rate)
    final = backend.get_joint_states()
    backend.close()

    print(
        f"[render_groove] {tag}: "
        f"{len(backend.commands)} ticks, "
        f"{'GIF=' + str(gif_path) if ok else 'GIF=skipped'}, "
        f"CSV={csv_path}"
    )
    if narrate_mode != "off":
        verbose = narrate_mode == "verbose"
        print(narrate(
            style,
            backend.commands if verbose else None,
            rate=rate,
            seconds=seconds,
            verbose=verbose,
            max_beats=narrate_max_beats,
        ))
    return RenderResult(tag=tag, csv_path=csv_path,
                        gif_path=gif_path if ok else None,
                        final_pose=final)


# ----------------------------------------------------------- style builders

def _forced_style(move: str, *, bpm: float, intensity: float) -> GrooveStyle:
    return GrooveStyle(
        move=move,
        intensity=float(intensity),
        genre="rock",
        mood="aggressive",
        mood_probs={"aggressive": 1.0},
        genre_probs={"rock": 1.0},
        tempo_bpm=float(bpm),
        arousal=float(intensity),
        arousal_bucket="mid",
    )


def _build_selector(
    *,
    panns_ckpt: str,
    genre_head: str,
    arousal_head: str,
    mood_source: str = "va",
    allow_untrained: bool = False,
):
    """Build a `GrooveStyleSelector` for the showcase pipeline.

    Validates that all three trained checkpoints exist (the PANNs CNN14
    backbone, the GTZAN-trained genre/mood classification head, and the
    DEAM-trained arousal/valence regression head). Missing files trigger
    a loud `INVALID: trained checkpoint not found` and a `SystemExit`.

    Pass `allow_untrained=True` for a smoke-test path with random
    weights (selector still emits legal output, but genre / arousal
    numbers are meaningless).
    """
    from groovebot.style.select import GrooveStyleSelector

    missing: list[tuple[str, str]] = []
    for label, path in (
        ("PANNs CNN14 backbone (data/raw/Cnn14_mAP=0.431.pth)", panns_ckpt),
        ("genre TL head (data/style_v3_fault/style_head.pt)", genre_head),
        ("arousal/valence TL head (data/style_v3_arousal/style_head_arousal.pt)",
         arousal_head),
    ):
        if not os.path.isfile(path):
            missing.append((label, path))

    if missing:
        if not allow_untrained:
            print(
                "\n\n==============================================================\n"
                "INVALID: trained checkpoint not found.\n"
                "==============================================================\n"
                "The --audio path is wired to load trained transfer-learning\n"
                "heads + the PANNs CNN14 backbone. The following files are\n"
                "missing:",
                file=os.sys.stderr,
            )
            for label, path in missing:
                print(f"  - {label}\n      expected at: {path}",
                      file=os.sys.stderr)
            print(
                "\n  - Train them with experiments/train_genre_tl.py and\n"
                "    experiments/train_arousal_tl.py (see CLAUDE.md v3 nodes),\n"
                "  - Or pass --allow-untrained to run with random weights\n"
                "    (legal output but genre / arousal are meaningless).\n",
                file=os.sys.stderr,
            )
            raise SystemExit(2)
        print(
            "[render_groove] WARNING: trained ckpts missing — running with "
            "random weights (--allow-untrained).",
            file=os.sys.stderr,
        )
        return GrooveStyleSelector()           # v1/v2 CNN, random weights

    return GrooveStyleSelector.from_panns(
        panns_ckpt,
        head_weights=genre_head,
        regression_head_weights=arousal_head,
        mood_source=mood_source,
    )


def _style_from_audio(
    audio_path: str,
    *,
    panns_ckpt: str,
    genre_head: str,
    arousal_head: str,
    mood_source: str = "va",
    allow_untrained: bool = False,
) -> GrooveStyle:
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    selector = _build_selector(
        panns_ckpt=panns_ckpt,
        genre_head=genre_head,
        arousal_head=arousal_head,
        mood_source=mood_source,
        allow_untrained=allow_untrained,
    )
    style = selector.select(np.asarray(y, dtype=np.float32), int(sr))
    print(f"[render_groove] selector on {audio_path}: {style.as_text()}")
    return style


# ------------------------------------------------------------ entry point

def _moves_to_render(args: argparse.Namespace) -> Iterable[tuple[str, GrooveStyle]]:
    if args.all_moves:
        for move in MOVE_PRIMITIVES.keys():
            yield move, _forced_style(move, bpm=args.bpm, intensity=args.intensity)
        return
    if args.audio:
        style = _style_from_audio(
            args.audio,
            panns_ckpt=getattr(args, "panns_ckpt", DEFAULT_PANNS_CKPT),
            genre_head=getattr(args, "genre_head", DEFAULT_GENRE_HEAD),
            arousal_head=getattr(args, "arousal_head", DEFAULT_AROUSAL_HEAD),
            mood_source=getattr(args, "mood_source", "va"),
            allow_untrained=getattr(args, "allow_untrained", False),
        )
        yield Path(args.audio).stem, style
        return
    yield args.style, _forced_style(args.style, bpm=args.bpm, intensity=args.intensity)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--audio", default=None,
                    help="WAV/MP3 clip → run selector → visualise.")
    ap.add_argument("--style", default="bob_nod",
                    choices=list(MOVE_PRIMITIVES.keys()),
                    help="Forced move when --audio is not given.")
    ap.add_argument("--all-moves", action="store_true",
                    help="Render every primitive in MOVE_PRIMITIVES.")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--rate", type=float, default=50.0,
                    help="Control rate (Hz, 30–50, NFR-7).")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--intensity", type=float, default=0.85)
    ap.add_argument("--use-ctx-arousal", action="store_true",
                    help="Multiply style.intensity by 0.5+0.5*ctx.arousal.")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--skip-render", action="store_true",
                    help="Write CSV only, no GIF (useful on headless boxes).")
    ap.add_argument("--narrate", action="store_true",
                    help="Print the window summary (perception → decision → "
                         "action) to stdout after each render.")
    ap.add_argument("--verbose", action="store_true",
                    help="Like --narrate, plus a per-beat trace of the "
                         "primary joint's peak angle.")
    ap.add_argument("--narrate-max-beats", type=int, default=None,
                    help="Cap the per-beat trace at N lines (verbose mode).")
    # Trained-checkpoint paths for the --audio path. Missing files → loud
    # INVALID stop unless --allow-untrained is set.
    ap.add_argument("--panns-ckpt", default=DEFAULT_PANNS_CKPT,
                    help="PANNs CNN14 backbone checkpoint (default: %(default)s).")
    ap.add_argument("--genre-head", default=DEFAULT_GENRE_HEAD,
                    help="Genre/mood TL head ckpt (default: %(default)s).")
    ap.add_argument("--arousal-head", default=DEFAULT_AROUSAL_HEAD,
                    help="Arousal/valence regression head ckpt (default: %(default)s).")
    ap.add_argument("--mood-source", default="va", choices=["head", "va"],
                    help="`va` (default) derives mood from learned A/V; "
                         "`head` uses the MTG-trained mood head (off by default).")
    ap.add_argument("--allow-untrained", action="store_true",
                    help="Permit random-weight selector for --audio path. "
                         "Without this flag, missing ckpts halt the run.")
    args = ap.parse_args()
    narrate_mode = "verbose" if args.verbose else ("on" if args.narrate else "off")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for tag, style in _moves_to_render(args):
        render_one(
            style=style,
            seconds=args.seconds,
            rate=args.rate,
            outdir=outdir,
            tag=tag,
            width=args.width,
            height=args.height,
            skip_render=args.skip_render,
            use_ctx_arousal=args.use_ctx_arousal,
            narrate_mode=narrate_mode,
            narrate_max_beats=args.narrate_max_beats,
        )


if __name__ == "__main__":
    main()
