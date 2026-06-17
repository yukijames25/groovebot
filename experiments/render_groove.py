"""
experiments/render_groove.py — visualise the JointCommand bridge v1.

Drives the MuJoCo backend with `StyleGrooveGenerator` and writes a short
GIF of the resulting motion. Two modes:

  * `--style MOVE` (default if no audio): force a single move primitive
    from the `table.MOVES` vocabulary. Bypasses the selector — handy
    for inspecting one primitive at a time without needing a clip.
  * `--audio PATH`: run `GrooveStyleSelector` (v1/v2 CNN path, random
    weights unless you wire a checkpoint) on the clip and visualise
    whatever style the table picks.

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


def _style_from_audio(audio_path: str) -> GrooveStyle:
    import librosa
    from groovebot.style.select import GrooveStyleSelector

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    # v1/v2 CNN path; random weights are fine for visualisation since the
    # table still produces a legal move from whatever soft-max comes out.
    selector = GrooveStyleSelector()
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
        style = _style_from_audio(args.audio)
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
