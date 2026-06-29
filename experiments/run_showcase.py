"""
experiments/run_showcase.py — end-to-end JointCommand bridge showcase.

Runs one full pipeline per audio clip:

    audio → PANNs CNN14 → trained heads (genre / arousal+valence)
          → mood (V/A circumplex by default)
          → table → GrooveStyle
          → StyleGrooveGenerator → JointCommand → MuJoCo backend
          → GIF + per-tick CSV + narration.txt
          → comparison row in showcase_summary.md

Trained checkpoints (PANNs + genre TL + arousal/valence TL) are required;
missing files trigger a loud `INVALID: trained checkpoint not found`
unless `--allow-untrained` is passed.

Default 4-clip set lives at `data/raw/showcase/` (gitignored). Each
clip's outputs go to `data/renders/showcase/<stem>/`. See
`data/renders/showcase/SOURCES.md` for clip provenance + licenses.

Usage:
    python -m experiments.run_showcase
    python -m experiments.run_showcase --clips data/raw/showcase/jp_uptempo.wav ...
    python -m experiments.run_showcase --seconds 8 --skip-render
"""
from __future__ import annotations
import argparse
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.render_groove import (
    DEFAULT_AROUSAL_HEAD,
    DEFAULT_GENRE_HEAD,
    DEFAULT_PANNS_CKPT,
    _build_selector,
    render_one,
)
from groovebot.style.narrate import narrate
from groovebot.style.select import GrooveStyle


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS_DIR = _REPO_ROOT / "data" / "raw" / "showcase"
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "renders" / "showcase"
DEFAULT_CLIPS = [
    "western_metal.wav",
    "western_classical.wav",
    "jp_uptempo.wav",
    "jp_ballad.wav",
]

# Move → accent-phase category (matches the v1.2 phase contract in
# SYSTEM_SPEC.md §14 v1.2 and the MOVE_PRIMITIVES comment table).
ACCENT_PHASE: dict[str, str] = {
    "headbang":      "on-beat",
    "bob_nod":       "on-beat",
    "fist_pump":     "on-beat",
    "clap":          "backbeat (2 & 4)",
    "sway":          "continuous (sin(πb))",
    "rock":          "continuous (sin(πb))",
    "penlight_wave": "continuous (sin(πb))",
    "quiet_listen":  "minimal",
}


def _infer_lang(stem: str) -> str:
    if stem.startswith("jp_") or stem.startswith("ja_"):
        return "ja"
    if stem.startswith("western_") or stem.startswith("en_"):
        return "en"
    return "?"


def _top_prob(probs: dict[str, float]) -> tuple[str, float]:
    label = max(probs, key=probs.get)
    return label, float(probs[label])


def _infer_valence(style: GrooveStyle, selector) -> float | None:
    """Run the selector's regression head once if available, return valence.

    The v3 `GrooveStyle` keeps only arousal in its public fields, but
    the regression head also emits valence. For the showcase comparison
    table we want both, so we re-embed and read the head's valence
    output. Returns None if no regression head is wired.
    """
    return None  # placeholder; real value injected by run_one


def run_one(
    *,
    clip_path: Path,
    out_dir: Path,
    selector,
    seconds: float,
    rate: float,
    skip_render: bool,
) -> dict:
    """Run the pipeline on one clip and return a comparison-row dict.

    Also writes (under `out_dir/<stem>/`):
      - <stem>.gif   (if --skip-render not set, and MuJoCo Renderer works)
      - <stem>.csv   (per-tick JointCommand.targets — always)
      - narration.txt  (window summary + per-beat trace)
    """
    import librosa

    print(f"\n[showcase] === {clip_path.name} ===", flush=True)
    y, sr = librosa.load(str(clip_path), sr=None, mono=True)
    audio = np.asarray(y, dtype=np.float32)
    style = selector.select(audio, int(sr))

    # Pull valence out of the regression head directly (one extra embed
    # for clarity, then thrown away — not on the realtime control path).
    valence: float | None = None
    if getattr(selector, "regression_head", None) is not None:
        emb = selector.backbone.embed(audio, int(sr))
        _, v_unit = selector._affect_from_emb(emb)
        valence = float(v_unit)

    print(f"[showcase] {clip_path.name}: {style.as_text()}", flush=True)

    stem = clip_path.stem
    clip_out = out_dir / stem
    clip_out.mkdir(parents=True, exist_ok=True)

    # Render (also captures per-tick CSV).
    res = render_one(
        style=style,
        seconds=seconds,
        rate=rate,
        outdir=clip_out,
        tag=stem,
        width=480,
        height=360,
        skip_render=skip_render,
        use_ctx_arousal=False,
        narrate_mode="off",
    )

    # Write narration.txt (window summary + verbose beat trace, taken
    # from the same per-tick targets we just CSV-dumped).
    commands = []
    with open(res.csv_path, "r", encoding="utf-8") as f:
        import csv
        rd = csv.reader(f)
        header = next(rd)
        joint_cols = header[1:]
        for row in rd:
            commands.append({n: float(v) for n, v in zip(joint_cols, row[1:])})

    narration = narrate(
        style, commands, rate=rate, seconds=seconds,
        verbose=True, max_beats=None,
    )
    (clip_out / "narration.txt").write_text(narration + "\n", encoding="utf-8")

    top_genre, top_genre_p = _top_prob(style.genre_probs)
    top_mood, top_mood_p = _top_prob(style.mood_probs)
    return {
        "clip": stem,
        "lang": _infer_lang(stem),
        "genre": f"{top_genre} ({top_genre_p:.2f})",
        "arousal": f"{style.arousal:.2f}",
        "valence": "—" if valence is None else f"{valence:.2f}",
        "mood": f"{top_mood} ({top_mood_p:.2f})",
        "style_move": style.move,
        "intensity": f"{style.intensity:.2f}",
        "dominant_move": style.move,
        "accent_phase": ACCENT_PHASE.get(style.move, "?"),
        "tempo_bpm": f"{style.tempo_bpm:.0f}",
        "arousal_bucket": style.arousal_bucket,
    }


def _format_markdown_table(rows: list[dict]) -> str:
    cols = [
        ("clip", "clip"),
        ("lang", "lang"),
        ("genre", "genre (top-1, p)"),
        ("arousal", "arousal"),
        ("valence", "valence"),
        ("mood", "mood (top-1, p)"),
        ("style_move", "style"),
        ("intensity", "intensity"),
        ("dominant_move", "dominant move"),
        ("accent_phase", "accent phase"),
    ]
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r[k]) for k, _ in cols) + " |")
    return "\n".join([head, sep, *body])


def _write_summary_md(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "showcase_summary.md"
    parts = [
        "# Showcase summary",
        "",
        "JointCommand bridge v1.2 end-to-end run: per-clip selector output",
        "and dominant move. See `SOURCES.md` for clip licenses; per-clip",
        "GIF / CSV / narration.txt under `<clip_stem>/`.",
        "",
        "## Comparison table",
        "",
        _format_markdown_table(rows),
        "",
        "## Per-clip details",
        "",
    ]
    for r in rows:
        parts += [
            f"### {r['clip']} ({r['lang']})",
            f"- tempo: {r['tempo_bpm']} BPM, arousal bucket: {r['arousal_bucket']}",
            f"- style: `{r['style_move']}` @ intensity {r['intensity']} → "
            f"accent phase **{r['accent_phase']}**",
            "",
        ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--clips", nargs="*", default=None,
                    help="Audio clip paths. Default: 4-clip set under "
                         "data/raw/showcase/.")
    ap.add_argument("--clip-dir", default=str(DEFAULT_CLIPS_DIR),
                    help="Directory for default clip set (default: %(default)s).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Output root (default: %(default)s).")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--skip-render", action="store_true",
                    help="Skip GIF rendering (CSV + narration still produced).")
    ap.add_argument("--panns-ckpt", default=DEFAULT_PANNS_CKPT)
    ap.add_argument("--genre-head", default=DEFAULT_GENRE_HEAD)
    ap.add_argument("--arousal-head", default=DEFAULT_AROUSAL_HEAD)
    ap.add_argument("--mood-source", default="va", choices=["head", "va"])
    ap.add_argument("--allow-untrained", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clips:
        clip_paths = [Path(c) for c in args.clips]
    else:
        clip_paths = [Path(args.clip_dir) / name for name in DEFAULT_CLIPS]

    missing_clips = [p for p in clip_paths if not p.is_file()]
    if missing_clips:
        print("ERROR: missing clip(s):", file=sys.stderr)
        for p in missing_clips:
            print(f"  - {p}", file=sys.stderr)
        return 2

    selector = _build_selector(
        panns_ckpt=args.panns_ckpt,
        genre_head=args.genre_head,
        arousal_head=args.arousal_head,
        mood_source=args.mood_source,
        allow_untrained=args.allow_untrained,
    )

    rows = []
    for p in clip_paths:
        rows.append(run_one(
            clip_path=p, out_dir=out_dir, selector=selector,
            seconds=args.seconds, rate=args.rate,
            skip_render=args.skip_render,
        ))

    md_path = _write_summary_md(out_dir, rows)
    print(f"\n[showcase] summary written → {md_path}\n", flush=True)
    print(_format_markdown_table(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
