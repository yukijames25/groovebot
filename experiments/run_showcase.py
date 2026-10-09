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

`--view` selects the camera pose (see render_groove.VIEWS). It is a
rendering-only knob, so every number in the summary table is identical
across views. Non-default views write `<stem>_front.*` / `<stem>_3q.*`
and never touch the side render.

Overwrite policy: `narration.txt` and `showcase_summary.md` are
view-independent, and `showcase_summary.md` carries hand-written
analysis appended after the generated section. Both are therefore
written only when absent; an existing file is compared against the
freshly computed content and left alone. Pass `--force-summary` to
deliberately regenerate the summary (this DROPS hand-written sections).

Usage:
    python -m experiments.run_showcase
    python -m experiments.run_showcase --clips data/raw/showcase/jp_uptempo.wav ...
    python -m experiments.run_showcase --seconds 8 --skip-render
    python -m experiments.run_showcase --view front
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
    DEFAULT_VIEW,
    VIEWS,
    _build_selector,
    render_one,
    view_suffix,
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
    view: str = DEFAULT_VIEW,
    arousal_mode: str = "bucket",
) -> dict:
    """Run the pipeline on one clip and return a comparison-row dict.

    Also writes (under `out_dir/<stem>/`):
      - <stem>[_cont][<view suffix>].gif  (if --skip-render not set, and
        the MuJoCo Renderer works)
      - <stem>[_cont][<view suffix>].csv  (per-tick JointCommand.targets
        — always)
      - narration[_cont].txt  (window summary + per-beat trace; written
        only if absent, since it does not depend on the camera)

    The `_cont` infix is added when `arousal_mode="continuous"` so the
    continuous run never overwrites the bucket-mode artefacts side-by-
    side. The bucket-mode default stays untouched.
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

    # Mode-aware suffix: continuous-mode outputs land next to the
    # bucket-mode ones with a `_cont` infix on every filename.
    mode_suffix = "_cont" if arousal_mode == "continuous" else ""
    tag = stem + mode_suffix

    # Render (also captures per-tick CSV).
    res = render_one(
        style=style,
        seconds=seconds,
        rate=rate,
        outdir=clip_out,
        tag=tag,
        width=480,
        height=360,
        skip_render=skip_render,
        use_ctx_arousal=False,
        narrate_mode="off",
        view=view,
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
    ) + "\n"
    narration_name = f"narration{mode_suffix}.txt"
    narr_path = clip_out / narration_name
    if not narr_path.exists():
        narr_path.write_text(narration, encoding="utf-8")
        print(f"[showcase] wrote {narr_path}", flush=True)
    else:
        # Narration derives from the style + the joint targets, neither of
        # which the camera touches, so a re-run under a different view must
        # reproduce it byte for byte. Verify rather than overwrite.
        existing = narr_path.read_text(encoding="utf-8")
        if existing == narration:
            print(f"[showcase] {narration_name} byte-identical "
                  "(view-independent) OK", flush=True)
        else:
            alt = clip_out / f"narration{mode_suffix}.{view}.txt"
            alt.write_text(narration, encoding="utf-8")
            print(
                f"[showcase] WARNING: narration differs from the existing "
                f"{narr_path.name}; kept the original and wrote the new "
                f"one to {alt.name}",
                file=sys.stderr, flush=True,
            )

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


def _write_summary_md(out_dir: Path, rows: list[dict], *,
                      force: bool = False,
                      arousal_mode: str = "bucket") -> Path | None:
    """Write showcase_summary[_cont].md, refusing to clobber an existing one.

    The committed summary carries hand-written analysis appended after
    the generated section, and the generated table itself is
    view-independent, so a re-render under a new camera has nothing to
    add. Returns the path written, or None when an existing file was
    deliberately left alone.

    `arousal_mode="continuous"` writes to `showcase_summary_cont.md` so
    the two modes' summaries live side-by-side.
    """
    name = "showcase_summary_cont.md" if arousal_mode == "continuous" \
           else "showcase_summary.md"
    path = out_dir / name
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
    body = "\n".join(parts) + "\n"
    if path.exists() and not force:
        print(
            f"[showcase] {path.name} already exists - left untouched "
            f"(it holds hand-written analysis, and the generated table is "
            f"view-independent). Pass --force-summary to regenerate.",
            flush=True,
        )
        return None
    path.write_text(body, encoding="utf-8")
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
    ap.add_argument("--view", default=DEFAULT_VIEW, choices=sorted(VIEWS),
                    help="Camera pose (rendering only - every number in "
                         "the summary is identical across views). "
                         "Non-default views write <stem>_front / <stem>_3q "
                         "and never overwrite the side render.")
    ap.add_argument("--arousal-mode", default="bucket",
                    choices=["bucket", "continuous"],
                    help="Arousal routing on the selector. 'bucket' "
                         "(default) is the legacy step-function; "
                         "'continuous' blends tertile anchors linearly "
                         "(see groovebot/style/table.py). Continuous mode "
                         "adds '_cont' to every output filename so the "
                         "two modes never overwrite each other.")
    ap.add_argument("--force-summary", action="store_true",
                    help="Regenerate showcase_summary.md even if it exists. "
                         "WARNING: drops any hand-written sections.")
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
        arousal_mode=args.arousal_mode,
    )

    rows = []
    for p in clip_paths:
        rows.append(run_one(
            clip_path=p, out_dir=out_dir, selector=selector,
            seconds=args.seconds, rate=args.rate,
            skip_render=args.skip_render, view=args.view,
            arousal_mode=args.arousal_mode,
        ))

    md_path = _write_summary_md(
        out_dir, rows,
        force=args.force_summary, arousal_mode=args.arousal_mode,
    )
    if md_path is not None:
        print(f"\n[showcase] summary written → {md_path}\n", flush=True)
    mode_suffix = "_cont" if args.arousal_mode == "continuous" else ""
    print(f"[showcase] view={args.view} "
          f"(suffix={view_suffix(args.view)!r})  "
          f"arousal_mode={args.arousal_mode} (suffix={mode_suffix!r})",
          flush=True)
    print(_format_markdown_table(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
