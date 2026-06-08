"""experiments/run_m0p_t2.py — M0' Tier 2 alignment evaluation pipeline.

Per spec §9.x M0' Tier 2: align real renditions (singing or humming) of a
known song to that song's reference timeline. Tier 1 only checks the
alignment *mechanism* against the song's own time-stretched audio; Tier 2
checks it against a *different performance* — different timbre, different
micro-timing, no harmonic stack when humming — which is the real test.

Input layout (per spec §9.x):

    data/m0p_t2/<song>/
        original.wav        full-mix reference (Demucs vocal stem comes from this)
        original.beats      reference beat times (one per line, seconds)
        rendition_sing*.wav (optional)  singing rendition(s) — aligned via chroma
        rendition_hum*.wav  (optional)  humming rendition(s) — aligned via pyin melody
        rendition_*.wav     (optional)  default to chroma when the kind is ambiguous
        gt.beats            ground-truth beat times for the renditions

Outputs (under `--out-dir`):

    m0p_t2_per_rendition.csv   one row per rendition (F / CMLt / AMLt / RT)
    m0p_t2_per_kind.csv        means by rendition kind (chroma vs pitch)
    m0p_t2_per_song.csv        means by song
    m0p_t2_overall.csv         overall means
    <song>_<rendition>.png     per-rendition overlay (query audio + GT vs
                               recovered beats + DTW warp path)

CLI:

    python -m experiments.run_m0p_t2 \\
        --root    data/m0p_t2 \\
        --out-dir data/m0p_t2_work \\
        [--sr 22050] [--hop 512] [--no-png]

Dependencies: librosa locally. Demucs is loaded lazily through
`groovebot.align.reference.build_reference()` (Colab/Kaggle profile);
the runner skips songs whose `build_reference` fails (e.g. Demucs absent).
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import extract_align_features
from groovebot.align.reference import ReferenceBundle, build_reference
from tools.eval_beat import load_beat_annotation, score_beats


@dataclass
class SongInputs:
    song_dir: Path
    original_wav: Path
    original_beats: Path
    gt_beats: Path
    renditions: list[Path]


def discover_songs(root: Path) -> list[SongInputs]:
    """List song directories under `root` that have all required files."""
    songs: list[SongInputs] = []
    if not root.exists():
        return songs
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        orig = d / "original.wav"
        ob = d / "original.beats"
        gt = d / "gt.beats"
        if not (orig.exists() and ob.exists() and gt.exists()):
            continue
        renditions = sorted(d.glob("rendition_*.wav"))
        if not renditions:
            continue
        songs.append(SongInputs(d, orig, ob, gt, renditions))
    return songs


def classify_rendition_kind(filename: str) -> str:
    """Return 'pitch' (pyin melody features) for humming filenames and
    'chroma' for everything else.

    Filenames are matched case-insensitively; 'hum' or 'humming' anywhere
    in the basename routes to the melody/pyin path.
    """
    n = filename.lower()
    if "hum" in n:
        return "pitch"
    return "chroma"


def score_rendition(
    rendition_wav: Path,
    bundle: ReferenceBundle,
    gt_beats: np.ndarray,
    aligner: OfflineDTWAligner,
    feature_kind: str,
) -> tuple[dict, np.ndarray, np.ndarray, int, np.ndarray]:
    """Align one rendition to `bundle` and score against `gt_beats`.

    Returns (row, warp_path, query_audio, sr, recovered_beats) so callers
    can render an overlay PNG without re-reading the wav.
    """
    query_audio, sr = sf.read(str(rendition_wav), dtype="float32",
                              always_2d=False)
    if query_audio.ndim > 1:
        query_audio = query_audio.mean(axis=1)
    if sr != bundle.sample_rate:
        raise ValueError(
            f"{rendition_wav.name}: sr={sr}, bundle sr={bundle.sample_rate}; "
            "resample upstream"
        )
    t0 = time.perf_counter()
    query_feats = extract_align_features(
        query_audio, sr, kind=feature_kind, hop_length=bundle.hop_length,
    )
    ref_feats = bundle.chroma if feature_kind == "chroma" else bundle.melody
    wp = aligner.align(query_feats, ref_feats)
    recovered = aligner.map_reference_beats(wp, bundle.beats)
    proc_sec = time.perf_counter() - t0
    audio_sec = len(query_audio) / sr
    scores = score_beats(
        track=rendition_wav.stem,
        bpm=None,
        gt=gt_beats,
        est=recovered,
        audio_sec=audio_sec,
        proc_sec=proc_sec,
    )
    return asdict(scores), wp, query_audio, sr, recovered


def run_song(
    inputs: SongInputs,
    out_dir: Path,
    aligner: OfflineDTWAligner,
    bundle: ReferenceBundle | None = None,
    make_png: bool = True,
) -> list[dict]:
    """Score every rendition in one song dir. Builds `bundle` from
    `original.wav` + `original.beats` if not supplied (Demucs runs there).
    """
    if bundle is None:
        audio, sr = sf.read(str(inputs.original_wav), dtype="float32",
                            always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        beats = load_beat_annotation(str(inputs.original_beats))
        bundle = build_reference(audio, sr, beats,
                                 hop_length=aligner.hop_length)
    gt_beats = load_beat_annotation(str(inputs.gt_beats))
    rows: list[dict] = []
    for r in inputs.renditions:
        kind = classify_rendition_kind(r.name)
        row, wp, query_audio, sr, recovered = score_rendition(
            r, bundle, gt_beats, aligner, kind,
        )
        row["song"] = inputs.song_dir.name
        row["feature_kind"] = kind
        rows.append(row)
        if make_png:
            png_path = out_dir / f"{inputs.song_dir.name}_{r.stem}.png"
            _save_overlay_png(
                query_audio, sr, gt_beats, recovered, wp, str(png_path),
                title=f"{inputs.song_dir.name}/{r.stem}  kind={kind}",
            )
    return rows


def _save_overlay_png(
    query: np.ndarray, sr: int,
    gt: np.ndarray, est: np.ndarray, wp: np.ndarray,
    path: str, title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax_wav, ax_wp) = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(len(query)) / sr
    ax_wav.plot(t, query, linewidth=0.4, color="#888888")
    for x in gt:
        ax_wav.axvline(x, color="#1f77b4", linewidth=0.8, alpha=0.7)
    for x in est:
        ax_wav.axvline(x, color="#d62728", linewidth=0.8, alpha=0.7,
                       linestyle="--")
    ax_wav.legend(handles=[
        Line2D([0], [0], color="#888888", label="query audio"),
        Line2D([0], [0], color="#1f77b4", label="GT beat"),
        Line2D([0], [0], color="#d62728", linestyle="--",
               label="recovered beat"),
    ], loc="upper right")
    ax_wav.set_xlabel("time [s]")
    ax_wav.set_title(title)
    ax_wav.set_xlim(0, t[-1] if len(t) else 1.0)

    if wp.size:
        ax_wp.plot(wp[:, 1], wp[:, 0], color="#2ca02c", linewidth=0.7)
    ax_wp.set_xlabel("reference frame")
    ax_wp.set_ylabel("query frame")
    ax_wp.set_title("DTW warp path")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return (per_kind, per_song, overall) means of F / CMLt / AMLt / RT."""
    def _means(group: list[dict]) -> dict:
        return {
            "n": len(group),
            "f_mean": float(np.mean([g["f_measure"] for g in group])),
            "cmlt_mean": float(np.mean([g["cmlt"] for g in group])),
            "amlt_mean": float(np.mean([g["amlt"] for g in group])),
            "rt_mean": float(np.mean([g["rt_factor"] for g in group])),
        }

    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["feature_kind"], []).append(r)
    per_kind = [{"feature_kind": k, **_means(g)} for k, g in sorted(by_kind.items())]

    by_song: dict[str, list[dict]] = {}
    for r in rows:
        by_song.setdefault(r["song"], []).append(r)
    per_song = [{"song": s, **_means(g)} for s, g in sorted(by_song.items())]

    overall = (_means(rows) if rows
               else {"n": 0, "f_mean": 0.0, "cmlt_mean": 0.0,
                     "amlt_mean": 0.0, "rt_mean": 0.0})
    return per_kind, per_song, overall


def save_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run_pipeline(
    root: Path,
    out_dir: Path,
    sample_rate: int = 22050,
    hop_length: int = 512,
    make_png: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligner = OfflineDTWAligner(sample_rate=sample_rate, hop_length=hop_length)
    rows: list[dict] = []
    for s in discover_songs(Path(root)):
        info = sf.info(str(s.original_wav))
        if info.samplerate != sample_rate:
            if verbose:
                print(f"skip {s.song_dir.name}: original sr={info.samplerate},"
                      f" need {sample_rate}", file=sys.stderr)
            continue
        try:
            song_rows = run_song(s, out_dir, aligner, make_png=make_png)
        except Exception as e:
            print(f"FAILED {s.song_dir.name}: {e}", file=sys.stderr)
            continue
        rows.extend(song_rows)
        if verbose:
            for r in song_rows:
                print(f"  {r['song']}/{r['track']}: kind={r['feature_kind']}  "
                      f"F={r['f_measure']:.3f}  CMLt={r['cmlt']:.3f}  "
                      f"AMLt={r['amlt']:.3f}  RT={r['rt_factor']:.2f}x")
    per_kind, per_song, overall = aggregate(rows)
    save_csv(rows,      out_dir / "m0p_t2_per_rendition.csv")
    save_csv(per_kind,  out_dir / "m0p_t2_per_kind.csv")
    save_csv(per_song,  out_dir / "m0p_t2_per_song.csv")
    save_csv([overall], out_dir / "m0p_t2_overall.csv")
    return rows, per_kind, per_song, overall


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True,
                    help="root dir containing <song>/ subdirectories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--no-png", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        sample_rate=args.sr,
        hop_length=args.hop,
        make_png=not args.no_png,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
