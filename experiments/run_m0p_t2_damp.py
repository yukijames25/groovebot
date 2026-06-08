"""experiments/run_m0p_t2_damp.py — M0' Tier 2 DAMP-route alignment eval
(spec §9.x DAMP).

DAMP-VSEP / DAMP-S-AG ship pre-separated stems, so we never call Demucs.
For each arrangement we build the reference from the **backing** track:

- **beats**         librosa.beat.beat_track on backing (instrumental,
                    reliable, offline; this is *not* the shelved blind
                    vocal beat path, see §14.3).
- **chroma ref**    librosa.feature.chroma_cqt on backing.
- **melody ref**    F0 chroma from one or more vocal renditions:
                      * `designated`  pick one rendition as the melody
                        source; score the rest.
                      * `consensus`   leave-one-out frame-wise nanmedian
                        of every rendition's F0 contour.

For each remaining vocal we score **two paths** with the same `mir_eval`
harness as Tier 1 / Tier 2 (recording):

- **chroma path**   query vocal chroma -> backing chroma.
- **pitch path**    query vocal F0 chroma -> melody reference.

Outputs (under `--out-dir`):

    m0p_t2_damp_per_path.csv          one row per (rendition, path)
    m0p_t2_damp_per_kind.csv          means by feature_kind (chroma / pitch)
    m0p_t2_damp_per_arrangement.csv   means by arrangement_id
    m0p_t2_damp_overall.csv           overall means
    <arrangement>__<rendition>__<kind>.png   per-row overlay PNG

CLI:

    python -m experiments.run_m0p_t2_damp \\
        --root    data/m0p_t2_damp \\
        --out-dir data/m0p_t2_damp_work \\
        [--melody-mode designated|consensus] \\
        [--designated <rendition_id>] \\
        [--sr 22050] [--hop 512] [--no-png]

The runner is CPU / librosa-only. See spec §9.x DAMP route for limits
(rendition timing is largely fixed to backing; pitch path uses real singing
as a humming proxy and may be optimistic for true humming).
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import (
    consensus_f0,
    extract_align_features,
    f0_to_pitch_chroma,
    pyin_f0,
)
from tools.eval_beat import score_beats
from tools.ingest_damp import DampArrangement, DampRendition, discover_arrangements


# --------------------------------------------------------------------------- #
# Per-arrangement reference
# --------------------------------------------------------------------------- #
@dataclass
class DampReferenceBundle:
    """Per-arrangement reference state.

    `melody_designated` is populated when `melody_mode == 'designated'`. In
    `consensus` mode we instead carry every rendition's F0 contour in
    `vocal_f0` so each query rendition can build a fresh leave-one-out
    melody reference on demand.
    """
    arrangement_id: str
    beats: np.ndarray
    backing_audio: np.ndarray
    chroma: np.ndarray
    sample_rate: int
    hop_length: int
    # One of these two is the source of the melody reference, depending on
    # the run mode.
    melody_designated: np.ndarray | None = None     # (12, T)
    vocal_f0: dict[str, np.ndarray] = field(default_factory=dict)


def beats_from_backing(
    backing_audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
) -> np.ndarray:
    """librosa beat-tracking on the backing -> beat times in seconds."""
    mono = _to_mono(backing_audio)
    _tempo, beats_sec = librosa.beat.beat_track(
        y=mono, sr=sr, hop_length=hop_length, units="time",
    )
    return np.asarray(beats_sec, dtype=float)


def chroma_from_backing(
    backing_audio: np.ndarray, sr: int, *, hop_length: int = 512,
) -> np.ndarray:
    mono = _to_mono(backing_audio)
    return librosa.feature.chroma_cqt(
        y=mono, sr=sr, hop_length=hop_length,
    ).astype(np.float32)


def melody_from_consensus(
    f0_contours: Sequence[np.ndarray],
) -> np.ndarray:
    """Frame-wise nanmedian F0 over the supplied contours -> (12, T)
    one-hot pitch chroma. Empty input -> empty (12, 0) matrix."""
    if not f0_contours:
        return np.zeros((12, 0), dtype=np.float32)
    f0 = consensus_f0(f0_contours)
    return f0_to_pitch_chroma(f0)


# --------------------------------------------------------------------------- #
# Per-arrangement run loop
# --------------------------------------------------------------------------- #
def _load_mono_wav(path: Path, expected_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != expected_sr:
        raise ValueError(
            f"{path.name}: sr={sr}, expected {expected_sr}; resample upstream"
        )
    return audio


def run_arrangement(
    arrangement: DampArrangement,
    out_dir: Path,
    aligner: OfflineDTWAligner,
    *,
    melody_mode: str = "designated",
    designated: str | None = None,
    make_png: bool = True,
) -> list[dict]:
    """Score every rendition of `arrangement` against both paths.

    Returns a list of row dicts (one per `(rendition, path)` pair).
    """
    if melody_mode not in ("designated", "consensus"):
        raise ValueError(f"unknown melody_mode: {melody_mode!r}")
    if len(arrangement.renditions) < 2:
        # designated needs ref + at least one query; consensus needs at
        # least 2 contours to be meaningful.
        return []

    sr = aligner.sample_rate
    hop = aligner.hop_length

    backing = _load_mono_wav(arrangement.backing_wav, sr)
    beats = beats_from_backing(backing, sr, hop_length=hop)
    chroma = chroma_from_backing(backing, sr, hop_length=hop)

    # Load every vocal and (for the pitch path) cache its F0 contour.
    vocals: dict[str, np.ndarray] = {}
    f0s: dict[str, np.ndarray] = {}
    for r in arrangement.renditions:
        v = _load_mono_wav(r.vocal_wav, sr)
        vocals[r.rendition_id] = v
        f0s[r.rendition_id] = pyin_f0(v, sr, hop_length=hop)

    # Decide which renditions are query candidates.
    rendition_ids = [r.rendition_id for r in arrangement.renditions]
    if melody_mode == "designated":
        ref_id = designated or rendition_ids[0]
        if ref_id not in f0s:
            raise ValueError(
                f"designated rendition {ref_id!r} not in arrangement "
                f"{arrangement.arrangement_id!r}"
            )
        melody_ref = f0_to_pitch_chroma(f0s[ref_id])
        query_ids = [rid for rid in rendition_ids if rid != ref_id]
    else:  # consensus
        melody_ref = None
        query_ids = list(rendition_ids)

    rows: list[dict] = []
    for qid in query_ids:
        if melody_mode == "consensus":
            # leave-one-out median
            others = [f for k, f in f0s.items() if k != qid]
            melody = melody_from_consensus(others)
        else:
            melody = melody_ref
        query_audio = vocals[qid]

        for kind in ("chroma", "pitch"):
            row = _score_one_path(
                arrangement_id=arrangement.arrangement_id,
                rendition_id=qid,
                query_audio=query_audio,
                query_f0=f0s[qid],
                sr=sr, hop=hop,
                backing_chroma=chroma,
                melody_chroma=melody,
                aligner=aligner,
                gt_beats=beats,
                feature_kind=kind,
            )
            rows.append(row["scores"])
            if make_png:
                png = (out_dir
                       / f"{arrangement.arrangement_id}__{qid}__{kind}.png")
                _save_overlay_png(
                    query_audio, sr, beats, row["recovered"], row["wp"],
                    str(png),
                    title=(f"{arrangement.arrangement_id}/{qid} "
                           f"kind={kind}"),
                )
    return rows


def _score_one_path(
    *,
    arrangement_id: str,
    rendition_id: str,
    query_audio: np.ndarray,
    query_f0: np.ndarray,
    sr: int, hop: int,
    backing_chroma: np.ndarray,
    melody_chroma: np.ndarray | None,
    aligner: OfflineDTWAligner,
    gt_beats: np.ndarray,
    feature_kind: str,
) -> dict:
    """Run one path and return the score row plus diagnostics for the PNG."""
    t0 = time.perf_counter()
    if feature_kind == "chroma":
        query_feats = extract_align_features(
            query_audio, sr, kind="chroma", hop_length=hop,
        )
        ref_feats = backing_chroma
    elif feature_kind == "pitch":
        query_feats = f0_to_pitch_chroma(query_f0)
        ref_feats = melody_chroma
        if ref_feats is None or ref_feats.shape[1] == 0:
            raise ValueError(
                f"{arrangement_id}/{rendition_id}: pitch path requires a "
                "non-empty melody reference"
            )
    else:
        raise ValueError(f"unknown feature_kind: {feature_kind!r}")
    wp = aligner.align(query_feats, ref_feats)
    recovered = aligner.map_reference_beats(wp, gt_beats)
    proc_sec = time.perf_counter() - t0

    audio_sec = len(query_audio) / sr
    scores = score_beats(
        track=f"{rendition_id}",
        bpm=None,
        gt=gt_beats,
        est=recovered,
        audio_sec=audio_sec,
        proc_sec=proc_sec,
    )
    row = asdict(scores)
    row["arrangement_id"] = arrangement_id
    row["feature_kind"] = feature_kind
    return {"scores": row, "recovered": recovered, "wp": wp}


# --------------------------------------------------------------------------- #
# PNG + CSV
# --------------------------------------------------------------------------- #
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
        Line2D([0], [0], color="#888888", label="query vocal"),
        Line2D([0], [0], color="#1f77b4", label="backing-derived beat"),
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
    """Return (per_kind, per_arrangement, overall) means."""
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
    per_kind = [{"feature_kind": k, **_means(g)}
                for k, g in sorted(by_kind.items())]

    by_arr: dict[str, list[dict]] = {}
    for r in rows:
        by_arr.setdefault(r["arrangement_id"], []).append(r)
    per_arr = [{"arrangement_id": a, **_means(g)}
               for a, g in sorted(by_arr.items())]

    overall = (_means(rows) if rows
               else {"n": 0, "f_mean": 0.0, "cmlt_mean": 0.0,
                     "amlt_mean": 0.0, "rt_mean": 0.0})
    return per_kind, per_arr, overall


def save_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Pipeline + CLI
# --------------------------------------------------------------------------- #
def run_pipeline(
    root: Path,
    out_dir: Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    melody_mode: str = "designated",
    designated: str | None = None,
    make_png: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligner = OfflineDTWAligner(sample_rate=sample_rate, hop_length=hop_length)
    rows: list[dict] = []
    for a in discover_arrangements(Path(root)):
        info = sf.info(str(a.backing_wav))
        if info.samplerate != sample_rate:
            if verbose:
                print(f"skip {a.arrangement_id}: backing sr="
                      f"{info.samplerate}, need {sample_rate}",
                      file=sys.stderr)
            continue
        try:
            arr_rows = run_arrangement(
                a, out_dir, aligner,
                melody_mode=melody_mode,
                designated=designated,
                make_png=make_png,
            )
        except Exception as e:
            print(f"FAILED {a.arrangement_id}: {e}", file=sys.stderr)
            continue
        rows.extend(arr_rows)
        if verbose:
            for r in arr_rows:
                print(f"  {r['arrangement_id']}/{r['track']} "
                      f"kind={r['feature_kind']}  "
                      f"F={r['f_measure']:.3f}  CMLt={r['cmlt']:.3f}  "
                      f"AMLt={r['amlt']:.3f}  RT={r['rt_factor']:.2f}x")
    per_kind, per_arr, overall = aggregate(rows)
    save_csv(rows,      out_dir / "m0p_t2_damp_per_path.csv")
    save_csv(per_kind,  out_dir / "m0p_t2_damp_per_kind.csv")
    save_csv(per_arr,   out_dir / "m0p_t2_damp_per_arrangement.csv")
    save_csv([overall], out_dir / "m0p_t2_damp_overall.csv")
    return rows, per_kind, per_arr, overall


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        axis = 0 if a.shape[0] < a.shape[-1] else -1
        a = a.mean(axis=axis)
    return a.astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True,
                    help="root dir containing arrangement subdirectories "
                         "(see tools/ingest_damp.py for layout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--melody-mode",
                    choices=("designated", "consensus"),
                    default="designated")
    ap.add_argument("--designated", default=None,
                    help="rendition_id to use as the melody reference "
                         "(designated mode only). Default: first by id.")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--no-png", action="store_true")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_pipeline(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        sample_rate=args.sr,
        hop_length=args.hop,
        melody_mode=args.melody_mode,
        designated=args.designated,
        make_png=not args.no_png,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
