"""experiments/compare_va_mood_vs_mtg.py — V/A-derived mood vs MTG mood head.

Runs both mood pipelines on the **same** clip set and reports their
agreement, per-class confusion, and (optionally) accuracy against a
held-out ground-truth label. This is the material for deciding whether
to retire the MTG-trained mood head in favour of the DEAM V/A → mood
map (see SYSTEM_SPEC §14 "affect 統合 (DEAM)" and README §"v3 affect
integration").

The script does NOT train anything. It loads:

  - PANNs CNN14 backbone (frozen, shared by both pipelines)
  - StyleHead with mood weights (the "MTG" pipeline)
  - StyleRegressionHead with arousal/valence weights (the "DEAM"
    pipeline, mood derived via `mood_from_va.mood_probs_from_va`)

and for each clip in `--manifest`, predicts a mood with each pipeline.

Two manifests are supported (auto-detected by column names):
  - MTG-style: rows with `path,mood_class,artist_id,...`. The
    `mood_class` column becomes the ground-truth label for accuracy.
  - generic:   rows with `path[,mood_gt]`. `mood_gt` optional.

Report (`report.json` + console):
  - agreement_rate       (% of clips where MTG-head and VA argmax match)
  - confusion_matrix     (MTG-head row × VA-argmax col)
  - per_class_profile    (avg arousal/valence per MTG-predicted class)
  - calm_sad_stability   (specific axis: confusion between calm and sad
                           in both pipelines — the spec calls out this
                           pair as the unstable one for MTG)
  - accuracy_against_gt  (per pipeline, when mood_gt present)

Use `--include-aux-moods` to enable the (draft) epic/dark prototypes
in the V/A map; default stays at the 4 clean quadrants so the
comparison isolates the V/A → quadrant story before piling on extras.

`--synthetic-stub` mode skips audio I/O + PANNs and generates random
predictions for a few clips so the report shape can be validated
without the trained checkpoints or audio on disk.

CLI (real):
    python -m experiments.compare_va_mood_vs_mtg \\
        --manifest data/mtg_moodtheme_manifest.csv \\
        --audio-root data/raw/mtg_moodtheme \\
        --panns-ckpt data/raw/Cnn14_mAP=0.431.pth \\
        --mood-head data/style_v3_mood/style_head_mood.pt \\
        --regression-head data/style_v3_arousal/style_head_arousal.pt \\
        --out-dir data/style_va_vs_mtg_report

CLI (stub):
    python -m experiments.compare_va_mood_vs_mtg \\
        --synthetic-stub --out-dir data/va_vs_mtg_stub
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.deam import sam_to_unit
from groovebot.style.model import MOODS, StyleHead, StyleRegressionHead
from groovebot.style.mood_from_va import (
    DEFAULT_QUADRANT_PROTOTYPES,
    PROTOTYPES_WITH_AUX,
    mood_probs_from_va,
    quadrant_label,
)


# ---------------------------------------------------------------- manifest

@dataclass
class ClipRow:
    path: Path
    mood_gt: str | None  # optional ground-truth mood label


def read_manifest(manifest_csv: Path, audio_root: Path) -> list[ClipRow]:
    out: list[ClipRow] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        path_col = cols.get("path")
        if path_col is None:
            raise ValueError(
                f"manifest {manifest_csv} missing `path` column "
                f"(got {reader.fieldnames!r})"
            )
        gt_col = cols.get("mood_class") or cols.get("mood_gt")
        for r in reader:
            apath = audio_root / r[path_col]
            if not apath.exists():
                continue
            gt = r.get(gt_col) if gt_col else None
            if gt is not None and gt not in MOODS:
                gt = None
            out.append(ClipRow(path=apath, mood_gt=gt))
    return out


# --------------------------------------------------- predictions per clip

def load_head_state(path: Path) -> dict:
    ck = torch.load(str(path), map_location="cpu")
    return ck.get("state_dict", ck) if isinstance(ck, dict) else ck


def predict_clip(
    audio_path: Path,
    backbone: PannsBackbone,
    mood_head: StyleHead,
    reg_head: StyleRegressionHead,
    *,
    window_sec: float = 10.0,
    aux: bool = False,
) -> dict:
    """Run both pipelines on one clip. Returns the per-clip prediction
    dict (the `per_clip.csv` row + JSON payload)."""
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = int(window_sec * sr)
    if len(audio) > n:
        start = (len(audio) - n) // 2
        audio = audio[start: start + n]

    emb = backbone.embed(audio, sr)
    x = torch.from_numpy(emb).unsqueeze(0)
    with torch.no_grad():
        mood_logits = mood_head(x)["mood"]
        mtg_probs = torch.softmax(mood_logits, dim=-1).squeeze(0).numpy()
        reg = reg_head(x)
        a_raw = float(reg["arousal"].item())
        v_raw = float(reg["valence"].item())

    a_unit = sam_to_unit(a_raw)
    v_unit = sam_to_unit(v_raw)
    va_probs = mood_probs_from_va(
        a_unit, v_unit,
        prototypes=PROTOTYPES_WITH_AUX if aux else DEFAULT_QUADRANT_PROTOTYPES,
    )
    return {
        "mtg_mood": MOODS[int(np.argmax(mtg_probs))],
        "mtg_probs": {m: float(p) for m, p in zip(MOODS, mtg_probs)},
        "va_mood": max(va_probs, key=va_probs.get),
        "va_probs": va_probs,
        "arousal_unit": float(a_unit),
        "valence_unit": float(v_unit),
        "arousal_sam": float(a_raw),
        "valence_sam": float(v_raw),
        "va_quadrant": quadrant_label(a_unit, v_unit),
    }


# ----------------------------------------------- aggregate metrics

def confusion_matrix(rows: list[dict], key_a: str, key_b: str) -> dict:
    """Square confusion-style matrix counts[row label][col label]."""
    out = {m: {n: 0 for n in MOODS} for m in MOODS}
    for r in rows:
        a, b = r.get(key_a), r.get(key_b)
        if a in out and b in out[a]:
            out[a][b] += 1
    return out


def per_class_profile(rows: list[dict], by_key: str) -> dict:
    """Mean / std of arousal & valence per predicted class for one of
    the pipelines. Shows whether the V/A coordinates the MTG-head
    keeps assigning to (say) `calm` are consistent with the V/A map's
    `calm` prototype."""
    profile = {m: {"n": 0, "a_mean": 0.0, "a_std": 0.0,
                   "v_mean": 0.0, "v_std": 0.0} for m in MOODS}
    by_class: dict[str, list[tuple[float, float]]] = {m: [] for m in MOODS}
    for r in rows:
        cls = r.get(by_key)
        if cls in by_class:
            by_class[cls].append((r["arousal_unit"], r["valence_unit"]))
    for cls, pairs in by_class.items():
        if not pairs:
            continue
        arr = np.array(pairs, dtype=np.float64)
        profile[cls]["n"] = len(pairs)
        profile[cls]["a_mean"] = float(arr[:, 0].mean())
        profile[cls]["a_std"] = float(arr[:, 0].std())
        profile[cls]["v_mean"] = float(arr[:, 1].mean())
        profile[cls]["v_std"] = float(arr[:, 1].std())
    return profile


def calm_sad_stability(rows: list[dict]) -> dict:
    """Counts of the four (mtg, va) cells in {calm, sad} × {calm, sad}.
    The MTG head collapses sad → calm 62% of the time on the v3 mood
    report; this measures whether the V/A path preserves the distinction
    on the same clips."""
    cells = {("mtg_" + a, "va_" + b): 0
             for a in ("calm", "sad") for b in ("calm", "sad")}
    for r in rows:
        m, v = r.get("mtg_mood"), r.get("va_mood")
        if m in ("calm", "sad") and v in ("calm", "sad"):
            cells[("mtg_" + m, "va_" + v)] += 1
    total = sum(cells.values())
    return {
        "n_clips_in_pair": total,
        "cells": {f"{k[0]}__{k[1]}": v for k, v in cells.items()},
        "agreement_rate": (
            sum(v for k, v in cells.items() if k[0][4:] == k[1][3:]) / total
            if total > 0 else 0.0
        ),
    }


def accuracy_vs_gt(rows: list[dict], pred_key: str) -> dict:
    """Accuracy of one pipeline against the manifest's mood_gt column."""
    n_gt = sum(1 for r in rows if r.get("mood_gt"))
    if n_gt == 0:
        return {"n": 0, "accuracy": None}
    correct = sum(1 for r in rows
                  if r.get("mood_gt") and r.get(pred_key) == r["mood_gt"])
    return {"n": n_gt, "accuracy": correct / n_gt}


def agreement_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["mtg_mood"] == r["va_mood"]) / len(rows)


# -------------------------------------------------------------- stub mode

def synthetic_rows(n: int = 60, seed: int = 0) -> list[dict]:
    """Generate fake per-clip predictions to validate the report's
    shape without trained heads or audio on disk."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a = float(rng.uniform(0, 1))
        v = float(rng.uniform(0, 1))
        mtg_probs = rng.dirichlet([1.0] * len(MOODS))
        va_probs = mood_probs_from_va(a, v)
        rows.append({
            "path": f"_stub_{i}.wav",
            "mood_gt": rng.choice(MOODS) if rng.uniform() < 0.5 else None,
            "mtg_mood": MOODS[int(np.argmax(mtg_probs))],
            "mtg_probs": {m: float(p) for m, p in zip(MOODS, mtg_probs)},
            "va_mood": max(va_probs, key=va_probs.get),
            "va_probs": va_probs,
            "arousal_unit": a,
            "valence_unit": v,
            "arousal_sam": 1.0 + 8.0 * a,
            "valence_sam": 1.0 + 8.0 * v,
            "va_quadrant": quadrant_label(a, v),
        })
    return rows


# ---------------------------------------------------------------- runner

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", default=None,
                    help="CSV with `path[,mood_class|,mood_gt][,...]`")
    ap.add_argument("--audio-root", default=None,
                    help="prefix for manifest paths")
    ap.add_argument("--panns-ckpt", default=None,
                    help="path to Cnn14_mAP=0.431.pth")
    ap.add_argument("--mood-head", default=None,
                    help="StyleHead checkpoint (mood pipeline)")
    ap.add_argument("--regression-head", default=None,
                    help="StyleRegressionHead checkpoint (DEAM pipeline)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-clips", type=int, default=None,
                    help="cap the clip count for a fast sweep")
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--include-aux-moods", action="store_true",
                    help="enable epic/dark via PROTOTYPES_WITH_AUX in "
                         "the V/A map; default is the 4 clean quadrants")
    ap.add_argument("--synthetic-stub", action="store_true",
                    help="skip audio + PANNs; fake per-clip predictions")
    ap.add_argument("--n-stub", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_stub = bool(args.synthetic_stub)
    if is_stub:
        per_clip = synthetic_rows(args.n_stub, args.seed)
    else:
        missing = [k for k in ("manifest", "audio_root", "panns_ckpt",
                               "mood_head", "regression_head")
                   if getattr(args, k.replace("-", "_")) is None]
        if missing:
            print(f"missing {missing} (or pass --synthetic-stub)",
                  file=sys.stderr)
            return 2
        clips = read_manifest(Path(args.manifest), Path(args.audio_root))
        if args.max_clips:
            clips = clips[: args.max_clips]
        if not clips:
            print("no clips matched (manifest + audio_root mismatch?)",
                  file=sys.stderr)
            return 2
        backbone = PannsBackbone(checkpoint_path=args.panns_ckpt, device="cpu")
        mood_head = StyleHead(emb_dim=EMBEDDING_DIM)
        mood_head.load_state_dict(load_head_state(Path(args.mood_head)))
        mood_head.eval()
        reg_head = StyleRegressionHead(emb_dim=EMBEDDING_DIM)
        reg_head.load_state_dict(load_head_state(Path(args.regression_head)))
        reg_head.eval()
        per_clip = []
        t0 = time.perf_counter()
        for i, clip in enumerate(clips):
            try:
                row = predict_clip(
                    clip.path, backbone, mood_head, reg_head,
                    window_sec=args.window_sec,
                    aux=args.include_aux_moods,
                )
            except Exception as e:
                print(f"  skip {clip.path}: {e}", file=sys.stderr)
                continue
            row["path"] = str(clip.path)
            row["mood_gt"] = clip.mood_gt
            per_clip.append(row)
            if (i + 1) % 20 == 0:
                avg = (time.perf_counter() - t0) / (i + 1)
                print(f"  {i+1}/{len(clips)}  avg {avg:.2f}s/clip",
                      flush=True)

    if not per_clip:
        print("no rows in per_clip; aborting", file=sys.stderr)
        return 2

    # ---- aggregates ----
    report = {
        "is_stub": is_stub,
        "include_aux_moods": bool(args.include_aux_moods),
        "n_clips": len(per_clip),
        "agreement_rate": agreement_rate(per_clip),
        "confusion_matrix_mtg_rows_va_cols": confusion_matrix(
            per_clip, "mtg_mood", "va_mood",
        ),
        "per_class_profile_by_mtg_mood": per_class_profile(
            per_clip, "mtg_mood",
        ),
        "per_class_profile_by_va_mood": per_class_profile(
            per_clip, "va_mood",
        ),
        "calm_sad_stability": calm_sad_stability(per_clip),
        "accuracy_vs_gt": {
            "mtg_mood": accuracy_vs_gt(per_clip, "mtg_mood"),
            "va_mood":  accuracy_vs_gt(per_clip, "va_mood"),
        },
        "class_distribution": {
            "mtg_mood": dict(Counter(r["mtg_mood"] for r in per_clip)),
            "va_mood":  dict(Counter(r["va_mood"]  for r in per_clip)),
            "va_quadrant": dict(Counter(r["va_quadrant"] for r in per_clip)),
            "mood_gt": dict(Counter(r["mood_gt"] for r in per_clip
                                    if r["mood_gt"])),
        },
        "config": vars(args),
        "interpretation_hints": [
            "calm_sad_stability: agreement_rate < 0.5 means the two "
            "pipelines disagree often on this pair — the MTG head's "
            "known sad→calm collapse may be eased by V/A.",
            "agreement_rate is a ceiling, not accuracy. A higher value "
            "just means the two pipelines tell the same story.",
            "per_class_profile_by_mtg_mood: if the MTG head's `calm` "
            "clips have V/A clustered in the calm quadrant, the V/A "
            "map should reproduce it. If they spread, MTG `calm` is "
            "leaking other classes.",
        ],
    }

    out_dir.joinpath("report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    with open(out_dir / "per_clip.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "mood_gt", "mtg_mood", "va_mood", "va_quadrant",
            "arousal_unit", "valence_unit", "arousal_sam", "valence_sam",
        ])
        for r in per_clip:
            w.writerow([
                r.get("path", ""), r.get("mood_gt") or "",
                r["mtg_mood"], r["va_mood"], r["va_quadrant"],
                f"{r['arousal_unit']:.4f}", f"{r['valence_unit']:.4f}",
                f"{r['arousal_sam']:.4f}", f"{r['valence_sam']:.4f}",
            ])

    print(f"\n[compare] {len(per_clip)} clips  (stub={is_stub})", flush=True)
    print(f"  agreement (MTG argmax == VA argmax): "
          f"{report['agreement_rate']:.3f}", flush=True)
    print(f"  calm/sad pair agreement: "
          f"{report['calm_sad_stability']['agreement_rate']:.3f} "
          f"(n={report['calm_sad_stability']['n_clips_in_pair']})", flush=True)
    if report["accuracy_vs_gt"]["mtg_mood"]["accuracy"] is not None:
        print(f"  acc vs gt (mtg): "
              f"{report['accuracy_vs_gt']['mtg_mood']['accuracy']:.3f} "
              f"(n={report['accuracy_vs_gt']['mtg_mood']['n']})", flush=True)
        print(f"  acc vs gt (va) : "
              f"{report['accuracy_vs_gt']['va_mood']['accuracy']:.3f}",
              flush=True)
    print(f"saved -> {out_dir/'report.json'}", flush=True)
    print(f"saved -> {out_dir/'per_clip.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
