"""tools/calibrate_affect.py — calibrate arousal/valence bucket boundaries.

We discovered (showcase v1.2 run, 2026-06-21) that the DEAM-trained arousal
head regresses to the mean: all four showcase clips landed in `mid`
(observed range 0.42–0.64) under the absolute 0.33/0.66 thresholds. The
honest fix is to derive bucket boundaries from the head's actual output
distribution on a held-out *reference* pool.

Strict no-leak rule: the calibration set must be DEAM only — never the
showcase clips. We use the DEAM **val** split (val_frac=0.15, test_frac=0.15,
seed=0 — same constants the trainer used in `experiments/train_arousal_tl.py`
so we reuse the cached embeddings under `data/style_emb_deam/`). Showcase
clips are held out as a downstream sanity check, not seen here.

What this script produces:

  * distribution stats (min/max/mean/std/median + 10/25/33.3/50/66.7/75/90 pct)
    on the raw 0..1 (post sam_to_unit) outputs for arousal and valence,
  * **tertile** bucket thresholds for arousal (low/mid/high at the 33.3/66.7
    percentiles of the calibration set),
  * the **median** of valence + arousal — the new neutral center used by
    `mood_from_va.py` (replacing the absolute 0.5 assumption),
  * `groovebot/style/affect_calibration.json` — a single source of truth
    read by `attributes.arousal_bucket()` and `mood_from_va.mood_probs_from_va()`.

Run with the same defaults as the trainer:

    python -m tools.calibrate_affect \
        --static-csv "data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv" \
        --static-csv "data/raw/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv" \
        --audio-root data/raw/deam \
        --cache-dir data/style_emb_deam \
        --arousal-head data/style_v3_arousal/style_head_arousal.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from groovebot.style.backbone import EMBEDDING_DIM
from groovebot.style.deam import (
    DeamRecord, read_static_annotations_many, sam_to_unit, song_disjoint_split,
)
from groovebot.style.model import StyleRegressionHead


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "groovebot" / "style" / "affect_calibration.json"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    pcts = [10.0, 25.0, 33.333333, 50.0, 66.666667, 75.0, 90.0]
    qs = np.quantile(values, [p / 100.0 for p in pcts])
    return {
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p33_3": float(qs[2]),
        "p50": float(qs[3]),
        "p66_7": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "quantiles": _quantiles(values),
    }


def _print_summary(label: str, st: dict) -> None:
    q = st["quantiles"]
    print(f"\n[{label}] n={st['n']}")
    print(f"  min={st['min']:.3f}  max={st['max']:.3f}  "
          f"mean={st['mean']:.3f}  std={st['std']:.3f}  "
          f"median={st['median']:.3f}")
    print(f"  quantiles  10%={q['p10']:.3f}  25%={q['p25']:.3f}  "
          f"33.3%={q['p33_3']:.3f}  50%={q['p50']:.3f}  "
          f"66.7%={q['p66_7']:.3f}  75%={q['p75']:.3f}  "
          f"90%={q['p90']:.3f}")


def _load_head(weights_path: Path) -> StyleRegressionHead:
    ckpt = torch.load(str(weights_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    hidden = ckpt.get("hidden", 256) if isinstance(ckpt, dict) else 256
    dropout = ckpt.get("dropout", 0.3) if isinstance(ckpt, dict) else 0.3
    emb_dim = ckpt.get("emb_dim", EMBEDDING_DIM) if isinstance(ckpt, dict) else EMBEDDING_DIM
    head = StyleRegressionHead(emb_dim=emb_dim, hidden=hidden, dropout=dropout)
    head.load_state_dict(sd)
    head.eval()
    return head


def _predict_on_split(
    head: StyleRegressionHead,
    records: list[DeamRecord],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run head on cached embeddings, return (arousal_unit, valence_unit, ids).

    Skips records whose .npy embedding is missing (logged), so the function
    is robust to a partial cache. Outputs are in 0..1 after `sam_to_unit`.
    """
    arousal_units: list[float] = []
    valence_units: list[float] = []
    kept_ids: list[int] = []
    missing = 0
    for rec in records:
        cp = cache_dir / f"{rec.song_id}.npy"
        if not cp.exists():
            missing += 1
            continue
        emb = np.load(str(cp)).astype(np.float32)
        with torch.no_grad():
            out = head(torch.from_numpy(emb).unsqueeze(0))
            a_raw = float(out["arousal"].item())
            v_raw = float(out["valence"].item())
        arousal_units.append(sam_to_unit(a_raw))
        valence_units.append(sam_to_unit(v_raw))
        kept_ids.append(rec.song_id)
    if missing:
        print(f"  [warn] {missing} records had no cached embedding; skipped.")
    return (
        np.asarray(arousal_units, dtype=np.float64),
        np.asarray(valence_units, dtype=np.float64),
        kept_ids,
    )


def _bucket_counts(values: np.ndarray, lo_thr: float, hi_thr: float) -> dict[str, int]:
    lo = int(np.sum(values < lo_thr))
    hi = int(np.sum(values >= hi_thr))
    mid = int(values.size - lo - hi)
    return {"low": lo, "mid": mid, "high": hi}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--static-csv", action="append", required=True,
                    help="DEAM static-annotation CSV. Pass twice for the "
                         "_1_2000 + _2000_2058 pair.")
    ap.add_argument("--audio-root", required=True,
                    help="DEAM audio root (used by the loader to find clips; "
                         "actual reads come from the .npy cache).")
    ap.add_argument("--cache-dir", required=True,
                    help="Cached PANNs embeddings dir "
                         "(default: data/style_emb_deam).")
    ap.add_argument("--arousal-head", required=True,
                    help="Trained StyleRegressionHead checkpoint "
                         "(default: data/style_v3_arousal/style_head_arousal.pt).")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Same as trainer default (do not change unless "
                         "the trainer changed).")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output JSON path (default: {DEFAULT_OUT}).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    csv_paths = [Path(p) for p in args.static_csv]
    audio_root = Path(args.audio_root)
    cache_dir = Path(args.cache_dir)
    head_path = Path(args.arousal_head)

    for label, p in [
        ("audio_root", audio_root), ("cache_dir", cache_dir),
        ("arousal_head", head_path),
    ]:
        if not p.exists():
            print(f"ERROR: {label} missing: {p}", file=sys.stderr)
            return 2

    print(f"[calibrate_affect] loading DEAM annotations from {len(csv_paths)} CSV(s)")
    records = read_static_annotations_many(csv_paths, audio_root)
    if not records:
        print("ERROR: no DEAM records loaded (csv + audio_root mismatch?)",
              file=sys.stderr)
        return 2
    print(f"[calibrate_affect] {len(records)} total DEAM records")

    # Same split params as trainer → same val songs.
    train_r, val_r, test_r = song_disjoint_split(
        records, args.val_frac, args.test_frac, args.seed,
    )
    print(f"[calibrate_affect] split: train={len(train_r)} val={len(val_r)} "
          f"test={len(test_r)} (val is the calibration pool)")

    pool_source = "DEAM val split (val_frac=%.2f, test_frac=%.2f, seed=%d)" % (
        args.val_frac, args.test_frac, args.seed,
    )
    pool_records = val_r
    if not pool_records:
        print("WARNING: val split is empty; falling back to a fixed 20% slice "
              "of train (seed=42).", file=sys.stderr)
        rng = np.random.default_rng(42)
        ids = sorted({r.song_id for r in train_r})
        rng.shuffle(ids)
        n_take = max(1, int(round(len(ids) * 0.20)))
        keep = set(ids[:n_take])
        pool_records = [r for r in train_r if r.song_id in keep]
        pool_source = "fallback: 20% of DEAM train (seed=42)"
        print(f"[calibrate_affect] fallback pool: {len(pool_records)} records",
              file=sys.stderr)

    print(f"[calibrate_affect] loading head from {head_path}")
    head = _load_head(head_path)
    print(f"[calibrate_affect] running head on {len(pool_records)} cached embeddings")
    arousal_unit, valence_unit, ids = _predict_on_split(
        head, pool_records, cache_dir,
    )
    if arousal_unit.size == 0:
        print("ERROR: no embeddings found for calibration pool.", file=sys.stderr)
        return 2

    a_stats = _stats(arousal_unit)
    v_stats = _stats(valence_unit)
    _print_summary("arousal (head 0..1)", a_stats)
    _print_summary("valence (head 0..1)", v_stats)

    # Tertile boundaries (low/mid/high) from the calibration pool.
    lo_thr = a_stats["quantiles"]["p33_3"]
    hi_thr = a_stats["quantiles"]["p66_7"]
    a_neutral = a_stats["median"]
    v_neutral = v_stats["median"]

    bcounts = _bucket_counts(arousal_unit, lo_thr, hi_thr)
    n_total = arousal_unit.size
    print("\n[calibrate_affect] arousal tertile boundaries (from calibration pool):")
    print(f"  low  = [min, {lo_thr:.3f})    count={bcounts['low']:>4d}  "
          f"({100.0 * bcounts['low'] / n_total:.1f}%)")
    print(f"  mid  = [{lo_thr:.3f}, {hi_thr:.3f})  count={bcounts['mid']:>4d}  "
          f"({100.0 * bcounts['mid'] / n_total:.1f}%)")
    print(f"  high = [{hi_thr:.3f}, max]    count={bcounts['high']:>4d}  "
          f"({100.0 * bcounts['high'] / n_total:.1f}%)")
    print(f"\n[calibrate_affect] V/A circumplex neutral center "
          f"(median): arousal={a_neutral:.3f}, valence={v_neutral:.3f}")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": pool_source,
        "head_checkpoint": str(head_path),
        "n_calibration": int(arousal_unit.size),
        "arousal": {
            "bucket_low_max": float(lo_thr),
            "bucket_high_min": float(hi_thr),
            **a_stats,
        },
        "valence": v_stats,
        "neutral_center": {
            "arousal": float(a_neutral),
            "valence": float(v_neutral),
        },
        "bucket_counts_on_calibration_pool": bcounts,
        "notes": (
            "Bucket boundaries are TERTILES of the arousal head's predictions "
            "on a held-out DEAM val split. The neutral center is the median of "
            "the head's predictions, used to re-center the V/A circumplex map "
            "in mood_from_va.py. Showcase clips are NEVER used here — they "
            "remain held-out for downstream qualitative checks."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[calibrate_affect] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
