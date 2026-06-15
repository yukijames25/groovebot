"""experiments/train_arousal_tl.py — DEAM -> arousal/valence head (v3).

Sibling of `train_mood_tl.py`, but the head is a regression head
(`StyleRegressionHead`) and the corpus is DEAM static annotations on
the 1..9 SAM scale.

What this script does:

  1. Reads DEAM static-annotation CSV(s) and finds matching audio
     under `--audio-root`.
  2. Song-disjoint train/val/test split (DEAM has no album/artist
     metadata in the static CSV, so the song id is the only honest
     split unit).
  3. Pre-computes PANNs CNN14 embeddings, cached to `<cache-dir>/
     <song_id>.npy`.
  4. Trains a `StyleRegressionHead` (arousal + valence) on MSE.
  5. Reports R^2, RMSE, Pearson r per target on val/test, and the
     **Pearson correlation between the existing v2 heuristic
     `estimate_arousal()` and DEAM ground truth** (the spec's
     mandatory cross-check — see SYSTEM_SPEC.md `arousal を DEAM で
     本物にする` section).

A `--synthetic-stub` mode skips real DEAM audio and generates a
class-of-target-conditional random dataset so the wiring goes
end-to-end without the dataset. Numbers are meaningless under
`--synthetic-stub`; the report carries `is_stub: true`.

CLI (real data):
    python -m experiments.train_arousal_tl \
        --static-csv data/raw/deam/static_annotations_averaged_songs_1_2000.csv \
        --static-csv data/raw/deam/static_annotations_averaged_songs_2000_2058.csv \
        --audio-root data/raw/deam/MEMD_audio \
        --panns-ckpt data/raw/Cnn14_mAP=0.431.pth \
        --cache-dir data/style_emb_deam \
        --out-dir data/style_v3_arousal \
        --epochs 50 --batch-size 64

CLI (stub):
    python -m experiments.train_arousal_tl \
        --synthetic-stub --out-dir data/style_v3_arousal_stub --epochs 10
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from groovebot.style.attributes import estimate_arousal
from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.deam import (
    DEAM_SAM_HI, DEAM_SAM_LO,
    DeamRecord, read_static_annotations_many, sam_to_unit, song_disjoint_split,
)
from groovebot.style.model import REGRESSION_TARGETS, StyleRegressionHead


# --------------------------------------------------------------------- metrics

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R^2, RMSE, Pearson r for a 1-d (y_true, y_pred) pair.

    R^2 is the coefficient of determination `1 - SS_res / SS_tot`
    (`SS_tot` is computed on the test sample, not the training mean,
    so the number is comparable to what scikit-learn reports). Pearson
    r is the linear correlation coefficient (sign-aware). NaNs in the
    numerator -> 0 (degenerate constant prediction).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"r2": 0.0, "rmse": 0.0, "pearson_r": 0.0, "n": 0}
    rmse = float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        if not np.isfinite(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    return {
        "r2": float(r2),
        "rmse": rmse,
        "pearson_r": pearson,
        "n": int(y_true.size),
    }


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0


# ----------------------------------------------------------- audio + embedding

def _center_crop(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    n = int(window_sec * sr)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    start = (len(audio) - n) // 2
    return audio[start: start + n]


def precompute_embeddings_deam(
    records: list[DeamRecord],
    backbone: PannsBackbone,
    cache_dir: Path,
    *,
    window_sec: float = 10.0,
    verbose: bool = True,
) -> dict[int, Path]:
    """Cache PANNs embeddings as `<cache-dir>/<song_id>.npy`. Reuses
    the cache across runs (delete the folder to invalidate)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    n_new = 0
    t0 = time.perf_counter()
    for rec in records:
        cache_path = cache_dir / f"{rec.song_id}.npy"
        out[rec.song_id] = cache_path
        if cache_path.exists():
            continue
        audio, sr = sf.read(str(rec.audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, window_sec)
        emb = backbone.embed(audio, sr)
        np.save(str(cache_path), emb)
        n_new += 1
        if verbose and (n_new % 50 == 0 or n_new == 1):
            avg = (time.perf_counter() - t0) / n_new
            print(f"  embedded {n_new} new clips, "
                  f"avg {avg:.2f}s/clip", flush=True)
    return out


def compute_heuristic_arousal_unit(
    records: list[DeamRecord],
    *,
    window_sec: float = 10.0,
    verbose: bool = True,
) -> dict[int, float]:
    """Run the existing v2 `estimate_arousal()` heuristic on the same
    center crop the head sees, returning a song_id -> 0..1 score.

    This is the cross-check the spec asks for: how well does the
    existing RMS x onset-density heuristic correlate with DEAM
    ground-truth arousal? The answer drives whether the heuristic stays
    or gets replaced by the learned head.
    """
    out: dict[int, float] = {}
    for i, rec in enumerate(records):
        audio, sr = sf.read(str(rec.audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, window_sec)
        out[rec.song_id] = estimate_arousal(audio, sr)
        if verbose and ((i + 1) % 100 == 0 or i == 0):
            print(f"  heuristic arousal on {i+1}/{len(records)}", flush=True)
    return out


def build_emb_dataset(
    records: list[DeamRecord],
    id_to_cache: dict[int, Path],
) -> TensorDataset:
    embs, ys = [], []
    for rec in records:
        cp = id_to_cache.get(rec.song_id)
        if cp is None or not cp.exists():
            continue
        embs.append(np.load(str(cp)).astype(np.float32))
        ys.append([rec.arousal, rec.valence])
    if not embs:
        return TensorDataset(
            torch.zeros(0, EMBEDDING_DIM),
            torch.zeros(0, 2, dtype=torch.float32),
        )
    X = torch.from_numpy(np.stack(embs, axis=0))
    Y = torch.tensor(ys, dtype=torch.float32)
    return TensorDataset(X, Y)


# ---------------------------------------------------------------- stub mode

def synthetic_records(
    n_songs: int = 240, seed: int = 0,
) -> tuple[list[DeamRecord], dict[int, np.ndarray]]:
    """Generate a song-disjoint DEAM-like dataset for stub training.

    Arousal/valence drawn from a uniform on the SAM scale (1..9).
    Embeddings = `arousal_direction * arousal + valence_direction *
    valence + noise` so the head can actually learn.
    """
    rng = np.random.default_rng(seed)
    arousal_dir = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.1
    valence_dir = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.1
    records: list[DeamRecord] = []
    fake: dict[int, np.ndarray] = {}
    for sid in range(n_songs):
        a = float(rng.uniform(DEAM_SAM_LO, DEAM_SAM_HI))
        v = float(rng.uniform(DEAM_SAM_LO, DEAM_SAM_HI))
        records.append(DeamRecord(
            audio_path=Path(f"_stub_{sid}.wav"),
            song_id=sid, arousal=a, valence=v,
        ))
        emb = (
            a * arousal_dir
            + v * valence_dir
            + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.5
        )
        fake[sid] = emb.astype(np.float32)
    return records, fake


def stub_dataset(
    records: list[DeamRecord], fake: dict[int, np.ndarray],
) -> TensorDataset:
    X = np.stack([fake[r.song_id] for r in records], axis=0)
    Y = np.array(
        [[r.arousal, r.valence] for r in records], dtype=np.float32,
    )
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))


# ------------------------------------------------------- training inner loop

@dataclass
class EpochMetrics:
    loss: float
    rmse_arousal: float
    rmse_valence: float


def _forward_targets(head: StyleRegressionHead, x: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor,
]:
    out = head(x)
    return out["arousal"], out["valence"]


def train_one_epoch(
    head: StyleRegressionHead, loader: DataLoader,
    opt: torch.optim.Optimizer, device: torch.device,
) -> EpochMetrics:
    head.train()
    losses, sq_a, sq_v, n = [], 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred_a, pred_v = _forward_targets(head, x)
        loss = F.mse_loss(pred_a, y[:, 0]) + F.mse_loss(pred_v, y[:, 1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        sq_a += float(((pred_a - y[:, 0]) ** 2).sum().item())
        sq_v += float(((pred_v - y[:, 1]) ** 2).sum().item())
        n += int(y.shape[0])
    return EpochMetrics(
        loss=float(np.mean(losses) if losses else 0.0),
        rmse_arousal=float(math.sqrt(sq_a / max(n, 1))),
        rmse_valence=float(math.sqrt(sq_v / max(n, 1))),
    )


@torch.no_grad()
def evaluate(
    head: StyleRegressionHead, loader: DataLoader, device: torch.device,
) -> tuple[EpochMetrics, dict[str, np.ndarray]]:
    head.eval()
    losses, sq_a, sq_v, n = [], 0.0, 0.0, 0
    pa_all, pv_all, ya_all, yv_all = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred_a, pred_v = _forward_targets(head, x)
        losses.append(float((F.mse_loss(pred_a, y[:, 0]) + F.mse_loss(pred_v, y[:, 1])).item()))
        sq_a += float(((pred_a - y[:, 0]) ** 2).sum().item())
        sq_v += float(((pred_v - y[:, 1]) ** 2).sum().item())
        n += int(y.shape[0])
        pa_all.append(pred_a.cpu().numpy())
        pv_all.append(pred_v.cpu().numpy())
        ya_all.append(y[:, 0].cpu().numpy())
        yv_all.append(y[:, 1].cpu().numpy())
    return EpochMetrics(
        loss=float(np.mean(losses) if losses else 0.0),
        rmse_arousal=float(math.sqrt(sq_a / max(n, 1))),
        rmse_valence=float(math.sqrt(sq_v / max(n, 1))),
    ), {
        "arousal_true": np.concatenate(ya_all) if ya_all else np.zeros(0),
        "arousal_pred": np.concatenate(pa_all) if pa_all else np.zeros(0),
        "valence_true": np.concatenate(yv_all) if yv_all else np.zeros(0),
        "valence_pred": np.concatenate(pv_all) if pv_all else np.zeros(0),
    }


# ------------------------------------------------------------------ runner

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--static-csv", action="append", default=None,
                    help="DEAM static-annotation CSV. Pass multiple times "
                         "for the _1_2000 + _2000_2058 pair.")
    ap.add_argument("--audio-root", default=None,
                    help="dir containing DEAM audio (MEMD_audio/<song_id>.mp3 "
                         "or flat <song_id>.mp3).")
    ap.add_argument("--panns-ckpt", default=None,
                    help="path to Cnn14_mAP=0.431.pth (required when not "
                         "using --synthetic-stub)")
    ap.add_argument("--cache-dir", default=None,
                    help="dir for .npy embedding cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synthetic-stub", action="store_true",
                    help="skip real DEAM; generate a target-conditional random "
                         "dataset to verify the training loop. Numbers are "
                         "meaningless.")
    ap.add_argument("--n-stub-songs", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--skip-heuristic-check", action="store_true",
                    help="skip the heuristic-vs-truth correlation step "
                         "(needs to run pyin/onset_strength on every test "
                         "clip; ~5s each).")
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
    if not is_stub:
        if not (args.static_csv and args.audio_root and args.panns_ckpt
                and args.cache_dir):
            print("missing --static-csv / --audio-root / --panns-ckpt / "
                  "--cache-dir (or pass --synthetic-stub)",
                  file=sys.stderr)
            return 2
        csv_paths = [Path(p) for p in args.static_csv]
        records = read_static_annotations_many(csv_paths, Path(args.audio_root))
        if not records:
            print("empty DEAM record set (csv + audio mismatch?)",
                  file=sys.stderr)
            return 2
        backbone = PannsBackbone(checkpoint_path=args.panns_ckpt, device="cpu")
        id_to_cache = precompute_embeddings_deam(
            records, backbone, Path(args.cache_dir),
            window_sec=args.window_sec,
        )
        train_r, val_r, test_r = song_disjoint_split(
            records, args.val_frac, args.test_frac, args.seed,
        )
        train_ds = build_emb_dataset(train_r, id_to_cache)
        val_ds = build_emb_dataset(val_r, id_to_cache)
        test_ds = build_emb_dataset(test_r, id_to_cache)
    else:
        records, fake = synthetic_records(args.n_stub_songs, args.seed)
        train_r, val_r, test_r = song_disjoint_split(
            records, args.val_frac, args.test_frac, args.seed,
        )
        train_ds = stub_dataset(train_r, fake)
        val_ds = stub_dataset(val_r, fake)
        test_ds = stub_dataset(test_r, fake)

    print(f"[head ] train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  (stub={is_stub})", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    head = StyleRegressionHead(emb_dim=EMBEDDING_DIM, hidden=args.hidden,
                               dropout=args.dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch(head, train_loader, opt, device)
        vl, _ = evaluate(head, val_loader, device)
        dt = time.perf_counter() - t0
        history.append({
            "epoch": epoch,
            "train": {"loss": tr.loss, "rmse_arousal": tr.rmse_arousal,
                      "rmse_valence": tr.rmse_valence},
            "val":   {"loss": vl.loss, "rmse_arousal": vl.rmse_arousal,
                      "rmse_valence": vl.rmse_valence},
            "dt_sec": round(dt, 2),
        })
        print(
            f"epoch {epoch:02d}  loss={tr.loss:.3f}  "
            f"val_a_rmse={vl.rmse_arousal:.3f}  "
            f"val_v_rmse={vl.rmse_valence:.3f}  ({dt:.1f}s)",
            flush=True,
        )
        if vl.loss < best_val:
            best_val = vl.loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if (args.early_stopping_patience > 0
                    and no_improve >= args.early_stopping_patience):
                print(f"early stopping at epoch {epoch} "
                      f"(no val improvement for {no_improve} epochs)",
                      flush=True)
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    test_em, test_preds = evaluate(head, test_loader, device)
    val_em, val_preds = evaluate(head, val_loader, device)

    targets_report = {}
    for tname in REGRESSION_TARGETS:
        targets_report[tname] = {
            "val": regression_metrics(val_preds[f"{tname}_true"],
                                      val_preds[f"{tname}_pred"]),
            "test": regression_metrics(test_preds[f"{tname}_true"],
                                       test_preds[f"{tname}_pred"]),
        }

    # Spec-mandated: heuristic vs DEAM correlation. Run on the test
    # split only (not train+val) so the answer is on a held-out sample
    # and runtime is bounded. Stub mode synthesises a heuristic value
    # from the (already-fake) embedding norm so the report shape is
    # identical between stub and real runs.
    heuristic_report = None
    if not args.skip_heuristic_check:
        if is_stub:
            rng = np.random.default_rng(args.seed)
            heuristic_unit = {r.song_id: float(rng.uniform(0, 1)) for r in test_r}
        else:
            print(f"[heur ] running estimate_arousal() on {len(test_r)} test clips",
                  flush=True)
            heuristic_unit = compute_heuristic_arousal_unit(
                test_r, window_sec=args.window_sec,
            )
        rec_by_id = {r.song_id: r for r in test_r}
        ids = sorted(set(heuristic_unit) & set(rec_by_id))
        if ids:
            h_unit = np.array([heuristic_unit[i] for i in ids])
            truth_sam = np.array([rec_by_id[i].arousal for i in ids])
            truth_unit = np.array([sam_to_unit(rec_by_id[i].arousal) for i in ids])
            heuristic_report = {
                "n": len(ids),
                "pearson_r_unit_vs_truth_sam": pearson_r(h_unit, truth_sam),
                "pearson_r_unit_vs_truth_unit": pearson_r(h_unit, truth_unit),
                "rmse_unit_vs_truth_unit": float(math.sqrt(
                    float(np.mean((h_unit - truth_unit) ** 2))
                )),
                "heuristic_mean_unit": float(np.mean(h_unit)),
                "heuristic_std_unit":  float(np.std(h_unit)),
                "truth_arousal_mean_sam": float(np.mean(truth_sam)),
                "truth_arousal_std_sam":  float(np.std(truth_sam)),
                "samples": [
                    {"song_id": int(i),
                     "heuristic_unit": float(heuristic_unit[i]),
                     "truth_sam": float(rec_by_id[i].arousal)}
                    for i in ids
                ],
            }

    ckpt_path = out_dir / "style_head_arousal.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "emb_dim": EMBEDDING_DIM,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "head_target": "arousal_valence",
        "targets": list(REGRESSION_TARGETS),
        "scale": {"lo": DEAM_SAM_LO, "hi": DEAM_SAM_HI},
        "is_stub": is_stub,
    }, ckpt_path)

    final_report = {
        "config": vars(args),
        "is_stub": is_stub,
        "n_total_records": len(train_ds) + len(val_ds) + len(test_ds),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 4),
        "targets": targets_report,
        "heuristic_arousal_vs_deam": heuristic_report,
        "caveats": [
            ("This is a STUB run — numbers are noise from a target-"
             "conditional embedding generator, not real DEAM audio.")
            if is_stub else
            ("DEAM annotations are crowdsourced (5+ annotators per song); "
             "static averages smooth that noise but inter-rater "
             "disagreement caps R^2 from above. Literature targets are "
             "around arousal R^2 ~= 0.6 / valence R^2 ~= 0.4 — interpret "
             "deltas accordingly."),
            "PANNs CNN14 was pretrained on AudioSet which has weak "
            "overlap with DEAM. R^2 reported here is for the frozen "
            "embedding + tiny MLP; fine-tuning the backbone could go "
            "higher but is out of scope.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nbest val loss: {best_val:.3f} @ ep {best_epoch}", flush=True)
    print(f"test  arousal: R^2={targets_report['arousal']['test']['r2']:.3f}  "
          f"RMSE={targets_report['arousal']['test']['rmse']:.3f}  "
          f"r={targets_report['arousal']['test']['pearson_r']:.3f}",
          flush=True)
    print(f"test  valence: R^2={targets_report['valence']['test']['r2']:.3f}  "
          f"RMSE={targets_report['valence']['test']['rmse']:.3f}  "
          f"r={targets_report['valence']['test']['pearson_r']:.3f}",
          flush=True)
    if heuristic_report:
        print(f"heur  vs DEAM truth (test, n={heuristic_report['n']}): "
              f"r={heuristic_report['pearson_r_unit_vs_truth_sam']:.3f}",
              flush=True)
    print(f"saved head -> {ckpt_path}", flush=True)
    print(f"saved report -> {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
