"""experiments/train_mood_tl.py — MTG-Jamendo → mood head (v3).

Same shape as `train_genre_tl.py` but for the mood head:

  1. Read the MTG manifest produced by
     `tools/ingest_mtg_moodtheme.py` (path, mtg_track_id, artist_id,
     mood_class, raw_tags).
  2. Group by artist for the train/val/test split (artist-non-overlap;
     MTG ships artist IDs for every track, so this is exact rather
     than best-effort).
  3. Pre-compute PANNs CNN14 embeddings for every clip, cached to
     `--cache-dir/<basename>.npy`.
  4. Train a `StyleHead` mood head on the cached embeddings.

When real MTG audio is not on disk (you have not run the upstream
`download.py` yet), pass `--synthetic-stub` and the script generates
a tiny labelled set with random embeddings so the wiring (loss,
artist split, confusion matrix, report.json schema) goes through end-
to-end. The reported numbers under the stub are meaningless and the
report carries an explicit `is_stub: true` flag.

CLI (real data):
    python -m experiments.train_mood_tl \\
        --manifest data/mtg_moodtheme_manifest.csv \\
        --audio-root data/raw/mtg_moodtheme \\
        --panns-ckpt data/raw/Cnn14_mAP=0.431.pth \\
        --cache-dir data/style_emb_mtg \\
        --out-dir data/style_v3_mood \\
        --epochs 40 --batch-size 64

CLI (stub):
    python -m experiments.train_mood_tl \\
        --synthetic-stub --out-dir data/style_v3_mood_stub --epochs 5
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
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

from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.model import MOODS, StyleHead


@dataclass
class MtgRecord:
    audio_path: Path
    artist_id: str
    mood_class: str


def read_manifest(manifest_csv: Path, audio_root: Path) -> list[MtgRecord]:
    """Read the ingest_mtg_moodtheme manifest, prepending audio_root to
    relative paths. Rows whose audio file is missing are silently
    dropped here (the ingest step's `--ignore-missing` may have kept
    them); they will show up as 0 records in the report counts."""
    out: list[MtgRecord] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            apath = audio_root / r["path"]
            if not apath.exists():
                continue
            out.append(MtgRecord(
                audio_path=apath,
                artist_id=r["artist_id"],
                mood_class=r["mood_class"],
            ))
    return out


def artist_disjoint_split(
    records: list[MtgRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[MtgRecord], list[MtgRecord], list[MtgRecord]]:
    """Group records by `artist_id`, shuffle the groups, then drop them
    into train / val / test buckets in order. No artist crosses the
    boundary. The per-bucket class balance is whatever the artist
    distribution allows (MTG is moderately imbalanced; the trainer
    does not re-weight)."""
    rng = random.Random(seed)
    by_artist: dict[str, list[MtgRecord]] = {}
    for r in records:
        by_artist.setdefault(r.artist_id, []).append(r)
    artist_ids = list(by_artist.keys())
    rng.shuffle(artist_ids)
    n_total = len(records)
    n_test_target = int(round(n_total * test_frac))
    n_val_target = int(round(n_total * val_frac))
    test_recs: list[MtgRecord] = []
    val_recs: list[MtgRecord] = []
    train_recs: list[MtgRecord] = []
    for aid in artist_ids:
        group = by_artist[aid]
        if len(test_recs) + len(group) <= n_test_target:
            test_recs.extend(group)
        elif len(val_recs) + len(group) <= n_val_target:
            val_recs.extend(group)
        else:
            train_recs.extend(group)
    return train_recs, val_recs, test_recs


def precompute_embeddings_mtg(
    records: list[MtgRecord],
    backbone: PannsBackbone,
    cache_dir: Path,
    *,
    window_sec: float = 10.0,
    verbose: bool = True,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    n_new = 0
    t0 = time.perf_counter()
    for rec in records:
        stem = rec.audio_path.stem
        cache_path = cache_dir / (stem + ".npy")
        out[str(rec.audio_path)] = cache_path
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


def _center_crop(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    n = int(window_sec * sr)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    start = (len(audio) - n) // 2
    return audio[start: start + n]


def build_emb_dataset(
    records: list[MtgRecord],
    path_to_cache: dict[str, Path],
) -> TensorDataset:
    embs = []
    labels = []
    for rec in records:
        cp = path_to_cache.get(str(rec.audio_path))
        if cp is None or not cp.exists():
            continue
        embs.append(np.load(str(cp)).astype(np.float32))
        labels.append(MOODS.index(rec.mood_class))
    if not embs:
        return TensorDataset(
            torch.zeros(0, EMBEDDING_DIM), torch.zeros(0, dtype=torch.long),
        )
    X = torch.from_numpy(np.stack(embs, axis=0))
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y)


def synthetic_records(n_per_class: int = 60, seed: int = 0) -> tuple[
    list[MtgRecord], dict[str, np.ndarray]
]:
    """Generate a class-balanced random dataset for stub training.

    Returns (records, fake_embeddings[path]). The training loop will
    skip the PANNs precompute path and use these pre-baked embeddings
    directly. Each "artist" gets a small group of clips (5) so the
    artist-disjoint split still has something meaningful to group on.
    """
    rng = np.random.default_rng(seed)
    records: list[MtgRecord] = []
    fake: dict[str, np.ndarray] = {}
    artist_counter = 0
    for mood in MOODS:
        # Class-conditional mean shift so the head can learn *something*
        # under the stub — verifies the loss decreases when the signal
        # is there.
        mean_shift = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.5
        for i in range(n_per_class):
            artist_id = f"stub_artist_{artist_counter // 5:04d}"
            artist_counter += 1
            stub_path = Path(f"_stub_{mood}_{i:03d}.wav")
            records.append(MtgRecord(
                audio_path=stub_path,
                artist_id=artist_id,
                mood_class=mood,
            ))
            emb = (mean_shift +
                   rng.standard_normal(EMBEDDING_DIM).astype(np.float32))
            fake[str(stub_path)] = emb
    return records, fake


def stub_dataset(records: list[MtgRecord], fake: dict[str, np.ndarray]) -> TensorDataset:
    embs = np.stack([fake[str(r.audio_path)] for r in records], axis=0)
    labels = np.array([MOODS.index(r.mood_class) for r in records], dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(embs), torch.from_numpy(labels),
    )


@dataclass
class EpochMetrics:
    loss: float
    acc: float


def train_one_epoch(head: StyleHead, loader: DataLoader,
                    opt: torch.optim.Optimizer, device: torch.device) -> EpochMetrics:
    head.train()
    losses, correct, n = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(x)["mood"]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        correct += int((logits.argmax(-1) == y).sum().item())
        n += int(y.numel())
    return EpochMetrics(
        loss=float(np.mean(losses) if losses else 0.0),
        acc=correct / max(n, 1),
    )


@torch.no_grad()
def evaluate(head: StyleHead, loader: DataLoader,
             device: torch.device) -> tuple[EpochMetrics, np.ndarray]:
    head.eval()
    losses, correct, n = [], 0, 0
    confusion = np.zeros((len(MOODS), len(MOODS)), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(x)["mood"]
        losses.append(float(F.cross_entropy(logits, y).item()))
        pred = logits.argmax(-1)
        correct += int((pred == y).sum().item())
        n += int(y.numel())
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[int(t), int(p)] += 1
    return EpochMetrics(
        loss=float(np.mean(losses) if losses else 0.0),
        acc=correct / max(n, 1),
    ), confusion


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    return (cm.astype(np.float64) / safe).round(3)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", default=None,
                    help="CSV produced by tools/ingest_mtg_moodtheme.py")
    ap.add_argument("--audio-root", default=None,
                    help="dir prepended to manifest path entries")
    ap.add_argument("--panns-ckpt", default=None,
                    help="path to Cnn14_mAP=0.431.pth (required when not "
                         "using --synthetic-stub)")
    ap.add_argument("--cache-dir", default=None,
                    help="dir for .npy embedding cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synthetic-stub", action="store_true",
                    help="skip real MTG; generate a class-conditional "
                         "random dataset to verify the training loop wires "
                         "end-to-end. Numbers are meaningless.")
    ap.add_argument("--n-per-class", type=int, default=60,
                    help="(stub only) clips per mood class")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
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
        if not (args.manifest and args.audio_root and args.panns_ckpt
                and args.cache_dir):
            print("missing --manifest / --audio-root / --panns-ckpt / "
                  "--cache-dir (or pass --synthetic-stub)",
                  file=sys.stderr)
            return 2
        records = read_manifest(Path(args.manifest), Path(args.audio_root))
        if not records:
            print(f"empty manifest: {args.manifest}", file=sys.stderr)
            return 2
        backbone = PannsBackbone(checkpoint_path=args.panns_ckpt, device="cpu")
        path_to_cache = precompute_embeddings_mtg(
            records, backbone, Path(args.cache_dir),
            window_sec=args.window_sec,
        )
        train_r, val_r, test_r = artist_disjoint_split(
            records, args.val_frac, args.test_frac, args.seed,
        )
        train_ds = build_emb_dataset(train_r, path_to_cache)
        val_ds = build_emb_dataset(val_r, path_to_cache)
        test_ds = build_emb_dataset(test_r, path_to_cache)
    else:
        records, fake = synthetic_records(args.n_per_class, args.seed)
        train_r, val_r, test_r = artist_disjoint_split(
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
    head = StyleHead(emb_dim=EMBEDDING_DIM, hidden=args.hidden,
                     dropout=args.dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    history = []
    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_train_at_best_val = 0.0
    no_improve = 0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch(head, train_loader, opt, device)
        vl, _ = evaluate(head, val_loader, device)
        dt = time.perf_counter() - t0
        history.append({
            "epoch": epoch,
            "train": {"loss": tr.loss, "acc_mood": tr.acc},
            "val":   {"loss": vl.loss, "acc_mood": vl.acc},
            "dt_sec": round(dt, 2),
        })
        print(
            f"epoch {epoch:02d}  loss={tr.loss:.3f}  "
            f"train_m={tr.acc:.3f}  val_m={vl.acc:.3f}  ({dt:.1f}s)",
            flush=True,
        )
        if vl.acc > best_val:
            best_val = vl.acc
            best_epoch = epoch
            best_train_at_best_val = tr.acc
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
    test_m, test_conf = evaluate(head, test_loader, device)

    ckpt_path = out_dir / "style_head_mood.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "emb_dim": EMBEDDING_DIM,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "head_target": "mood",
        "is_stub": is_stub,
    }, ckpt_path)

    final_report = {
        "config": vars(args),
        "is_stub": is_stub,
        "n_total_records": len(train_ds) + len(val_ds) + len(test_ds),
        "history": history,
        "best_val_mood_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "train_val_gap_at_best": round(best_train_at_best_val - best_val, 4),
        "test": {
            "loss": round(test_m.loss, 4),
            "acc_mood": round(test_m.acc, 4),
        },
        "confusion_matrix": {
            "labels": list(MOODS),
            "counts": test_conf.tolist(),
            "row_normalised": _row_normalise(test_conf).tolist(),
        },
        "caveats": [
            ("This is a STUB run — numbers are noise from a class-"
             "conditional Gaussian generator, not real MTG audio.")
            if is_stub else
            ("Mood label noise: the MTG-Jamendo moodtheme tags are "
             "crowdsourced and noisy. Several tags map ambiguously "
             "(see groovebot/style/mood_mapping.py); the conflict-rule "
             "and drop list both affect the resulting class balance."),
            "PANNs CNN14 was pretrained on AudioSet which contains some "
            "of the same Jamendo distribution. The transfer-learning "
            "headroom here is a fair embedding-quality measure, not a "
            "from-scratch baseline.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nbest val: {best_val:.3f} @ ep {best_epoch}", flush=True)
    print(f"test acc: {test_m.acc:.3f}", flush=True)
    print(f"saved head -> {ckpt_path}", flush=True)
    print(f"saved report -> {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
