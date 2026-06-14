"""experiments/train_genre_tl.py — transfer-learning genre head (v3).

Replaces the v2 from-scratch StyleCNN training with a two-stage pipeline:

  1. **Pre-compute** PANNs CNN14 embeddings for every GTZAN clip in the
     fault-filtered split (jongpillee, see `tools/gtzan_split.py`).
     Embeddings are cached to `--cache-dir/<stem>.npy` (gitignored).
     One pass takes ~60 min on CPU; subsequent runs read .npy and are
     fast.

  2. **Train** a small `StyleHead` MLP on the cached embeddings. CPU,
     under one minute per epoch.

Reports the same fields v2 did (`best_val_genre_acc`, `test`,
`train_val_gap_at_best`, `confusion_matrix`, representative
`GrooveStyle` per genre) so the v3 number lands directly alongside the
v2 fault-split number in the README table.

Mood head and naive split are **not** the focus here; v3 mood is wired
in `experiments/train_mood_tl.py`, and the naive split is intentionally
omitted because the v2 result already showed the leakage delta.

CLI:
    python -m experiments.train_genre_tl \\
        --gtzan-root data/raw/gtzan_full/Data/genres_original \\
        --splits-dir data/raw/gtzan_splits \\
        --panns-ckpt data/raw/Cnn14_mAP=0.431.pth \\
        --cache-dir data/style_emb \\
        --out-dir data/style_v3_fault \\
        --epochs 40 --batch-size 32
"""
from __future__ import annotations
import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from groovebot.style.backbone import EMBEDDING_DIM, PannsBackbone
from groovebot.style.model import GENRES, MOODS, StyleHead
from groovebot.style.select import GrooveStyleSelector
from tools.gtzan_split import GTZANClip, SplitReport, build_split


def precompute_embeddings(
    clips: Iterable[GTZANClip],
    backbone: PannsBackbone,
    cache_dir: Path,
    *,
    window_sec: float = 10.0,
    verbose: bool = True,
) -> dict[str, Path]:
    """Embed every clip and write a `<stem>.npy` per file.

    Returns a `{rel: cache_path}` map so downstream code can build a
    tensor dataset without re-walking the disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    n_new = 0
    t0 = time.perf_counter()
    for i, clip in enumerate(clips):
        cache_path = cache_dir / (clip.path.stem + ".npy")
        out[clip.rel] = cache_path
        if cache_path.exists():
            continue
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, window_sec)
        emb = backbone.embed(audio, sr)
        np.save(str(cache_path), emb)
        n_new += 1
        if verbose and (n_new % 25 == 0 or n_new == 1):
            avg = (time.perf_counter() - t0) / n_new
            print(f"  embedded {n_new} new clips, "
                  f"avg {avg:.2f}s/clip", flush=True)
    if verbose:
        total = time.perf_counter() - t0
        print(f"  precompute done: {n_new} new, {len(out) - n_new} cached, "
              f"total {total:.1f}s", flush=True)
    return out


def _center_crop(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    n = int(window_sec * sr)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    start = (len(audio) - n) // 2
    return audio[start: start + n]


def build_emb_dataset(
    clips: list[GTZANClip],
    rel_to_cache: dict[str, Path],
) -> tuple[TensorDataset, list[GTZANClip]]:
    """Load all cached embeddings into one tensor pair.

    Files missing from the cache are silently dropped from the returned
    clip list — caller can compare lengths to detect partial precompute.
    """
    embs = []
    labels = []
    kept = []
    for c in clips:
        p = rel_to_cache.get(c.rel)
        if p is None or not p.exists():
            continue
        embs.append(np.load(str(p)).astype(np.float32))
        labels.append(GENRES.index(c.genre))
        kept.append(c)
    if not embs:
        # Empty TensorDataset is fine; caller logs the issue.
        return (
            TensorDataset(torch.zeros(0, EMBEDDING_DIM), torch.zeros(0, dtype=torch.long)),
            kept,
        )
    X = torch.from_numpy(np.stack(embs, axis=0))
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y), kept


@dataclass
class EpochMetrics:
    loss: float
    acc: float


def train_one_epoch(
    head: StyleHead, loader: DataLoader, opt: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    head.train()
    losses, correct, n = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(x)["genre"]
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
def evaluate(
    head: StyleHead, loader: DataLoader, device: torch.device,
) -> tuple[EpochMetrics, np.ndarray]:
    head.eval()
    losses, correct, n = [], 0, 0
    confusion = np.zeros((len(GENRES), len(GENRES)), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(x)["genre"]
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


def representative_labels(
    selector: GrooveStyleSelector, clips: list[GTZANClip],
    n_per_genre: int = 1,
) -> list[dict]:
    rows: list[dict] = []
    seen: dict[str, int] = {g: 0 for g in GENRES}
    for clip in clips:
        if seen[clip.genre] >= n_per_genre:
            continue
        seen[clip.genre] += 1
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, 10.0)
        style = selector.select(audio, sr)
        rows.append({
            "path": str(clip.path),
            "true_genre": clip.genre,
            "predicted_genre": style.genre,
            "move": style.move,
            "intensity": round(style.intensity, 3),
            "tempo_bpm": round(style.tempo_bpm, 1),
            "arousal": round(style.arousal, 3),
            "arousal_bucket": style.arousal_bucket,
            "text": style.as_text(),
        })
    return rows


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    return (cm.astype(np.float64) / safe).round(3)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gtzan-root", required=True)
    ap.add_argument("--splits-dir", required=True,
                    help="jongpillee fault-filtered splits dir")
    ap.add_argument("--panns-ckpt", required=True,
                    help="path to Cnn14_mAP=0.431.pth")
    ap.add_argument("--cache-dir", required=True,
                    help="dir for .npy embedding cache (per-stem)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-mode", choices=("naive", "fault"), default="fault")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--window-sec", type=float, default=10.0,
                    help="center-crop window fed to PANNs (PANNs was "
                         "trained on AudioSet 10 s clips)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-representative", action="store_true",
                    help="skip representative GrooveStyle rendering at the "
                         "end (saves a few minutes since it re-runs the "
                         "PANNs backbone per genre).")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[split] mode={args.split_mode}", flush=True)
    report: SplitReport = build_split(
        Path(args.gtzan_root),
        mode=args.split_mode,
        splits_dir=Path(args.splits_dir),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"[split] counts={report.counts()}", flush=True)

    print(f"[embed] backbone={args.panns_ckpt}", flush=True)
    backbone = PannsBackbone(checkpoint_path=args.panns_ckpt, device="cpu")
    all_clips = report.train + report.val + report.test
    rel_to_cache = precompute_embeddings(
        all_clips, backbone, cache_dir, window_sec=args.window_sec,
    )

    train_ds, _ = build_emb_dataset(report.train, rel_to_cache)
    val_ds, _ = build_emb_dataset(report.val, rel_to_cache)
    test_ds, _ = build_emb_dataset(report.test, rel_to_cache)
    print(f"[head ] train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}", flush=True)

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
            "train": {"loss": tr.loss, "acc_genre": tr.acc},
            "val":   {"loss": vl.loss, "acc_genre": vl.acc},
            "dt_sec": round(dt, 2),
        })
        print(
            f"epoch {epoch:02d}  loss={tr.loss:.3f}  "
            f"train_g={tr.acc:.3f}  val_g={vl.acc:.3f}  ({dt:.1f}s)",
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

    ckpt_path = out_dir / "style_head.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "emb_dim": EMBEDDING_DIM,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "split_mode": args.split_mode,
        "panns_ckpt": str(args.panns_ckpt),
    }, ckpt_path)

    examples = []
    if not args.no_representative:
        print("[examples] rendering representative GrooveStyles "
              "(re-runs PANNs per genre, ~1 min)", flush=True)
        selector = GrooveStyleSelector(backbone=backbone, head=head)
        examples = representative_labels(selector, report.test or report.val)

    final_report = {
        "config": vars(args),
        "split_mode": args.split_mode,
        "split_counts": report.counts(),
        "split_sources": report.sources,
        "skipped": report.skipped,
        "history": history,
        "best_val_genre_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "train_val_gap_at_best": round(best_train_at_best_val - best_val, 4),
        "test": {
            "loss": round(test_m.loss, 4),
            "acc_genre": round(test_m.acc, 4),
        },
        "confusion_matrix": {
            "labels": list(GENRES),
            "counts": test_conf.tolist(),
            "row_normalised": _row_normalise(test_conf).tolist(),
        },
        "representative_groove_styles": examples,
        "backbone": {
            "name": "PANNs CNN14",
            "checkpoint": str(args.panns_ckpt),
            "embedding_dim": EMBEDDING_DIM,
            "sample_rate": 32000,
            "source": (
                "Kong et al. 2020, "
                "https://zenodo.org/record/3987831, "
                "https://github.com/qiuqiangkong/audioset_tagging_cnn"
            ),
        },
        "caveats": [
            "PANNs CNN14 was pretrained on AudioSet, which contains music. "
            "Transfer accuracy here is a fair feature-quality measure, but "
            "is NOT a clean self-supervised benchmark on raw audio.",
            "GTZAN published faults make every number slightly optimistic, "
            "even under the fault-filtered split (Sturm 2013).",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nbest val: {best_val:.3f} @ ep {best_epoch}", flush=True)
    print(f"test acc: {test_m.acc:.3f}  "
          f"(train-val gap at best: "
          f"{best_train_at_best_val - best_val:+.3f})", flush=True)
    print(f"saved head -> {ckpt_path}", flush=True)
    print(f"saved report -> {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
