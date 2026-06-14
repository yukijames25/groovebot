"""experiments/train_style.py — GTZAN genre training for StyleCNN.

v2 scope (genre head, real numbers):

  * Two splits, run separately and reported alongside each other:
      - **fault** : jongpillee fault-filtered + artist-aware
        (train/val/test = 443/197/290 of 930 tracks). Numbers from here
        are the "honest" ones.
      - **naive** : per-genre random stratified split, fixed seed.
        Numbers from here are the optimistic baseline; the gap between
        naive and fault accuracy is the leakage bias.

  * Overfitting controls (small data, must-have):
      - random time crop (`--random-crop`)
      - SpecAugment freq + time masks (`--specaugment`)
      - head dropout (`--dropout`, default 0.3)
      - val-loss early stopping (`--early-stopping-patience`)

  * Mood head is **STUB** and not the focus of v2:
      - kept wired so checkpoints stay forward-compatible.
      - `--mood-weight 0.0` (the default) zeros the mood loss so the
        gradient does not push the backbone toward the fake target.

Report (`report.json`) contains: config, split source attribution,
skipped-file log, per-epoch history, best val acc + epoch, final
held-out test acc, full confusion matrix (counts + normalised),
train-val gap at the best-val epoch, representative GrooveStyle outputs
per genre. CPU-only by default.

CLI:
    python -m experiments.train_style \\
        --gtzan-root data/raw/gtzan_full/Data/genres_original \\
        --splits-dir data/raw/gtzan_splits \\
        --split-mode fault \\
        --out-dir data/style_full_fault \\
        --epochs 30 --batch-size 16 --dropout 0.3 \\
        --specaugment --random-crop --early-stopping-patience 6
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from groovebot.style.attributes import estimate_tempo
from groovebot.style.augment import random_time_crop, spec_augment
from groovebot.style.features import DEFAULT_SR, log_mel_spectrogram
from groovebot.style.model import GENRES, MOODS, StyleCNN
from groovebot.style.select import GrooveStyleSelector
from tools.gtzan_split import GTZANClip, SplitReport, build_split


# Same deterministic genre -> mood map as v1, retained so the mood head
# logits remain interpretable for reporting. STUB — replace with a real
# CC mood-tagged loader (MTG-Jamendo) for v3. v2's --mood-weight default
# of 0.0 means this map does NOT influence the genre gradient.
_STUB_MOOD: dict[str, str] = {
    "blues":     "sad",
    "classical": "calm",
    "country":   "calm",
    "disco":     "happy",
    "hiphop":    "dark",
    "jazz":      "calm",
    "metal":     "aggressive",
    "pop":       "happy",
    "reggae":    "happy",
    "rock":      "epic",
}


class GTZANStyleDataset(Dataset):
    """Loads GTZAN clips on the fly, crops to a window, optional
    augmentation.

    Random crop (when `random_crop=True`) keeps the time dim at
    `crop_frames`; the underlying audio window is still
    `window_sec = crop_frames * hop / sr`. When `random_crop=False` a
    deterministic center crop is used (val/test path).
    """

    def __init__(
        self,
        clips: list[GTZANClip],
        *,
        target_sr: int = DEFAULT_SR,
        window_sec: float = 8.0,
        n_mels: int = 64,
        random_crop: bool = False,
        crop_frames: int | None = None,
        specaugment: bool = False,
        seed: int = 0,
    ):
        self.clips = clips
        self.target_sr = int(target_sr)
        self.window_sec = float(window_sec)
        self.n_mels = int(n_mels)
        self.random_crop = bool(random_crop)
        self.specaugment = bool(specaugment)
        self.seed = int(seed)
        # crop_frames default: floor(window_sec * sr / hop_length=512)
        self.crop_frames = int(crop_frames) if crop_frames is not None else int(
            self.window_sec * self.target_sr / 512
        )
        # Bound the source audio we pull from disk: read a generous
        # 12 s superwindow so random_crop has room to wander; then mel
        # then crop. Keeps I/O cheap.
        self.source_window_sec = max(self.window_sec + 4.0, 12.0)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.random_crop:
            audio = _random_crop_audio(
                audio, sr, self.source_window_sec, seed=self.seed + idx,
            )
        else:
            audio = _center_crop(audio, sr, self.source_window_sec)
        mel = log_mel_spectrogram(
            audio, sr,
            target_sr=self.target_sr, n_mels=self.n_mels,
        )
        mel_t = torch.from_numpy(mel)
        if self.random_crop:
            g = torch.Generator().manual_seed(self.seed + idx)
            mel_t = random_time_crop(mel_t, self.crop_frames, generator=g)
        else:
            mel_t = _center_time_crop(mel_t, self.crop_frames)
        if self.specaugment:
            g = torch.Generator().manual_seed(self.seed + idx + 7919)
            mel_t = spec_augment(
                mel_t,
                n_freq_masks=2, freq_mask_max=8,
                n_time_masks=2, time_mask_max=20,
                generator=g,
            )
        return {
            "mel": mel_t.unsqueeze(0),  # (1, n_mels, crop_frames)
            "genre_idx": GENRES.index(clip.genre),
            "mood_idx": MOODS.index(_STUB_MOOD[clip.genre]),
            "path": str(clip.path),
            "genre": clip.genre,
        }


def _center_crop(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    n = int(window_sec * sr)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    start = (len(audio) - n) // 2
    return audio[start: start + n]


def _random_crop_audio(audio: np.ndarray, sr: int, window_sec: float,
                       seed: int) -> np.ndarray:
    n = int(window_sec * sr)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    rng = random.Random(seed)
    start = rng.randint(0, len(audio) - n)
    return audio[start: start + n]


def _center_time_crop(mel: torch.Tensor, crop_frames: int) -> torch.Tensor:
    T = mel.shape[-1]
    if T <= crop_frames:
        return F.pad(mel, (0, crop_frames - T))
    start = (T - crop_frames) // 2
    return mel[..., start: start + crop_frames]


def _collate(batch: list[dict]) -> dict:
    return {
        "mel": torch.stack([b["mel"] for b in batch], dim=0),
        "genre_idx": torch.tensor([b["genre_idx"] for b in batch], dtype=torch.long),
        "mood_idx": torch.tensor([b["mood_idx"] for b in batch], dtype=torch.long),
        "paths": [b["path"] for b in batch],
        "genres": [b["genre"] for b in batch],
    }


@dataclass
class EpochMetrics:
    loss_genre: float
    loss_mood: float
    acc_genre: float
    acc_mood: float


def train_one_epoch(
    model: StyleCNN, loader: DataLoader, opt: torch.optim.Optimizer,
    device: torch.device, mood_weight: float,
) -> EpochMetrics:
    model.train()
    lg, lm, cg, cm, n = [], [], 0, 0, 0
    for batch in loader:
        mel = batch["mel"].to(device)
        gt_g = batch["genre_idx"].to(device)
        gt_m = batch["mood_idx"].to(device)
        out = model(mel)
        loss_g = F.cross_entropy(out["genre"], gt_g)
        loss_m = F.cross_entropy(out["mood"], gt_m)
        loss = loss_g + mood_weight * loss_m
        opt.zero_grad()
        loss.backward()
        opt.step()
        lg.append(float(loss_g.item()))
        lm.append(float(loss_m.item()))
        cg += int((out["genre"].argmax(-1) == gt_g).sum().item())
        cm += int((out["mood"].argmax(-1) == gt_m).sum().item())
        n += int(gt_g.numel())
    return EpochMetrics(
        loss_genre=float(np.mean(lg) if lg else 0.0),
        loss_mood=float(np.mean(lm) if lm else 0.0),
        acc_genre=cg / max(n, 1),
        acc_mood=cm / max(n, 1),
    )


@torch.no_grad()
def evaluate(
    model: StyleCNN, loader: DataLoader, device: torch.device,
) -> tuple[EpochMetrics, np.ndarray]:
    """Return per-split metrics and a (10, 10) confusion matrix of genre."""
    model.eval()
    lg, lm, cg, cm, n = [], [], 0, 0, 0
    confusion = np.zeros((len(GENRES), len(GENRES)), dtype=np.int64)
    for batch in loader:
        mel = batch["mel"].to(device)
        gt_g = batch["genre_idx"].to(device)
        gt_m = batch["mood_idx"].to(device)
        out = model(mel)
        lg.append(float(F.cross_entropy(out["genre"], gt_g).item()))
        lm.append(float(F.cross_entropy(out["mood"], gt_m).item()))
        pred_g = out["genre"].argmax(-1)
        cg += int((pred_g == gt_g).sum().item())
        cm += int((out["mood"].argmax(-1) == gt_m).sum().item())
        n += int(gt_g.numel())
        for true_i, pred_i in zip(gt_g.cpu().numpy(), pred_g.cpu().numpy()):
            confusion[int(true_i), int(pred_i)] += 1
    metrics = EpochMetrics(
        loss_genre=float(np.mean(lg) if lg else 0.0),
        loss_mood=float(np.mean(lm) if lm else 0.0),
        acc_genre=cg / max(n, 1),
        acc_mood=cm / max(n, 1),
    )
    return metrics, confusion


def representative_labels(
    selector: GrooveStyleSelector, clips: list[GTZANClip], n_per_genre: int = 1,
) -> list[dict]:
    """One representative GrooveStyle output per genre, from the val/test
    side so the model has not seen it. Each row prints
    `move@intensity (genre/mood, BPM, arousal=X/bucket)`."""
    rows: list[dict] = []
    seen: dict[str, int] = {g: 0 for g in GENRES}
    for clip in clips:
        if seen[clip.genre] >= n_per_genre:
            continue
        seen[clip.genre] += 1
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, 8.0)
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gtzan-root", required=True,
                    help="root containing <genre>/<file>.wav for the 10 GTZAN genres")
    ap.add_argument("--splits-dir", default=None,
                    help="dir with {train,valid,test}_filtered.txt for "
                         "--split-mode fault (jongpillee/music_dataset_split)")
    ap.add_argument("--split-mode", choices=("naive", "fault"), default="fault")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="(naive only) fraction of clips per genre for val")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="(naive only) fraction of clips per genre for test")
    ap.add_argument("--window-sec", type=float, default=6.0)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--specaugment", action="store_true")
    ap.add_argument("--random-crop", action="store_true")
    ap.add_argument("--early-stopping-patience", type=int, default=6,
                    help="stop if val_acc_genre does not improve for N "
                         "consecutive epochs (0 disables)")
    ap.add_argument("--mood-weight", type=float, default=0.0,
                    help="loss weight for the mood STUB head; default 0 "
                         "leaves the mood head wired but not trained")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else None
    report: SplitReport = build_split(
        Path(args.gtzan_root),
        mode=args.split_mode,
        splits_dir=splits_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    if not report.train or not report.val:
        print(f"empty split: counts={report.counts()}", file=sys.stderr)
        return 2
    if not report.test:
        print("WARNING: no test clips — only val will be reported",
              file=sys.stderr)

    train_ds = GTZANStyleDataset(
        report.train,
        window_sec=args.window_sec, n_mels=args.n_mels,
        random_crop=args.random_crop, specaugment=args.specaugment,
        seed=args.seed,
    )
    # val/test always use center crop + no augment for stable measurement.
    val_ds = GTZANStyleDataset(
        report.val,
        window_sec=args.window_sec, n_mels=args.n_mels,
        random_crop=False, specaugment=False, seed=args.seed,
    )
    test_ds = GTZANStyleDataset(
        report.test,
        window_sec=args.window_sec, n_mels=args.n_mels,
        random_crop=False, specaugment=False, seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
    )

    device = torch.device("cpu")
    model = StyleCNN(n_mels=args.n_mels, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    best_train_acc_at_best_val = 0.0
    no_improve = 0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_m = train_one_epoch(
            model, train_loader, opt, device, args.mood_weight,
        )
        val_m, _ = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        history.append({
            "epoch": epoch,
            "train": train_m.__dict__,
            "val": val_m.__dict__,
            "dt_sec": round(dt, 2),
        })
        print(
            f"epoch {epoch:02d}  loss_g={train_m.loss_genre:.3f}  "
            f"train_g={train_m.acc_genre:.3f}  val_g={val_m.acc_genre:.3f}  "
            f"({dt:.1f}s)"
        )
        if val_m.acc_genre > best_val_acc:
            best_val_acc = val_m.acc_genre
            best_epoch = epoch
            best_train_acc_at_best_val = train_m.acc_genre
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
                print(f"early stopping at epoch {epoch} "
                      f"(no val improvement for {no_improve} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, test_confusion = evaluate(model, test_loader, device)

    ckpt_path = out_dir / "style_cnn.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_mels": args.n_mels,
        "dropout": args.dropout,
        "split_mode": args.split_mode,
    }, ckpt_path)

    selector = GrooveStyleSelector(model=model, n_mels=args.n_mels)
    test_examples = representative_labels(selector, report.test or report.val)

    final_report = {
        "config": vars(args),
        "split_mode": args.split_mode,
        "split_counts": report.counts(),
        "split_sources": report.sources,
        "skipped": report.skipped,
        "history": history,
        "best_val_genre_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "train_val_gap_at_best": round(
            best_train_acc_at_best_val - best_val_acc, 4
        ),
        "test": {
            "loss_genre": round(test_m.loss_genre, 4),
            "acc_genre": round(test_m.acc_genre, 4),
        },
        "confusion_matrix": {
            "labels": list(GENRES),
            "counts": test_confusion.tolist(),
            "row_normalised": _row_normalise(test_confusion).tolist(),
        },
        "representative_groove_styles": test_examples,
        "stub_mood_map": _STUB_MOOD,
        "caveats": [
            "GTZAN has well-documented faults (duplicates, mislabels, "
            "artist-overlap). The 'fault' split mode applies the "
            "jongpillee fault-filtered + artist-aware partition "
            "(Kereliuk 2015 / Sturm 2013) but artist labels are "
            "partial — artist-non-overlap is best-effort.",
            "Mood head (--mood-weight default 0.0) is a STUB tied to a "
            "deterministic genre->mood map. Its accuracy is meaningless. "
            "Replace with MTG-Jamendo mood subset before drawing any "
            "conclusion about mood.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nbest val acc: {best_val_acc:.3f} @ epoch {best_epoch}")
    print(f"test acc:     {test_m.acc_genre:.3f}  "
          f"(train-val gap at best: "
          f"{best_train_acc_at_best_val - best_val_acc:+.3f})")
    print(f"saved checkpoint -> {ckpt_path}")
    print(f"saved report     -> {out_dir / 'report.json'}")
    return 0


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    return (cm.astype(np.float64) / safe).round(3)


if __name__ == "__main__":
    sys.exit(main())
