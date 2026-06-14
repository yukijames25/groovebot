"""experiments/train_style.py — GTZAN genre training + mood stub for StyleCNN.

Vertical-slice training script for GrooveStyleSelector v1 (spec §14 module
note). End-to-end goal: show that the feature → CNN → table pipeline
trains, holds out, and produces sensible GrooveStyle labels on
representative clips. SOTA accuracy is NOT the goal — the gtzan_mini
subset is 10 clips/genre, so genre top-1 in the 30-50% range is
expected.

Two heads:

  * **genre** (real): GTZAN 10-class. Walk `--gtzan-root/<genre>/*.wav`.
  * **mood** (stub): no CC mood-tagged audio is wired up yet (the spec
    points at MTG-Jamendo mood subset; see TODO at bottom). For v1 we
    use a deterministic genre -> mood pseudo-map so the multi-head
    forward + loss + reporting code is exercised. Replace `_STUB_MOOD`
    with a real loader once MTG-Jamendo (or similar) is on disk.

Run:
    python -m experiments.train_style \\
        --gtzan-root data/raw/gtzan_mini/genres \\
        --out-dir data/style_work \\
        --epochs 20

CPU-only. Saves model + JSON report to `--out-dir`.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from groovebot.style.attributes import (
    arousal_bucket,
    estimate_arousal,
    estimate_tempo,
)
from groovebot.style.features import DEFAULT_SR, log_mel_spectrogram
from groovebot.style.model import GENRES, MOODS, StyleCNN
from groovebot.style.select import GrooveStyleSelector
from groovebot.style.table import select_move


# Deterministic genre -> mood pseudo-mapping. STUB — replace with a
# real mood-tagged loader (MTG-Jamendo / FMA mood, spec §14 module note).
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


@dataclass
class _Clip:
    path: Path
    genre: str
    mood: str    # stub label


class GTZANStyleDataset(Dataset):
    """Loads GTZAN clips on the fly, crops to `window_sec`, returns log-mel.

    GTZAN clips are 30 s at 22050 Hz. We take a fixed-position center
    crop so train and val see the same audio per clip (deterministic;
    a random crop would be a lighter form of augmentation, deferred).
    """

    def __init__(
        self,
        clips: list[_Clip],
        *,
        target_sr: int = DEFAULT_SR,
        window_sec: float = 8.0,
        n_mels: int = 64,
    ):
        self.clips = clips
        self.target_sr = int(target_sr)
        self.window_sec = float(window_sec)
        self.n_mels = int(n_mels)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, self.window_sec)
        mel = log_mel_spectrogram(
            audio, sr,
            target_sr=self.target_sr, n_mels=self.n_mels,
        )
        return {
            "mel": torch.from_numpy(mel).unsqueeze(0),  # (1, n_mels, T)
            "genre_idx": GENRES.index(clip.genre),
            "mood_idx": MOODS.index(clip.mood),
            "path": str(clip.path),
            "genre": clip.genre,
            "mood": clip.mood,
        }


def _center_crop(audio: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    n_target = int(window_sec * sr)
    if len(audio) <= n_target:
        # zero-pad short signals
        return np.pad(audio, (0, n_target - len(audio)))
    start = (len(audio) - n_target) // 2
    return audio[start: start + n_target]


def _collate(batch: list[dict]) -> dict:
    """Pad time dim to the max in the batch so all mels stack."""
    max_t = max(b["mel"].shape[-1] for b in batch)
    mels = []
    for b in batch:
        m = b["mel"]
        if m.shape[-1] < max_t:
            m = F.pad(m, (0, max_t - m.shape[-1]))
        mels.append(m)
    return {
        "mel": torch.stack(mels, dim=0),
        "genre_idx": torch.tensor([b["genre_idx"] for b in batch], dtype=torch.long),
        "mood_idx": torch.tensor([b["mood_idx"] for b in batch], dtype=torch.long),
        "paths": [b["path"] for b in batch],
        "genres": [b["genre"] for b in batch],
        "moods": [b["mood"] for b in batch],
    }


def discover_clips(gtzan_root: Path) -> list[_Clip]:
    """Walk `gtzan_root/<genre>/*.wav` and tag each clip with stub mood."""
    clips: list[_Clip] = []
    for genre in GENRES:
        d = gtzan_root / genre
        if not d.is_dir():
            continue
        mood = _STUB_MOOD[genre]
        for wav in sorted(d.glob("*.wav")):
            clips.append(_Clip(path=wav, genre=genre, mood=mood))
    return clips


def stratified_split(
    clips: list[_Clip], val_frac: float, seed: int = 0,
) -> tuple[list[_Clip], list[_Clip]]:
    """Per-genre stratified train/val split."""
    rng = random.Random(seed)
    train: list[_Clip] = []
    val: list[_Clip] = []
    by_genre: dict[str, list[_Clip]] = {}
    for c in clips:
        by_genre.setdefault(c.genre, []).append(c)
    for genre, group in by_genre.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def train_one_epoch(
    model: StyleCNN,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    mood_weight: float,
) -> dict:
    model.train()
    losses_g, losses_m = [], []
    correct_g, correct_m, total = 0, 0, 0
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
        losses_g.append(float(loss_g.item()))
        losses_m.append(float(loss_m.item()))
        correct_g += int((out["genre"].argmax(-1) == gt_g).sum().item())
        correct_m += int((out["mood"].argmax(-1) == gt_m).sum().item())
        total += int(gt_g.numel())
    return {
        "loss_genre": float(np.mean(losses_g)),
        "loss_mood": float(np.mean(losses_m)),
        "acc_genre": correct_g / max(total, 1),
        "acc_mood": correct_m / max(total, 1),
    }


@torch.no_grad()
def evaluate(model: StyleCNN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct_g, correct_m, total = 0, 0, 0
    for batch in loader:
        mel = batch["mel"].to(device)
        gt_g = batch["genre_idx"].to(device)
        gt_m = batch["mood_idx"].to(device)
        out = model(mel)
        correct_g += int((out["genre"].argmax(-1) == gt_g).sum().item())
        correct_m += int((out["mood"].argmax(-1) == gt_m).sum().item())
        total += int(gt_g.numel())
    return {
        "acc_genre": correct_g / max(total, 1),
        "acc_mood": correct_m / max(total, 1),
    }


def representative_labels(
    selector: GrooveStyleSelector,
    clips: list[_Clip],
    n_per_genre: int = 1,
) -> list[dict]:
    """Run the full GrooveStyleSelector on one clip per genre, plus a
    tempo error reading. Used to make the report concrete: the reader can
    see the actual `move@intensity (genre/mood, BPM, arousal)` string a
    real clip produced."""
    rows: list[dict] = []
    seen_per_genre: dict[str, int] = {g: 0 for g in GENRES}
    for clip in clips:
        if seen_per_genre[clip.genre] >= n_per_genre:
            continue
        seen_per_genre[clip.genre] += 1
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _center_crop(audio, sr, 8.0)
        style = selector.select(audio, sr)
        rows.append({
            "path": str(clip.path),
            "true_genre": clip.genre,
            "true_mood_stub": clip.mood,
            "predicted_genre": style.genre,
            "predicted_mood": style.mood,
            "move": style.move,
            "intensity": round(style.intensity, 3),
            "tempo_bpm": round(style.tempo_bpm, 1),
            "arousal": round(style.arousal, 3),
            "arousal_bucket": style.arousal_bucket,
            "text": style.as_text(),
        })
    return rows


def tempo_error_summary(clips: list[_Clip], n: int = 6) -> dict:
    """Tempo estimates on a handful of clips. There's no ground-truth BPM
    in gtzan_mini, so we just report the distribution and let the reader
    sanity-check (classical should be slow, metal fast, etc)."""
    sample = clips[:n] if len(clips) >= n else clips
    rows = []
    for clip in sample:
        audio, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        bpm = estimate_tempo(audio, sr)
        rows.append({"path": str(clip.path), "genre": clip.genre,
                     "estimated_bpm": round(float(bpm), 1)})
    return {"clips": rows}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gtzan-root", required=True,
                    help="dir with <genre>/<file>.wav subfolders")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--mood-weight", type=float, default=1.0,
                    help="loss multiplier on the stub mood head (lowering "
                         "to 0 disables mood gradients while leaving the "
                         "head wired for reporting)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clips = discover_clips(Path(args.gtzan_root))
    if not clips:
        print(f"no clips found under {args.gtzan_root!r}", file=sys.stderr)
        return 2
    train_clips, val_clips = stratified_split(clips, args.val_frac, args.seed)

    train_ds = GTZANStyleDataset(
        train_clips, window_sec=args.window_sec, n_mels=args.n_mels,
    )
    val_ds = GTZANStyleDataset(
        val_clips, window_sec=args.window_sec, n_mels=args.n_mels,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate,
    )

    device = torch.device("cpu")
    model = StyleCNN(n_mels=args.n_mels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, opt, device, args.mood_weight,
        )
        val_metrics = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "dt_sec": round(dt, 2),
        })
        print(
            f"epoch {epoch:02d}  loss_g={train_metrics['loss_genre']:.3f} "
            f"loss_m={train_metrics['loss_mood']:.3f} "
            f"train_g={train_metrics['acc_genre']:.2f} "
            f"val_g={val_metrics['acc_genre']:.2f} "
            f"val_m={val_metrics['acc_mood']:.2f} ({dt:.1f}s)"
        )

    ckpt_path = out_dir / "style_cnn.pt"
    torch.save({"state_dict": model.state_dict(), "n_mels": args.n_mels},
               ckpt_path)

    selector = GrooveStyleSelector(model=model, n_mels=args.n_mels)
    rep_rows = representative_labels(selector, val_clips)
    tempo_rows = tempo_error_summary(val_clips, n=min(10, len(val_clips)))

    report = {
        "config": vars(args),
        "n_train": len(train_clips),
        "n_val": len(val_clips),
        "history": history,
        "final": history[-1] if history else {},
        "representative_labels": rep_rows,
        "tempo_estimates": tempo_rows,
        "stub_mood_map": _STUB_MOOD,
        "TODO": [
            "Replace _STUB_MOOD with a CC mood-tagged loader "
            "(MTG-Jamendo mood subset, ~14k clips with autotag mood "
            "labels). Until then mood val accuracy is meaningless: it "
            "just measures whether the model learned the genre->mood "
            "deterministic map.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved checkpoint -> {ckpt_path}")
    print(f"saved report     -> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
