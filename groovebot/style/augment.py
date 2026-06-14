"""groovebot.style.augment — log-mel augmentations for training.

Two augmentations, both train-only (inference path is `select.py` and
sees raw, full mels):

  * `random_time_crop`: pick a random `crop_frames`-wide window of the
    mel along the time axis. Adds positional variance so a fixed center
    crop of GTZAN can't be memorised.

  * `spec_augment`: SpecAugment (Park et al. 2019) — random freq and
    time masks zero out small bands. Cheap, well-studied, and the
    canonical defence against the overfitting we expect on
    ~700-train-clip GTZAN.

Both functions are pure: they take a tensor, return a tensor. State
(seed) lives in the caller / DataLoader worker. Designed to be applied
inside the `Dataset.__getitem__` after `log_mel_spectrogram`, BEFORE
batching, because per-clip random masks differ.
"""
from __future__ import annotations

import torch


def random_time_crop(
    mel: torch.Tensor,
    crop_frames: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Crop along the time axis to exactly `crop_frames`.

    Accepts (..., n_mels, T). If `T <= crop_frames`, zero-pads to
    `crop_frames` on the right. Otherwise picks a uniformly random start
    in `[0, T - crop_frames]`.
    """
    if mel.dim() < 2:
        raise ValueError(f"expected (..., n_mels, T), got shape {tuple(mel.shape)}")
    T = mel.shape[-1]
    if T <= crop_frames:
        pad = crop_frames - T
        return torch.nn.functional.pad(mel, (0, pad))
    if generator is not None:
        start = int(torch.randint(
            low=0, high=T - crop_frames + 1, size=(1,), generator=generator,
        ).item())
    else:
        start = int(torch.randint(
            low=0, high=T - crop_frames + 1, size=(1,),
        ).item())
    return mel[..., start: start + crop_frames]


def spec_augment(
    mel: torch.Tensor,
    *,
    n_freq_masks: int = 2,
    freq_mask_max: int = 8,
    n_time_masks: int = 2,
    time_mask_max: int = 20,
    fill_value: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply SpecAugment freq + time masks. Returns a *new* tensor.

    Mask widths are sampled uniformly in `[0, *_mask_max]`, then a start
    offset is sampled uniformly so the mask lies within the axis. The
    `fill_value` defaults to 0; for log-mel (dB scale) you may prefer
    the mel batch mean — keep 0 for v2 to match the canonical
    formulation. Both axes are masked even on tiny tensors; widths are
    clipped to the available range.
    """
    if mel.dim() < 2:
        raise ValueError(f"expected (..., n_mels, T), got shape {tuple(mel.shape)}")
    out = mel.clone()
    n_mels = out.shape[-2]
    T = out.shape[-1]

    def _randint(high: int) -> int:
        if high <= 0:
            return 0
        if generator is not None:
            return int(torch.randint(
                low=0, high=high, size=(1,), generator=generator,
            ).item())
        return int(torch.randint(low=0, high=high, size=(1,)).item())

    for _ in range(int(n_freq_masks)):
        max_w = min(freq_mask_max, n_mels)
        w = _randint(max_w + 1) if max_w > 0 else 0
        if w <= 0 or n_mels <= 0:
            continue
        f0 = _randint(max(1, n_mels - w + 1))
        out[..., f0: f0 + w, :] = fill_value

    for _ in range(int(n_time_masks)):
        max_w = min(time_mask_max, T)
        w = _randint(max_w + 1) if max_w > 0 else 0
        if w <= 0 or T <= 0:
            continue
        t0 = _randint(max(1, T - w + 1))
        out[..., :, t0: t0 + w] = fill_value

    return out


__all__ = ["random_time_crop", "spec_augment"]
