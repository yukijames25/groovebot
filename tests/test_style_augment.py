"""Tests for groovebot.style.augment (random crop + SpecAugment)."""
from __future__ import annotations

import torch

from groovebot.style.augment import random_time_crop, spec_augment


def test_random_time_crop_length():
    mel = torch.randn(64, 300)
    out = random_time_crop(mel, crop_frames=128)
    assert out.shape == (64, 128)


def test_random_time_crop_pads_when_input_shorter():
    mel = torch.randn(64, 50)
    out = random_time_crop(mel, crop_frames=128)
    assert out.shape == (64, 128)
    # the trailing padding is zeros
    assert torch.all(out[..., 50:] == 0)


def test_random_time_crop_with_batch_dim():
    mel = torch.randn(4, 1, 64, 300)
    out = random_time_crop(mel, crop_frames=200)
    assert out.shape == (4, 1, 64, 200)


def test_random_time_crop_deterministic_with_generator():
    mel = torch.arange(64 * 300, dtype=torch.float32).reshape(64, 300)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = random_time_crop(mel, 128, generator=g1)
    b = random_time_crop(mel, 128, generator=g2)
    assert torch.equal(a, b)


def test_spec_augment_preserves_shape():
    mel = torch.randn(64, 200)
    out = spec_augment(mel, n_freq_masks=2, freq_mask_max=8,
                       n_time_masks=2, time_mask_max=20)
    assert out.shape == mel.shape


def test_spec_augment_creates_zeroed_bands():
    mel = torch.ones(64, 200)
    out = spec_augment(
        mel, n_freq_masks=1, freq_mask_max=4,
        n_time_masks=1, time_mask_max=20,
        generator=torch.Generator().manual_seed(42),
    )
    # At least one element should be zero (a masked band)
    assert torch.any(out == 0.0)
    # And the rest should still be 1 (unmodified)
    assert torch.any(out == 1.0)


def test_spec_augment_does_not_mutate_input():
    mel = torch.ones(64, 200)
    before = mel.clone()
    _ = spec_augment(mel, generator=torch.Generator().manual_seed(7))
    assert torch.equal(mel, before)


def test_spec_augment_zero_masks_returns_copy_equal_to_input():
    mel = torch.randn(64, 50)
    out = spec_augment(mel, n_freq_masks=0, n_time_masks=0)
    assert torch.equal(out, mel)
    assert out.data_ptr() != mel.data_ptr()  # genuine copy, not aliased


def test_spec_augment_with_batch_dim():
    mel = torch.randn(4, 1, 64, 200)
    out = spec_augment(mel, n_freq_masks=1, n_time_masks=1,
                       generator=torch.Generator().manual_seed(0))
    assert out.shape == mel.shape


def test_spec_augment_clips_widths_for_tiny_input():
    # 5-frame input vs time_mask_max=20: mask should clip to T, not crash.
    mel = torch.randn(8, 5)
    out = spec_augment(mel, n_freq_masks=1, freq_mask_max=4,
                       n_time_masks=1, time_mask_max=20,
                       generator=torch.Generator().manual_seed(0))
    assert out.shape == (8, 5)
