"""Tests for tools/degrade_audio.py.

Audio degradation primitives for the noise-robustness experiment. Each
function is tested against the physical invariant it claims to enforce
(SNR, decay, bandwidth, …), plus determinism (same seed → identical
output) so the experiment can be re-run without drift.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ─── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sine():
    """A 1-second mono 440 Hz sine wave at amplitude 0.5."""
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32), sr


# ─── (a) add_white_noise ──────────────────────────────────────────────────

def test_add_white_noise_reaches_target_snr(sine):
    """Measured SNR must land within 1 dB of the requested value."""
    from tools.degrade_audio import add_white_noise
    y, _ = sine
    for target in (30.0, 20.0, 10.0, 5.0):
        noisy = add_white_noise(y, snr_db=target, seed=7)
        sig_pow = float(np.mean(y.astype(np.float64) ** 2))
        n = (noisy - y).astype(np.float64)
        noise_pow = float(np.mean(n ** 2))
        measured = 10.0 * math.log10(sig_pow / noise_pow)
        assert abs(measured - target) < 1.0, (target, measured)


def test_add_white_noise_is_deterministic(sine):
    from tools.degrade_audio import add_white_noise
    y, _ = sine
    a = add_white_noise(y, snr_db=10.0, seed=42)
    b = add_white_noise(y, snr_db=10.0, seed=42)
    assert np.array_equal(a, b)
    c = add_white_noise(y, snr_db=10.0, seed=43)
    assert not np.array_equal(a, c)


def test_add_white_noise_zero_input_stays_zero():
    from tools.degrade_audio import add_white_noise
    y = np.zeros(1024, dtype=np.float32)
    out = add_white_noise(y, snr_db=20.0, seed=0)
    assert np.array_equal(out, y)   # no SNR defined -> pass-through


# ─── (b) apply_reverb ─────────────────────────────────────────────────────

def test_reverb_output_shape_matches_input(sine):
    from tools.degrade_audio import apply_reverb
    y, sr = sine
    for preset in ("short", "medium", "long"):
        out = apply_reverb(y, sr, preset=preset, seed=1)
        assert out.shape == y.shape
        assert out.dtype == np.float32


def test_reverb_longer_rt60_increases_tail_energy(sine):
    """A tail-energy proxy: energy of the last 25% of the clip, as a
    fraction of total energy. Longer RT60 → more energy left at the end
    (slower decay). Compare short vs long with a feed-through dirac."""
    from tools.degrade_audio import apply_reverb
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)
    y[0] = 1.0  # a dirac so the output is just the IR, truncated to 1s
    tail = lambda z: float(np.sum(z[int(0.75 * len(z)):] ** 2))
    e_short = tail(apply_reverb(y, sr, preset="short", seed=1))
    e_long  = tail(apply_reverb(y, sr, preset="long",  seed=1))
    assert e_long > e_short * 3.0, (e_short, e_long)


def test_reverb_is_deterministic(sine):
    from tools.degrade_audio import apply_reverb
    y, sr = sine
    a = apply_reverb(y, sr, preset="medium", seed=11)
    b = apply_reverb(y, sr, preset="medium", seed=11)
    assert np.array_equal(a, b)


# ─── (c) change_volume ────────────────────────────────────────────────────

def test_change_volume_scales_rms(sine):
    """-20 dB attenuation must drop RMS by 10x (-20 dB)."""
    from tools.degrade_audio import change_volume
    y, _ = sine
    rms = lambda z: float(np.sqrt(np.mean(z.astype(np.float64) ** 2)))
    down = change_volume(y, gain_db=-20.0, clip=False)
    assert rms(down) / rms(y) == pytest.approx(0.1, abs=1e-5)


def test_change_volume_clips_at_plus_6db():
    """A sine at 0.7 amplitude boosted by +6 dB (~2x = 1.4) must saturate.
    (0.5 would multiply to ~0.9977 — just under the rail — so the test
    picks 0.7 to put the uncompressed peak reliably over 1.0.)"""
    from tools.degrade_audio import change_volume
    sr = 22050
    y = (0.7 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr) / sr)).astype(np.float32)
    out = change_volume(y, gain_db=+6.0, clip=True)
    assert float(np.abs(out).max()) <= 1.0 + 1e-6
    # some samples should land exactly at the clip rail
    assert np.sum(np.abs(out) >= 0.999) > 0


# ─── (d) band_limit ───────────────────────────────────────────────────────

def test_band_limit_telephone_rolls_off_low_and_high():
    """A broadband impulse driven through the telephone band must show
    sharp attenuation below 300 Hz and above 3400 Hz."""
    from tools.degrade_audio import band_limit
    sr = 22050
    rng = np.random.default_rng(0)
    y = rng.standard_normal(sr).astype(np.float32)  # 1 s broadband
    out = band_limit(y, sr, preset="telephone")
    # Welch PSD: average power in and out of the passband.
    import scipy.signal as sps
    f, p_in  = sps.welch(y,   fs=sr, nperseg=2048)
    f, p_out = sps.welch(out, fs=sr, nperseg=2048)
    in_band = (f > 500) & (f < 3000)
    below   = (f > 20)  & (f < 150)
    above   = f > 5000
    gain_in    = p_out[in_band].mean() / p_in[in_band].mean()
    gain_below = p_out[below].mean()   / p_in[below].mean()
    gain_above = p_out[above].mean()   / p_in[above].mean()
    assert gain_in >= 0.5                       # in-band preserved
    assert gain_below < 0.01                    # strong low-cut
    assert gain_above < 0.01                    # strong high-cut


def test_band_limit_rejects_backwards_bounds(sine):
    from tools.degrade_audio import band_limit
    y, sr = sine
    with pytest.raises(ValueError):
        band_limit(y, sr, low_hz=4000, high_hz=200)


# ─── (e) lossy_recompress ─────────────────────────────────────────────────

def test_lossy_recompress_mp3_roundtrip_preserves_length_and_shape(sine):
    from tools.degrade_audio import lossy_recompress
    y, sr = sine
    out = lossy_recompress(y, sr, codec="mp3", compression_level=0.5)
    assert out.shape == y.shape
    assert out.dtype == np.float32
    # Signal is still music, not silence: correlation with original
    # is high even after a round trip.
    corr = float(np.corrcoef(y, out)[0, 1])
    assert corr > 0.6, corr


def test_lossy_recompress_lower_quality_departs_further(sine):
    """compression_level higher = worse quality = MSE from original rises."""
    from tools.degrade_audio import lossy_recompress
    y, sr = sine
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    high_q = lossy_recompress(y, sr, codec="mp3", compression_level=0.1)
    low_q  = lossy_recompress(y, sr, codec="mp3", compression_level=0.9)
    assert mse(y, low_q) > mse(y, high_q)


# ─── Condition dataclasses ────────────────────────────────────────────────

def test_default_conditions_cover_all_five_families():
    from tools.degrade_audio import default_conditions
    conds = default_conditions()
    families = {c.family for c in conds}
    assert families == {"noise", "reverb", "volume", "band", "codec"}


def test_each_condition_apply_returns_matching_shape(sine):
    """Each registered condition must preserve waveform shape — the
    pipeline downstream assumes mono float32 of the input length."""
    from tools.degrade_audio import default_conditions
    y, sr = sine
    for c in default_conditions():
        out = c.apply(y, sr, seed=0)
        assert out.shape == y.shape, (c.label, out.shape, y.shape)
        assert out.dtype == np.float32, (c.label, out.dtype)
