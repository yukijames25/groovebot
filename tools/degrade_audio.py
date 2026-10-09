"""tools/degrade_audio.py — reproducible audio degradation functions.

Each public function takes a mono float32 waveform and returns a degraded
version of the same shape / sample rate. All randomness is seeded; the
same (audio, params, seed) triple always produces the same output.

The functions model the five degradation families from the noise-
robustness experiment (SYSTEM_SPEC §6.4 adjacent, not yet in the spec):

  (a) `add_white_noise`  — additive Gaussian noise at a target SNR
  (b) `apply_reverb`     — synthetic exponentially-decaying room IR
  (c) `change_volume`    — scalar gain in dB (clips to [-1, 1] after)
  (d) `band_limit`       — IIR 4th-order Butterworth bandpass
  (e) `lossy_recompress` — write + read via soundfile at MP3 / OGG
                           (compression_level is the quality knob)

They are implemented in pure NumPy + SciPy + soundfile; no training-time
dependencies. Each function is a pure transform of (audio, params); the
runner at experiments/run_noise_robustness.py composes them.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.signal as sps
import soundfile as sf


EPS = 1e-12


# ─── (a) additive noise ───────────────────────────────────────────────────

def add_white_noise(
    audio: np.ndarray,
    *,
    snr_db: float,
    seed: int = 0,
) -> np.ndarray:
    """Return `audio + n` where `n ~ N(0, sigma)` and sigma is picked so
    that 10·log10(signal_power / noise_power) = snr_db.

    Signal power is RMS^2 of the input (full waveform); if the input is
    all-zero the function returns it unchanged (no SNR defined).
    """
    y = np.asarray(audio, dtype=np.float32)
    sig_pow = float(np.mean(y.astype(np.float64) ** 2))
    if sig_pow < EPS:
        return y.copy()
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=math.sqrt(noise_pow),
                       size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)


# ─── (b) reverb (synthetic exponentially-decaying noise IR) ──────────────

@dataclass(frozen=True)
class ReverbParams:
    rt60_s: float    # time (s) for the IR to decay by 60 dB
    direct_mix: float = 0.6   # dry/wet mix: 1.0 = dry only, 0.0 = wet only


_REVERB_PRESETS: dict[str, ReverbParams] = {
    # Rough matches: a dry voice booth, a living room, a cathedral hallway.
    "short":  ReverbParams(rt60_s=0.20, direct_mix=0.70),
    "medium": ReverbParams(rt60_s=0.60, direct_mix=0.55),
    "long":   ReverbParams(rt60_s=1.50, direct_mix=0.40),
}


def _make_ir(sr: int, rt60_s: float, seed: int) -> np.ndarray:
    """Build a synthetic room impulse response.

    Model: coloured noise multiplied by an exponential envelope
    `exp(-t · 3·ln10 / RT60)`. The constant chosen makes the envelope
    reach -60 dB at exactly RT60. Zero-pads out to 2·RT60 so the tail
    is captured and the convolution length is predictable.
    """
    n = int(round(2.0 * max(0.01, float(rt60_s)) * sr))
    if n < 2:
        n = 2
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(n).astype(np.float32)
    # bandpass to make the colouration sound less like white hiss
    sos = sps.butter(2, [120 / (sr / 2), min(0.98, 8000 / (sr / 2))],
                     btype="bandpass", output="sos")
    noise = sps.sosfiltfilt(sos, noise).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / float(sr)
    env = np.exp(-t * (3.0 * math.log(10.0) / float(rt60_s))).astype(np.float32)
    ir = noise * env
    # unit-energy normalise, add a direct delta at t=0 for the dry source
    ir /= (np.sqrt(np.sum(ir.astype(np.float64) ** 2)) + EPS)
    ir[0] += 1.0
    return ir


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    *,
    preset: Literal["short", "medium", "long"] | None = None,
    rt60_s: float | None = None,
    direct_mix: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Convolve `audio` with a synthetic IR and mix with the dry signal.

    Either pass a named `preset` or give an explicit `rt60_s` (+ optional
    `direct_mix`). The output length matches the input (the reverb tail
    beyond the input duration is discarded, as would happen in a live
    streaming scenario).
    """
    y = np.asarray(audio, dtype=np.float32)
    if preset is not None:
        p = _REVERB_PRESETS[preset]
        rt60 = p.rt60_s
        mix = p.direct_mix
    else:
        if rt60_s is None:
            raise ValueError("pass either preset or rt60_s")
        rt60 = float(rt60_s)
        mix = float(direct_mix)
    ir = _make_ir(int(sr), rt60, seed=seed)
    wet = sps.fftconvolve(y, ir, mode="full")[: len(y)].astype(np.float32)
    # the dry component is already baked into the IR (impulse at t=0),
    # so the mix here just rebalances dry-vs-tail. Scale the already-
    # convolved wet by its own peak so the output level ≈ input level.
    peak = float(np.max(np.abs(wet))) + EPS
    wet = wet / peak * float(np.max(np.abs(y)) + EPS)
    return (mix * y + (1.0 - mix) * wet).astype(np.float32)


# ─── (c) volume / clipping ────────────────────────────────────────────────

def change_volume(
    audio: np.ndarray,
    *,
    gain_db: float,
    clip: bool = True,
) -> np.ndarray:
    """Apply a gain of `gain_db` dB. By default, clip to [-1, 1].

    `clip=True` models the "digital overflow" failure mode when a loud
    gain pushes samples past the DAC range; positive gain combined with
    an already-hot signal is the usual way callers trigger clipping.
    """
    y = np.asarray(audio, dtype=np.float32)
    g = 10.0 ** (float(gain_db) / 20.0)
    out = (y * g).astype(np.float32)
    if clip:
        np.clip(out, -1.0, 1.0, out=out)
    return out


# ─── (d) band limit ───────────────────────────────────────────────────────

_BAND_PRESETS: dict[str, tuple[float, float]] = {
    # 300-3400 Hz is the ITU-T G.712 telephone band.
    # 100-8000 Hz is a wide-cheap-mic / podcast-mic rough spec.
    "telephone": (300.0, 3400.0),
    "cheap_mic": (100.0, 8000.0),
}


def band_limit(
    audio: np.ndarray,
    sr: int,
    *,
    preset: Literal["telephone", "cheap_mic"] | None = None,
    low_hz: float | None = None,
    high_hz: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """4th-order Butterworth bandpass. Nyquist-safe (clamps high_hz)."""
    y = np.asarray(audio, dtype=np.float32)
    if preset is not None:
        lo, hi = _BAND_PRESETS[preset]
    else:
        if low_hz is None or high_hz is None:
            raise ValueError("pass preset or (low_hz, high_hz)")
        lo, hi = float(low_hz), float(high_hz)
    nyq = sr / 2.0
    hi = min(hi, nyq - 1.0)
    lo = max(lo, 1.0)
    if hi <= lo:
        raise ValueError(f"band_limit: high_hz ({hi}) must exceed low_hz ({lo})")
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


# ─── (e) lossy recompression (MP3 / OGG round-trip) ──────────────────────

def lossy_recompress(
    audio: np.ndarray,
    sr: int,
    *,
    codec: Literal["mp3", "ogg"] = "mp3",
    compression_level: float = 0.5,
) -> np.ndarray:
    """Encode `audio` through `codec` at the given compression level,
    then decode back. Higher compression_level = smaller file / lower
    quality (soundfile convention: 0.0 = highest quality, 1.0 = lowest).

    `soundfile` writes to an in-memory BytesIO, so no temp files touch
    disk. The returned waveform matches the input length (padded or
    trimmed as needed) and is float32 mono.
    """
    y = np.asarray(audio, dtype=np.float32)
    fmt = {"mp3": "MP3", "ogg": "OGG"}[codec]
    subtype = {"mp3": "MPEG_LAYER_III", "ogg": "VORBIS"}[codec]
    buf = io.BytesIO()
    sf.write(
        buf, y, int(sr), format=fmt, subtype=subtype,
        compression_level=float(compression_level),
    )
    buf.seek(0)
    decoded, _sr = sf.read(buf, dtype="float32", always_2d=False)
    if decoded.ndim == 2:
        decoded = decoded.mean(axis=1).astype(np.float32)
    # Encoders may add a few frames of latency; trim or pad to source length.
    if len(decoded) > len(y):
        decoded = decoded[: len(y)]
    elif len(decoded) < len(y):
        decoded = np.pad(decoded, (0, len(y) - len(decoded)))
    return decoded.astype(np.float32)


# ─── named condition set used by the experiment runner ───────────────────

@dataclass(frozen=True)
class Condition:
    """One row in the robustness grid.

    `family` is the degradation category (noise / reverb / volume / band /
    codec), `label` is a short display string, and `apply` applies the
    degradation to a (audio, sr) pair.
    """
    family: str
    label: str

    def apply(self, audio: np.ndarray, sr: int, *, seed: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class NoiseCondition(Condition):
    snr_db: float

    def apply(self, audio, sr, *, seed):
        return add_white_noise(audio, snr_db=self.snr_db, seed=seed)


@dataclass(frozen=True)
class ReverbCondition(Condition):
    preset: str

    def apply(self, audio, sr, *, seed):
        return apply_reverb(audio, sr, preset=self.preset, seed=seed)


@dataclass(frozen=True)
class VolumeCondition(Condition):
    gain_db: float

    def apply(self, audio, sr, *, seed):
        return change_volume(audio, gain_db=self.gain_db, clip=True)


@dataclass(frozen=True)
class BandCondition(Condition):
    preset: str

    def apply(self, audio, sr, *, seed):
        return band_limit(audio, sr, preset=self.preset)


@dataclass(frozen=True)
class CodecCondition(Condition):
    codec: str
    compression_level: float

    def apply(self, audio, sr, *, seed):
        return lossy_recompress(audio, sr, codec=self.codec,
                                compression_level=self.compression_level)


def default_conditions() -> list[Condition]:
    """Full robustness grid used by the experiment runner."""
    return [
        NoiseCondition( "noise",  "SNR 30 dB", snr_db=30.0),
        NoiseCondition( "noise",  "SNR 20 dB", snr_db=20.0),
        NoiseCondition( "noise",  "SNR 10 dB", snr_db=10.0),
        NoiseCondition( "noise",  "SNR  5 dB", snr_db=5.0),
        ReverbCondition("reverb", "RT60 0.2s (small room)",   preset="short"),
        ReverbCondition("reverb", "RT60 0.6s (living room)",  preset="medium"),
        ReverbCondition("reverb", "RT60 1.5s (hall)",         preset="long"),
        VolumeCondition("volume", "-20 dB (quiet)",  gain_db=-20.0),
        VolumeCondition("volume", "-10 dB (soft)",   gain_db=-10.0),
        VolumeCondition("volume", "+6 dB  (clipping)", gain_db=+6.0),
        BandCondition(  "band",   "telephone (300-3400 Hz)", preset="telephone"),
        BandCondition(  "band",   "cheap mic (100-8000 Hz)", preset="cheap_mic"),
        CodecCondition( "codec",  "MP3 high (cl=0.1)",  codec="mp3", compression_level=0.1),
        CodecCondition( "codec",  "MP3 mid  (cl=0.5)",  codec="mp3", compression_level=0.5),
        CodecCondition( "codec",  "MP3 low  (cl=0.9)",  codec="mp3", compression_level=0.9),
    ]
