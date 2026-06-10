"""tools/_diagnose_damp.py — *throwaway* diagnostic for the DAMP-S-AG MIDI run.

Goal: tell apart task-difficulty (rubato hymn, amateur singers) from
methodology artifact (timeline / GT / feature issues). Touches nothing in
the public pipeline. Run on three pre-selected renditions (low / mid /
high F-measure) drawn from the 20-rendition subset run.

Outputs:
  - structured text findings to stdout
  - 3 plots under data/m0p_t2_damp_diag/
        <rendition>_offset_sweep.png   GT-offset vs F curve, both paths
        <rendition>_warp.png           DTW warp paths (chroma + pitch) over the diagonal
        <rendition>_onsets.png         onset strength + GT + recovered beats overlay (chroma path)
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import soundfile as sf

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import (
    extract_align_features,
    f0_to_pitch_chroma,
    pyin_f0,
)
from groovebot.align.midi_ref import load_reference_from_midi


# ---------- Config -----------------------------------------------------------
SR = 22050
HOP = 512
MIDI_PATH = Path("data/m0p_t2_damp/amazing_grace/reference.midi")
ARR_DIR = Path("data/m0p_t2_damp/amazing_grace")
DIAG_DIR = Path("data/m0p_t2_damp_diag")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Picked from data/m0p_t2_damp_work/m0p_t2_damp_per_path.csv (avg F).
TARGETS = [
    ("LOW",  "100125241_14688700"),
    ("MID",  "100128262_13604411"),
    ("HIGH", "100030930_13364402"),
]

RMS_FRAME_LEN = 2048
RMS_HOP = 512
RMS_DB_THRESHOLD = -45.0   # dBFS-ish; anything below counts as silence.
OFFSET_RANGE = np.arange(-1.0, 1.0 + 1e-9, 0.05)


@dataclass
class RenditionDiagnostics:
    label: str
    rendition_id: str
    duration_s: float
    leading_silence_s: float
    trailing_silence_s: float
    first_sung_s: float
    silence_frac: float
    voiced_frac: float
    f0_pc_sparse_frac: float
    midi_melody_sparse_frac: float


def rms_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(
        y=audio, frame_length=RMS_FRAME_LEN, hop_length=RMS_HOP,
    )[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=RMS_HOP)
    return t, rms


def silence_boundaries(audio: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """Return (leading_silence_s, trailing_silence_s, first_sung_s, silence_frac)."""
    t, rms = rms_envelope(audio, sr)
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    voiced = db > RMS_DB_THRESHOLD
    if not voiced.any():
        dur = len(audio) / sr
        return dur, 0.0, dur, 1.0
    first_voiced_frame = int(np.argmax(voiced))
    last_voiced_frame = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    leading = float(t[first_voiced_frame])
    trailing = float(len(audio) / sr - t[last_voiced_frame])
    first_sung = float(t[first_voiced_frame])
    silence_frac = float(1.0 - voiced.mean())
    return leading, trailing, first_sung, silence_frac


def warp_diag_deviation(wp: np.ndarray, T_query: int, T_ref: int) -> tuple[float, float]:
    """Median + max absolute deviation of the warp path from a unit-slope
    diagonal scaled to (T_query, T_ref). Returns frames in query-axis units."""
    q = wp[:, 0].astype(float)
    r = wp[:, 1].astype(float)
    # Project r onto the query axis with slope T_query/T_ref
    expected_q = r * (T_query / T_ref)
    dev = np.abs(q - expected_q)
    return float(np.median(dev)), float(np.max(dev))


def f_at_offset(recovered: np.ndarray, gt_beats: np.ndarray, offset: float) -> float:
    """Re-score with GT shifted by `offset` seconds."""
    gt_shift = gt_beats + offset
    gt_trim = mir_eval.beat.trim_beats(np.asarray(gt_shift, dtype=float))
    est_trim = mir_eval.beat.trim_beats(np.asarray(recovered, dtype=float))
    if len(gt_trim) == 0 or len(est_trim) == 0:
        return 0.0
    return float(mir_eval.beat.f_measure(gt_trim, est_trim))


def trim_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Drop leading + trailing silence based on RMS threshold. Returns
    (trimmed_audio, leading_trim_seconds) so the caller can shift GT."""
    t, rms = rms_envelope(audio, sr)
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    voiced = db > RMS_DB_THRESHOLD
    if not voiced.any():
        return audio.copy(), 0.0
    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    s = int(t[first] * sr)
    e = int(min(len(audio), t[last] * sr + RMS_FRAME_LEN))
    return audio[s:e].copy(), float(s / sr)


def run_one(midi_ref, label: str, rendition_id: str) -> dict:
    print(f"\n=== {label}: {rendition_id} ===")
    wav_path = ARR_DIR / f"vocal_{rendition_id}.m4a"
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr

    # Timeline / silence
    leading, trailing, first_sung, silence_frac = silence_boundaries(audio, sr)
    midi_first_note = float(np.min([
        n.start for inst in __import__("pretty_midi").PrettyMIDI(str(MIDI_PATH)).instruments
        for n in inst.notes
    ]))
    midi_end = float(midi_ref.beats[-1] if len(midi_ref.beats) else 0.0)

    print(f"  dur(rendition)={dur:.2f}s  leading_silence={leading:.2f}s  "
          f"trailing_silence={trailing:.2f}s")
    print(f"  first_sung_at={first_sung:.2f}s  "
          f"midi_first_note_at={midi_first_note:.3f}s  "
          f"midi_last_beat_at={midi_end:.2f}s")
    print(f"  silence_frac={silence_frac:.3f}")

    # Features
    t0 = time.perf_counter()
    chroma_q = extract_align_features(audio, sr, kind="chroma", hop_length=HOP)
    f0 = pyin_f0(audio, sr, hop_length=HOP)
    pitch_q = f0_to_pitch_chroma(f0)
    voiced_frac = float(np.mean(np.isfinite(f0)))
    f0_pc_sparse = float(np.mean(pitch_q.sum(axis=0) > 0))
    midi_mel_sparse = float(np.mean(midi_ref.melody.sum(axis=0) > 0))
    print(f"  chroma_q.shape={chroma_q.shape}  pitch_q.shape={pitch_q.shape}  "
          f"midi.melody.shape={midi_ref.melody.shape}  "
          f"midi.chroma.shape={midi_ref.chroma_template.shape}")
    print(f"  voiced_frac(F0)={voiced_frac:.3f}  "
          f"f0_pc_active_frac={f0_pc_sparse:.3f}  "
          f"midi_melody_active_frac={midi_mel_sparse:.3f}")
    print(f"  feature time={time.perf_counter() - t0:.1f}s")

    # Align both paths
    aligner = OfflineDTWAligner(sample_rate=sr, hop_length=HOP)
    wp_c = aligner.align(chroma_q, midi_ref.chroma_template)
    wp_p = aligner.align(pitch_q, midi_ref.melody)
    rec_c = aligner.map_reference_beats(wp_c, midi_ref.beats)
    rec_p = aligner.map_reference_beats(wp_p, midi_ref.beats)

    # Warp diagonality
    Tq_c, Tr_c = chroma_q.shape[1], midi_ref.chroma_template.shape[1]
    Tq_p, Tr_p = pitch_q.shape[1], midi_ref.melody.shape[1]
    med_c, max_c = warp_diag_deviation(wp_c, Tq_c, Tr_c)
    med_p, max_p = warp_diag_deviation(wp_p, Tq_p, Tr_p)
    print(f"  chroma DTW path deviation: median={med_c:.1f}f  max={max_c:.1f}f")
    print(f"  pitch  DTW path deviation: median={med_p:.1f}f  max={max_p:.1f}f")

    # GT-offset sweep (chroma + pitch)
    f_c_at0 = f_at_offset(rec_c, midi_ref.beats, 0.0)
    f_p_at0 = f_at_offset(rec_p, midi_ref.beats, 0.0)
    f_c_curve = np.array([f_at_offset(rec_c, midi_ref.beats, o) for o in OFFSET_RANGE])
    f_p_curve = np.array([f_at_offset(rec_p, midi_ref.beats, o) for o in OFFSET_RANGE])
    best_off_c = float(OFFSET_RANGE[int(np.argmax(f_c_curve))])
    best_off_p = float(OFFSET_RANGE[int(np.argmax(f_p_curve))])
    print(f"  F at offset=0:  chroma={f_c_at0:.3f}  pitch={f_p_at0:.3f}")
    print(f"  best offset (chroma)={best_off_c:+.2f}s  F={f_c_curve.max():.3f}  "
          f"gain=+{f_c_curve.max() - f_c_at0:.3f}")
    print(f"  best offset (pitch) ={best_off_p:+.2f}s  F={f_p_curve.max():.3f}  "
          f"gain=+{f_p_curve.max() - f_p_at0:.3f}")

    # Trim leading/trailing silence + realign
    audio_trim, leading_trim_s = trim_audio(audio, sr)
    if len(audio_trim) > sr * 5:    # need >= 5s of audio
        chroma_q2 = extract_align_features(audio_trim, sr, kind="chroma", hop_length=HOP)
        f02 = pyin_f0(audio_trim, sr, hop_length=HOP)
        pitch_q2 = f0_to_pitch_chroma(f02)
        # Build a beats-on-rendition-timeline by shifting MIDI beats back by
        # the leading trim duration (the trimmed audio's t=0 == leading_trim_s
        # in the original).
        gt_trimmed_axis = midi_ref.beats - leading_trim_s
        gt_trimmed_axis = gt_trimmed_axis[gt_trimmed_axis >= 0]
        wp_c2 = aligner.align(chroma_q2, midi_ref.chroma_template)
        wp_p2 = aligner.align(pitch_q2, midi_ref.melody)
        rec_c2 = aligner.map_reference_beats(wp_c2, midi_ref.beats) - leading_trim_s
        rec_p2 = aligner.map_reference_beats(wp_p2, midi_ref.beats) - leading_trim_s
        f_c2 = f_at_offset(rec_c2[rec_c2 >= 0],
                           gt_trimmed_axis, 0.0)
        f_p2 = f_at_offset(rec_p2[rec_p2 >= 0],
                           gt_trimmed_axis, 0.0)
        print(f"  trim+realign: leading_trim={leading_trim_s:.2f}s  "
              f"trimmed_dur={len(audio_trim)/sr:.2f}s")
        print(f"  trim+realign F: chroma={f_c2:.3f} "
              f"(was {f_c_at0:.3f}, delta={f_c2 - f_c_at0:+.3f})  "
              f"pitch={f_p2:.3f} (was {f_p_at0:.3f}, delta={f_p2 - f_p_at0:+.3f})")
    else:
        f_c2 = f_p2 = float("nan")
        print("  trim+realign skipped (trimmed audio too short)")

    # PNGs
    _plot_warp(wp_c, wp_p, midi_ref.chroma_template.shape[1],
               midi_ref.melody.shape[1], chroma_q.shape[1], pitch_q.shape[1],
               DIAG_DIR / f"{rendition_id}_warp.png",
               title=f"{label} {rendition_id} — DTW warp")
    _plot_offset_sweep(OFFSET_RANGE, f_c_curve, f_p_curve,
                       DIAG_DIR / f"{rendition_id}_offset_sweep.png",
                       title=f"{label} {rendition_id} — GT-offset sweep")
    _plot_onsets(audio, sr, midi_ref.beats, rec_c,
                 DIAG_DIR / f"{rendition_id}_onsets.png",
                 title=f"{label} {rendition_id} — onset strength + beats (chroma)")

    return {
        "label": label,
        "rendition_id": rendition_id,
        "duration_s": dur,
        "leading_silence_s": leading,
        "trailing_silence_s": trailing,
        "first_sung_s": first_sung,
        "silence_frac": silence_frac,
        "midi_first_note_s": midi_first_note,
        "midi_last_beat_s": midi_end,
        "voiced_frac": voiced_frac,
        "f0_pc_sparse_frac": f0_pc_sparse,
        "midi_melody_sparse_frac": midi_mel_sparse,
        "warp_chroma_dev_median_frames": med_c,
        "warp_chroma_dev_max_frames": max_c,
        "warp_pitch_dev_median_frames": med_p,
        "warp_pitch_dev_max_frames": max_p,
        "F_chroma_at0": f_c_at0,
        "F_pitch_at0": f_p_at0,
        "best_offset_chroma_s": best_off_c,
        "best_F_chroma": float(f_c_curve.max()),
        "best_offset_pitch_s": best_off_p,
        "best_F_pitch": float(f_p_curve.max()),
        "F_chroma_after_trim": f_c2,
        "F_pitch_after_trim": f_p2,
    }


def _plot_warp(wp_c, wp_p, Tr_c, Tr_p, Tq_c, Tq_p, path, title):
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
    ax_c.plot(wp_c[:, 1], wp_c[:, 0], color="#1f77b4", linewidth=0.7)
    ax_c.plot([0, Tr_c], [0, Tq_c], "--", color="#888", linewidth=0.8)
    ax_c.set_title("chroma DTW")
    ax_c.set_xlabel("ref frame")
    ax_c.set_ylabel("query frame")
    ax_p.plot(wp_p[:, 1], wp_p[:, 0], color="#d62728", linewidth=0.7)
    ax_p.plot([0, Tr_p], [0, Tq_p], "--", color="#888", linewidth=0.8)
    ax_p.set_title("pitch DTW")
    ax_p.set_xlabel("ref frame")
    ax_p.set_ylabel("query frame")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_offset_sweep(offsets, f_c, f_p, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(offsets, f_c, color="#1f77b4", label="chroma")
    ax.plot(offsets, f_p, color="#d62728", label="pitch")
    ax.axvline(0.0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GT offset [s]")
    ax.set_ylabel("F-measure")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_onsets(audio, sr, gt, est, path, title):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP)
    t = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, onset_env / max(onset_env.max(), 1e-9),
            color="#444", linewidth=0.6, label="onset strength (normalised)")
    for x in gt:
        ax.axvline(x, color="#1f77b4", linewidth=0.8, alpha=0.5)
    for x in est:
        ax.axvline(x, color="#d62728", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_xlabel("time [s]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    midi_ref = load_reference_from_midi(MIDI_PATH, sample_rate=SR, hop_length=HOP)
    print(f"MIDI beats: n={len(midi_ref.beats)}  "
          f"first={midi_ref.beats[0]:.3f}s  last={midi_ref.beats[-1]:.3f}s  "
          f"tempo={midi_ref.tempo:.2f} BPM")
    results = []
    for label, rid in TARGETS:
        results.append(run_one(midi_ref, label, rid))

    summary = DIAG_DIR / "diag_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    print(f"plots under {DIAG_DIR}")


if __name__ == "__main__":
    main()
