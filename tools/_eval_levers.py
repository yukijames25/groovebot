"""tools/_eval_levers.py — *throwaway* multi-config sweep for the DAMP-S-AG
MIDI route.

The full runner pays pyin's ~78 s/rendition for every pipeline invocation,
so testing N lever combos would burn N × 26 min. This script caches
chroma + pyin features per rendition *once*, then reuses them for every
lever combo, so the marginal cost of an extra combo is just DTW + score
(~1 s per rendition).

Designed to be run interactively (prints a comparison table; writes one
CSV under data/m0p_t2_damp_work). Not in the public CLI surface.

Configs evaluated (Amazing Grace, top-20 by sort order, MIDI reference):

  baseline           full DTW, one-hot pitch
  leverA             + Sakoe-Chiba band (band_rad=0.10)
  leverA_cpitch      + band + continuous-semitone pitch (Lever 2)
  leverA_anchor      + band + GT-non-referenced origin anchor (Lever B)
  leverB_only        no band, but anchor — to isolate Lever B's effect

All other knobs (subseq=False, silence_trim=False) stay at default.
"""
from __future__ import annotations
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from groovebot.align.dtw_align import OfflineDTWAligner
from groovebot.align.features import (
    extract_align_features,
    f0_to_pitch_chroma,
    pitch_contour_feature,
    pyin_f0,
)
from groovebot.align.midi_ref import load_reference_from_midi
from groovebot.align.origin import estimate_origin_offset
from tools.eval_beat import score_beats
from tools.ingest_damp import discover_arrangements


SR = 22050
HOP = 512
MAX_RENDITIONS = 20
ROOT = Path("data/m0p_t2_damp")
OUT_DIR = Path("data/m0p_t2_damp_work")

CONFIGS = [
    # (label, band_rad, pitch_mode, origin_anchor)
    ("baseline",        None,  "one-hot",    False),
    ("leverA",          0.10,  "one-hot",    False),
    ("leverA_cpitch",   0.10,  "continuous", False),
    ("leverA_anchor",   0.10,  "one-hot",    True),
    ("leverB_only",     None,  "one-hot",    True),
]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrangements = discover_arrangements(ROOT, vocal_glob="vocal_*.m4a")
    if not arrangements:
        print(f"no arrangements under {ROOT}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for arr in arrangements:
        if arr.reference_midi is None:
            print(f"skip {arr.arrangement_id}: no reference.midi", file=sys.stderr)
            continue
        print(f"\n=== {arr.arrangement_id} ===")
        midi_ref = load_reference_from_midi(
            arr.reference_midi, sample_rate=SR, hop_length=HOP,
        )
        print(f"  MIDI beats n={len(midi_ref.beats)}  "
              f"note_onsets n={len(midi_ref.note_onsets)}  "
              f"tempo={midi_ref.tempo:.1f} BPM")

        # Feature cache (pyin paid once per rendition).
        cache: dict[str, dict] = {}
        rendi = arr.renditions[:MAX_RENDITIONS]
        for r in rendi:
            t0 = time.perf_counter()
            audio, sr = _load_mono(r.vocal_wav)
            if sr != SR:
                print(f"  skip {r.rendition_id}: sr={sr} != {SR}", file=sys.stderr)
                continue
            chroma = extract_align_features(audio, sr, kind="chroma", hop_length=HOP)
            f0 = pyin_f0(audio, sr, hop_length=HOP)
            pitch_one_hot = f0_to_pitch_chroma(f0)
            pitch_cont = pitch_contour_feature(f0)
            cache[r.rendition_id] = {
                "audio": audio,
                "chroma": chroma,
                "pitch_one_hot": pitch_one_hot,
                "pitch_cont": pitch_cont,
                "dur_sec": len(audio) / sr,
            }
            print(f"  feat {r.rendition_id}: {time.perf_counter() - t0:.1f}s "
                  f"(audio {len(audio) / sr:.1f}s)")

        # Sweep configs against the cached features.
        for label, band_rad, pitch_mode, anchor in CONFIGS:
            aligner = OfflineDTWAligner(
                sample_rate=SR, hop_length=HOP, band_rad=band_rad,
            )
            for r in rendi:
                if r.rendition_id not in cache:
                    continue
                c = cache[r.rendition_id]
                # Estimate the anchor lag once per rendition per config —
                # it depends only on audio + MIDI onsets, not on the DTW
                # path. (Audio shape doesn't change across paths within a
                # config, so this is correct.)
                anchor_lag = 0.0
                if anchor:
                    anchor_lag = estimate_origin_offset(
                        c["audio"], midi_ref.note_onsets,
                        sr=SR, hop_length=HOP,
                    )
                for kind in ("chroma", "pitch"):
                    if kind == "chroma":
                        q = c["chroma"]
                        ref_feats = midi_ref.chroma_template
                    else:
                        if pitch_mode == "continuous":
                            q = c["pitch_cont"]
                            ref_feats = midi_ref.pitch_contour
                        else:
                            q = c["pitch_one_hot"]
                            ref_feats = midi_ref.melody
                    t1 = time.perf_counter()
                    wp = aligner.align(q, ref_feats)
                    recovered = aligner.map_reference_beats(wp, midi_ref.beats)
                    if anchor:
                        recovered = recovered - anchor_lag
                    proc = time.perf_counter() - t1
                    scores = score_beats(
                        track=r.rendition_id, bpm=None,
                        gt=midi_ref.beats, est=recovered,
                        audio_sec=c["dur_sec"], proc_sec=proc,
                    )
                    row = asdict(scores)
                    row["config"] = label
                    row["arrangement_id"] = arr.arrangement_id
                    row["feature_kind"] = kind
                    row["anchor_lag_sec"] = float(anchor_lag)
                    rows.append(row)

    # Summary table — what we paste into CLAUDE.md.
    print("\n=== Summary (mean across renditions) ===")
    print(f"{'config':<18} {'kind':<8} {'n':>3} {'F':>7} {'CMLt':>7} "
          f"{'AMLt':>7} {'RT':>6} {'lag_med[s]':>11}")
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["config"], r["feature_kind"]), []).append(r)
    for (cfg, kind), group in sorted(by.items()):
        f = float(np.mean([g["f_measure"] for g in group]))
        c = float(np.mean([g["cmlt"] for g in group]))
        a = float(np.mean([g["amlt"] for g in group]))
        rt = float(np.mean([g["rt_factor"] for g in group]))
        lag = float(np.median([g["anchor_lag_sec"] for g in group]))
        print(f"{cfg:<18} {kind:<8} {len(group):>3} {f:>7.3f} {c:>7.3f} "
              f"{a:>7.3f} {rt:>6.3f} {lag:>11.3f}")

    out_csv = OUT_DIR / "m0p_t2_levers_per_path.csv"
    if rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
