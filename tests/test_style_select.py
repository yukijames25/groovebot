"""End-to-end test for groovebot.style.select (GrooveStyleSelector).

Uses synthetic audio so this stays CPU-only and runs without GTZAN /
Demucs / GPU. The model is randomly initialised, so we only assert
contract-level properties (output shape, label vocabulary, value
ranges) — not classification accuracy. Accuracy is the job of
`experiments/train_style.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from groovebot.style.attributes import (
    arousal_bucket,
    estimate_arousal,
    estimate_tempo,
)
from groovebot.style.model import GENRES, MOODS
from groovebot.style.select import GrooveStyle, GrooveStyleSelector
from groovebot.style.table import MOVES


SR = 22050


def _synth_click_track(sr: int, duration_sec: float, bpm: float) -> np.ndarray:
    """Pulse train at the given BPM, useful for tempo + onset density."""
    n = int(sr * duration_sec)
    sig = np.zeros(n, dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    click_len = max(1, sr // 200)  # 5 ms clicks
    for start in range(0, n, period):
        end = min(n, start + click_len)
        sig[start:end] = 1.0
    return sig


def _synth_quiet_tone(sr: int, duration_sec: float, amp: float = 0.01) -> np.ndarray:
    t = np.arange(int(sr * duration_sec)) / sr
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_attributes_tempo_close_to_synth_bpm():
    audio = _synth_click_track(SR, duration_sec=6.0, bpm=120.0)
    bpm = estimate_tempo(audio, SR)
    # librosa tempo can octave-error; allow either 60 or 120 ±10%
    assert (108 <= bpm <= 132) or (54 <= bpm <= 66)


def test_attributes_arousal_higher_for_dense_clicks():
    quiet = _synth_quiet_tone(SR, 4.0, amp=0.005)
    dense = _synth_click_track(SR, 4.0, bpm=180.0)
    a_q = estimate_arousal(quiet, SR)
    a_d = estimate_arousal(dense, SR)
    assert a_d > a_q


def test_arousal_bucket_thresholds():
    assert arousal_bucket(0.0) == "low"
    assert arousal_bucket(0.32) == "low"
    assert arousal_bucket(0.33) == "mid"
    assert arousal_bucket(0.65) == "mid"
    assert arousal_bucket(0.66) == "high"
    assert arousal_bucket(1.0) == "high"


def test_selector_end_to_end_returns_groove_style():
    selector = GrooveStyleSelector()
    audio = _synth_click_track(SR, duration_sec=5.0, bpm=120.0)
    style = selector.select(audio, SR)
    assert isinstance(style, GrooveStyle)
    assert style.move in MOVES
    assert style.genre in GENRES
    assert style.mood in MOODS
    assert 0.0 <= style.intensity <= 1.0
    assert 0.0 <= style.arousal <= 1.0
    assert style.arousal_bucket in {"low", "mid", "high"}
    assert style.tempo_bpm >= 0.0


def test_selector_probs_dicts_have_all_labels():
    selector = GrooveStyleSelector()
    audio = _synth_click_track(SR, duration_sec=5.0, bpm=120.0)
    style = selector.select(audio, SR)
    assert set(style.genre_probs.keys()) == set(GENRES)
    assert set(style.mood_probs.keys()) == set(MOODS)
    # Softmax → sums close to 1
    assert abs(sum(style.genre_probs.values()) - 1.0) < 1e-4
    assert abs(sum(style.mood_probs.values()) - 1.0) < 1e-4


def test_as_text_renders():
    selector = GrooveStyleSelector()
    audio = _synth_click_track(SR, duration_sec=5.0, bpm=120.0)
    style = selector.select(audio, SR)
    text = style.as_text()
    assert style.move in text
    assert style.genre in text


def test_selector_handles_short_window():
    # 2 s is shorter than spec's 5-10 s window but should still produce a
    # valid contract output, not crash.
    selector = GrooveStyleSelector()
    audio = _synth_click_track(SR, duration_sec=2.0, bpm=140.0)
    style = selector.select(audio, SR)
    assert style.move in MOVES


def _mock_panns_emb(audio, sr):
    import numpy as np
    from groovebot.style.backbone import EMBEDDING_DIM
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-9))
    rng = np.random.default_rng(int(rms * 1e6) % (2**31))
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def test_selector_embedding_mode_end_to_end():
    from groovebot.style.backbone import PannsBackbone, EMBEDDING_DIM
    from groovebot.style.model import StyleHead

    backbone = PannsBackbone(_inference_fn=_mock_panns_emb)
    head = StyleHead(emb_dim=EMBEDDING_DIM)
    sel = GrooveStyleSelector(backbone=backbone, head=head)
    assert sel.mode == "embedding"
    audio = _synth_click_track(SR, 5.0, bpm=120.0)
    style = sel.select(audio, SR)
    assert style.move in MOVES
    assert style.genre in GENRES
    assert style.mood in MOODS
    assert 0.0 <= style.intensity <= 1.0


def test_selector_rejects_model_plus_backbone():
    from groovebot.style.backbone import PannsBackbone, EMBEDDING_DIM
    from groovebot.style.model import StyleCNN, StyleHead
    import pytest

    backbone = PannsBackbone(_inference_fn=_mock_panns_emb)
    head = StyleHead(emb_dim=EMBEDDING_DIM)
    with pytest.raises(ValueError, match="not both"):
        GrooveStyleSelector(
            model=StyleCNN(n_mels=64), backbone=backbone, head=head,
        )


def test_selector_rejects_partial_v3_args():
    from groovebot.style.backbone import PannsBackbone
    import pytest

    backbone = PannsBackbone(_inference_fn=_mock_panns_emb)
    with pytest.raises(ValueError, match="together"):
        GrooveStyleSelector(backbone=backbone)  # missing head
