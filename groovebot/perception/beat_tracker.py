"""
beat_tracker.py — wraps BeatNet (mjhydri) in the spec §5.2 BeatTracker contract.

  update(frames) -> (beat_pos, downbeat, tempo)        # spec §5.2
  process_wav(path) -> ndarray[N, 2]                   # M0 file-batch entry point

Why both? `update(frames)` is the protocol the control loop will eventually
see (M2: mic streaming). `process_wav` is what M0's `tools/eval_beat.py`
actually calls — a whole-file pass, but BeatNet runs causally inside
(`mode='online'`), so it stays NFR-2-compliant.

BeatNet is imported lazily so this module is safe to import on a machine that
has not installed BeatNet (e.g. the M1 local dev box). The wrapper raises a
clear, actionable error the moment you ask it to do work without BeatNet.

Run BeatNet experiments on Colab/Kaggle (see requirements-experiments.txt and
the README M0 section).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


_BEATNET_INSTALL_HINT = (
    "BeatNet is not installed. M0 evaluation runs on Colab/Kaggle, not local.\n"
    "  pip install -r requirements-experiments.txt\n"
    "Local tests use pytest.importorskip('BeatNet') and skip when absent."
)


@dataclass(frozen=True)
class BeatEvent:
    """One beat output by the tracker."""
    time: float          # seconds from start of audio
    beat_in_bar: int     # 1..beats_per_bar; 1 == downbeat
    is_downbeat: bool


def _require_beatnet():
    """Lazy import. Raises a clear RuntimeError if BeatNet isn't installed."""
    try:
        from BeatNet.BeatNet import BeatNet  # type: ignore
        return BeatNet
    except Exception as e:  # ImportError or transitive failure (madmom etc.)
        raise RuntimeError(_BEATNET_INSTALL_HINT) from e


class BeatTrackerPerception:
    """Causal beat tracker built on BeatNet (online mode + particle filter).

    Conforms to spec §5.2 BeatTracker:  `update(frames) -> (beat_pos, downbeat, tempo)`.

    Two entry points are exposed:
      - `process_wav(path)` — what `tools/eval_beat.py` uses. Runs BeatNet over
        the whole file in `online` mode (still causal inside) and returns the
        full list of beats. Good for offline accuracy + latency measurement.
      - `update(frames)`    — the streaming contract. Buffers frames and asks
        BeatNet for the latest estimate. Not optimised; the real low-latency
        path will be wired in M2 alongside the mic input.
    """

    def __init__(self, sample_rate: int = 22050, beats_per_bar: int = 4):
        self.sample_rate = sample_rate
        self.beats_per_bar = beats_per_bar
        self._estimator = None              # lazily created
        self._buf: list[float] = []         # for update(frames)
        self._last_beat_idx: int = -1       # for update(frames) bookkeeping
        # Last value returned by update(), so we have something to return when
        # the buffer is too short for BeatNet to make a decision.
        self._last_state: tuple[float, bool, float] = (0.0, False, 0.0)

    # -- batch entry point used by tools/eval_beat.py --------------------------
    def process_wav(self, path: str) -> list[BeatEvent]:
        """Return all beats BeatNet reports for a WAV file."""
        BeatNet = _require_beatnet()
        if self._estimator is None:
            self._estimator = BeatNet(
                1,
                mode="online",
                inference_model="PF",
                plot=[],
                thread=False,
            )
        raw = self._estimator.process(path)
        return _to_beat_events(raw)

    # -- streaming entry point (spec §5.2 contract) ----------------------------
    def update(self, frames) -> tuple[float, bool, float]:
        """Buffer `frames` and return the latest (beat_pos, downbeat, tempo).

        Implementation note: this re-runs BeatNet on the accumulated buffer
        each call. Correct, but O(buffer_length) per call. The mic-streaming
        path in M2 will replace this with a true streaming hook.
        """
        import numpy as np
        BeatNet = _require_beatnet()
        if self._estimator is None:
            self._estimator = BeatNet(
                1, mode="online", inference_model="PF", plot=[], thread=False,
            )
        chunk = np.asarray(frames, dtype=np.float32).ravel()
        self._buf.extend(chunk.tolist())
        # BeatNet expects a path; write the buffer to a temp wav each call.
        # Costly — fine for the protocol-conformance smoke; not the real path.
        import tempfile, soundfile as sf, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, np.asarray(self._buf, dtype=np.float32), self.sample_rate)
            raw = self._estimator.process(tmp)
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        events = _to_beat_events(raw)
        if not events:
            return self._last_state
        last = events[-1]
        # beat_pos = total beat count from the start (downbeats restart bar).
        beat_pos = float(len(events))
        tempo = _tempo_bpm_from_events(events) or self._last_state[2]
        self._last_state = (beat_pos, last.is_downbeat, tempo)
        return self._last_state


# --------------------------------------------------------------------------- #
# Helpers (kept module-level so eval_beat / tests can reuse them without
# instantiating BeatTrackerPerception).
# --------------------------------------------------------------------------- #
def _to_beat_events(raw) -> list[BeatEvent]:
    """BeatNet returns ndarray of shape (N, 2): [time_sec, beat_in_bar].

    beat_in_bar is 1..beats_per_bar (1 == downbeat).
    """
    if raw is None:
        return []
    out: list[BeatEvent] = []
    for row in raw:
        t = float(row[0])
        b = int(row[1])
        out.append(BeatEvent(time=t, beat_in_bar=b, is_downbeat=(b == 1)))
    return out


def _tempo_bpm_from_events(events: list[BeatEvent]) -> Optional[float]:
    """Median IBI -> BPM. None if fewer than 2 beats."""
    if len(events) < 2:
        return None
    iois = [events[i + 1].time - events[i].time for i in range(len(events) - 1)]
    iois = [x for x in iois if x > 0]
    if not iois:
        return None
    iois.sort()
    mid = iois[len(iois) // 2]
    return 60.0 / mid if mid > 0 else None
