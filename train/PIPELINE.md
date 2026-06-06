# B-3 training pipeline (offline, free, Kaggle)

Goal: learn `voice features + beat → robot groove`, in the "fun karaoke" style,
so the body in `robot/groovebot.urdf` moves *with* the singer. This runs offline
on data — the physics sim is **not** in the training loop.

## Why this is tractable despite "no paired data"

There is no dataset of (a cappella voice → robot groove). We bootstrap, the same
way singing-beat-tracking did:

1. Take **AIST++** (full songs ↔ 3D dancer motion, 10 genres — style matches
   "fun/energetic", which is why we chose this target).
2. **Demucs** → separate the vocal stem. The dancer's motion now has a *vocal*
   paired with it, and inherits the original track's beat grid.
3. Train conditioned on the **vocal stem**, not the full mix.

Honest caveat (design risk #2): the dancer originally moved to the *full* track
(drums/bass), not the vocal. So let the robust **beat-phase channel** (from the
singing beat tracker) carry timing, and let the learned model shape **style**.
Don't ask the generator to invent timing the voice can't provide.

## Steps

1. **Data prep (CPU, local or Kaggle)**
   - Download AIST++ motion (SMPL params) + audio. Skip the multi-view video.
   - Demucs on each track → vocal stems.
   - Run the beat tracker → per-frame beat phase + downbeat.

2. **Feature extraction (GPU, Kaggle)**
   - Voice → WavLM/HuBERT embeddings (frame-aligned).
   - Voice → arousal/valence (MER head; the M2 model).
   - Stack: [beat phase, downbeat, arousal/valence, voice embedding] per frame.

3. **Retarget SMPL → robot (the real ML-engineering crux, CPU)**
   - Map AIST++ upper-body joints (spine, neck, shoulders, elbows) to the 10
     joint angles in `groovebot.urdf`. This defines the model's output space.
   - Clamp to the URDF joint limits → guarantees physically feasible targets.

4. **Train the groove generator (GPU, Kaggle)**
   - VQ-VAE: learn a codebook of short, feasible groove primitives
     (à la Bailando's "choreographic memory").
   - Autoregressive transformer: predict the next codebook token from the
     conditioning features. Predict N future steps (FACT trick) to avoid
     "freezing". Save checkpoints to Kaggle Datasets / Drive (sessions reset).

5. **Deploy**
   - Export the model; load it in `groovebot/groove.py` behind the same
     `compute(...)` signature. The sim/robot pipeline is unchanged.

## Ablations to plan from day one
- lyrics-singing vs **humming** (humming may be out-of-distribution for speech
  SSL — pitch/energy features may matter more).
- learned model (M3) vs hand-authored (M1) vs **desynchronised** control — the
  desync condition is what proves the synchrony effect in the user study.
```
