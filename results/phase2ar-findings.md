# Phase 2AR Findings — Minimal Acoustic Adaptation for Concise Speech

Date: 2026-08-11

## 1. Executive Summary & Strategy
Phase 2AR evaluated whether a **Decoder Residual Adapter** with minimal trainable parameters (**115,200 parameters**, ~0.15% of the model) could resolve the perceptual quality failures of Phase 2AQ (duration-only training) by allowing the acoustic generator to adapt and learn compatible representations for shortened temporal slices.

All other 28 million parameters of VITS were frozen. Normal mode preservation was structurally guaranteed (0.0 ms drift). Training was executed for exactly 250 steps on a microscopic set of 8 difficult items: `F`, `N`, `m`, `b`, `V`, `list`, `link`, `comma`.

---

## 2. Authoritative Perceptual Evaluation

Manual diagnostic listening compared three modes across a 5-item representative set:
- **N0 (Baseline Normal):** Unconditioned Lessac Low
- **I1 (Duration-Only Concise):** From Phase 2AQ (unadapted decoder)
- **I2 (Adapted Concise):** Phase 2AR Step 250 (decoder-adapted)

### A. Acceptability Matrix (5 Core Items)

| Item | N0 Baseline | I1 Duration-Only | I2 Decoder-Adapted | Perceptual Comparison (I1 -> I2) |
|---|---|---|---|---|
| **F** | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | Preserved. Quality remains stable. |
| **N** | ACCEPTABLE | ACCEPTABLE | **NOT ACCEPTABLE** | **REGRESSED**. Sounded overly muffled, losing nasal crispness. |
| **b** | ACCEPTABLE | **NOT ACCEPTABLE** | **ACCEPTABLE** | **IMPROVED**. Plosive release smoothed and pop distortion removed. |
| **list** | ACCEPTABLE | ACCEPTABLE | **NOT ACCEPTABLE** | **REGRESSED**. High-frequency sibilant transitions became slurred. |
| **comma** | ACCEPTABLE | **NOT ACCEPTABLE** | **ACCEPTABLE** | **IMPROVED**. Severe decay crackles/clicks completely smoothed out. |

### B. Summary Score
- **N0 Baseline Acceptance:** 5 / 5 (**100%**)
- **I1 Duration-Only Acceptance:** 3 / 5 (**60%**)
- **I2 Decoder-Adapted Acceptance:** 3 / 5 (**60%**)

---

## 3. Quantitative vs. Perceptual Correlation

While automatic metrics (Mel L1 spectral difference to fast target waveforms) and training loss converged beautifully (L1 spectrogram loss dropped **14.2%** from step 50 to step 250), **Mel-loss reduction does not guarantee net perceptual-quality improvement**. 

*   **Spectral Convergers (`m`, `link`, `comma`):** Mel L1 loss decreased (`comma` dropped from `2.5026` to `2.3435`). This aligned perfectly with the perceptual improvement observed for `comma`, proving that adapting trailing boundary silences succeeds.
*   **Spectral Divergers (`N`, `list`):** Mel L1 loss increased for `N` (`2.7354` -> `3.2591`) and `list` (`1.9137` -> `2.1082`). This matched the perceptual regressions, indicating that the adapter introduced muffled distortion on continuous nasals and complex consonant-vowel transitions.

---

## 4. Phase 2AR Classification & Outcome

### Phase Classification: PARTIAL / Outcome B
> Minimal acoustic adaptation has a real, powerful, item-specific acoustic influence but is not sufficiently general for corpus-scale training.

The decoder residual adapter does not provide a general, stable perceptual quality boost. Instead, it trades off which items succeed: it fixes plosives and punctuation (`b`, `comma`) but compromises nasals and sibilants (`N`, `list`), failing to raise net perceptual acceptability.

---

## 5. Decision & Next Permitted Actions
- **Decision:** **STOP PHASE 2AR**. Do not proceed to Stage 2 (scaled training).
- **Trainable Parameters:** Frozen. Do not expand or unfreeze additional layers.
- **Retraining Fallbacks:** Blocked. Do not automatically test second architectures or unfreeze the whole model.
- **Checkpoints:** Stored privately.
- **Production NVDA Status:** Completely untouched and isolated.

---

## 6. Recommended Next Experiment

### Phase 2AS: Multi-Mode Conditioning via Soft MoE (Mixture of Experts)
*   **Diagnosis:** A single mode embedding $\mathbf{g}_{mode}$ applied to a single adapter weight $\mathbf{W}$ acts as a global linear shift. This shifts representations across all phonemes uniformly, which successfully adapts plosives and boundaries but distorts nasals and continuous sibilants.
*   **Hypothesis:** The model needs *phoneme-class-specific* acoustic adapters.
*   **Proposed Design:** Implement a tiny Mixture of Experts (MoE) block inside `self.dec.cond` or `self.cond`. The experts are specialized for different phoneme classes:
    1. **Expert 1:** Specialized for transient/plosive acoustics.
    2. **Expert 2:** Specialized for continuous nasals/vowels.
    3. **Expert 3:** Specialized for high-frequency sibilants/fricatives.
    A router (conditioned on the current phoneme's class or identity) softly blends these expert embeddings, ensuring that nasal acoustics are adapted differently than stops or boundaries.
*   **Feasibility:** Highly parameter-efficient (< 300,000 parameters total) and compatible with VITS.
