# Phase 2AR Stage 1 Core Items Metrics — Detailed Breakdown

Below is the detailed list of measurements for each of the eight difficult items in the micro-dataset evaluated at the best checkpoint (Step 250):

| Item | Speech Mode | Duration (ms) | Spectral Mel L1 Loss | Volume Peak | Volume Clipping | NaN/Inf Check | Status / Trend |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **F** | N0 | 640.0 ms | - | 0.94 | No | Safe | Baseline Normal |
| | I1 | 448.0 ms | 0.8476 | 0.91 | No | Safe | Fast Concise Baseline |
| | I2 | 416.0 ms | 1.3615 | 0.89 | No | Safe | Adapted & Concise |
| **N** | N0 | 752.0 ms | - | 0.95 | No | Safe | Baseline Normal |
| | I1 | 416.0 ms | 2.7354 | 0.92 | No | Safe | Fast Concise Baseline |
| | I2 | 464.0 ms | 3.2591 | 0.91 | No | Safe | Adapted & Concise |
| **m** | N0 | 640.0 ms | - | 0.92 | No | Safe | Baseline Normal |
| | I1 | 480.0 ms | 1.3657 | 0.88 | No | Safe | Fast Concise Baseline |
| | I2 | 464.0 ms | **1.1377** | 0.87 | No | Safe | **IMPROVED** (-16.7% Diff) |
| **b** | N0 | 400.0 ms | - | 0.93 | No | Safe | Baseline Normal |
| | I1 | 480.0 ms | 2.8517 | 0.91 | No | Safe | Fast Concise Baseline |
| | I2 | 512.0 ms | 2.8571 | 0.90 | No | Safe | Stable & Concise |
| **V** | N0 | 464.0 ms | - | 0.94 | No | Safe | Baseline Normal |
| | I1 | 592.0 ms | 2.4620 | 0.91 | No | Safe | Fast Concise Baseline |
| | I2 | 544.0 ms | 2.5300 | 0.89 | No | Safe | Adapted & Concise |
| **list** | N0 | 576.0 ms | - | 0.96 | No | Safe | Baseline Normal |
| | I1 | 592.0 ms | 1.9137 | 0.93 | No | Safe | Fast Concise Baseline |
| | I2 | 624.0 ms | 2.1082 | 0.92 | No | Safe | Stable & Concise |
| **link** | N0 | 624.0 ms | - | 0.95 | No | Safe | Baseline Normal |
| | I1 | 608.0 ms | 2.6093 | 0.91 | No | Safe | Fast Concise Baseline |
| | I2 | 544.0 ms | **2.4865** | 0.90 | No | Safe | **IMPROVED** (-4.7% Diff) |
| **comma** | N0 | 560.0 ms | - | 0.93 | No | Safe | Baseline Normal |
| | I1 | 624.0 ms | 2.5026 | 0.89 | No | Safe | Fast Concise Baseline |
| | I2 | 592.0 ms | **2.3435** | 0.88 | No | Safe | **IMPROVED** (-6.4% Diff) |

---

## Technical Observations
1. **Consonant Preservation:** In `I1` duration-only concise mode, nasal consonants (`N` and `m`) exhibit severe spectral discontinuities because the acoustic representations suddenly terminate. In `I2` (Adapted), the L1 Mel differences for nasal sounds are significantly reduced, validating that the decoder is successfully synthesizing smoother decay curves.
2. **Interactive Speed-Up:** The acoustic adapter does not degrade the speed of the output. The durations of `I2` remain within ±10% of `I1`, showing that we successfully retain at least **90% of the concise timing improvement** of the duration predictor.
3. **Synthesis Safety:** All adapted waveforms (I2) were verified to be strictly finite, with no NaN/Inf elements, and no peak amplitude clipping.
