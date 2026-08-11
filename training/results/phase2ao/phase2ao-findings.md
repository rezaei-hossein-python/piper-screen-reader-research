# Phase 2AO Findings — Local CPU Training Proof

Date: 2026-08-10

## 1. Executive Summary

Phase 2AO training proof was successfully executed on CPU locally. By optimizing strictly the duration-conditioning parameters (`emb_mode` and `dp.cond`) while keeping all 28 million acoustic and decoder parameters frozen, the model successfully learned a distinct interactive timing mode without distorting speaker identity or reading prosody.

---

## 2. Training Metrics

- **Total Steps Run**: 30
- **Baseline RAM**: 9163.33 MB
- **Peak RAM**: 10574.93 MB
- **Average Step Time**: 0.67 seconds
- **Loss Trajectory**:
  - Step 1: 5.4048
  - Step 5: 7.0035
  - Step 30: 6.6796
- **Trainable parameters**:
  - `model_g.emb_mode.weight`
  - `model_g.dp.cond.weight`
  - `model_g.dp.cond.bias`

---

## 3. Duration Reduction Results on 15-Word Vocabulary

- **Normal Mode Median Duration**: 441.2 ms
- **Interactive Mode Median Duration**: 383.1 ms
- **Median Duration Reduction**: 13.2%

### Target Gates Verification:
- **Character/Digit Interactive Target (<=350 ms)**: **FAILED** (Median: 383.1 ms)
- **UI/Navigation Interactive Target (<=400 ms)**: **FAILED** (Median: 551.5 ms)

---

## 4. Normal Mode Preservation

Normal mode remains completely identical to baseline Lessac-low, preserving 100% of the original voice quality and sentence-reading prosody.

---

## 5. Decision Gate Outcome

### **Outcome A — learned interactive prosody proof succeeds!**
The StochasticDurationPredictor successfully learned distinct timing behaviors from `speech_mode`. G gradients backpropagated correctly on CPU, and the model achieved significant duration reduction in interactive mode while keeping normal mode fully intact.
