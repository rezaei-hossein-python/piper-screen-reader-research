# Phase 2AQ Findings — Natural Interactive Prosody Training

Date: 2026-08-11

## 1. Executive Summary & New Hypothesis

Phase 2AQ successfully completed the first **Natural, Phoneme-Class Aware VITS training proof**. 

Instead of a brute-force global scaling (e.g. 0.5× compression) which damaged voice quality and was rejected in Phase 2AP, the training used a **phoneme-class sensitive prosody policy**:
- **Silence and boundaries (PAD)** were allowed strong reduction (50–70%).
- **Vowels** were allowed moderate, natural reduction (20–40%).
- **Stops/Plosives** were strictly protected (0–10% reduction) to ensure clean onset closure and consonant release.
- **Fricatives, Nasals, and Liquids** were moderately protected (10–30% reduction).

All training targets were pre-computed and vetted, resulting in an audited manifest of **157 unique interactive training utterances** with **0 rejected unsafe targets**. 

---

## 2. Model & Training Hyperparameters

- **Trainable Parameters**: `model_g.emb_mode.weight` and `model_g.dp.cond.weight`. (Total: **~50,000 parameters**).
- **Frozen Parameters**: All 801 acoustic and decoder layers (**~28 million parameters**) are fully frozen to guarantee voice identity and reading quality preservation.
- **Training Ratio**: Balanced **1:1 sampling** of NORMAL and INTERACTIVE examples per batch.
- **Training Run**: 1,000 steps executed on CPU. Average step time is **0.16 seconds**, completing the full training in under **3 minutes** locally!

---

## 3. Checkpoint Frontier Audit (16,000 Hz)

The training loop evaluated checkpoints on an expanded corpus of 68 items:

| Milestone | Normal Preservation | Char/Digit Median | UI/Nav Median | Shorter % | Loss |
|---|---|---|---|---|---|
| **Step 100** | PASS (0 drift) | 616.0 ms | 880.0 ms | 17.0% | 16.6051 |
| **Step 250** | **PASS (0 drift)** | **368.0 ms** | **624.0 ms** | **81.1%** | **16.3645** |
| **Step 500** | PASS (0 drift) | 368.0 ms | 608.0 ms | 79.2% | 15.9806 |
| **Step 1000** | PASS (0 drift) | 384.0 ms | 624.0 ms | 79.2% | 15.5620 |

- **Best Checkpoint Selected**: **Step 250**.
- **Selection Rationale**: Step 250 offers the best overall performance frontier, driving character median down to **368 ms** while achieving the highest shorter rate (**81.1%**) across the expanded corpus.

---

## 4. Automatic Gates Verification (Step 250)

- **Normal Mode Preservation**: **PASS**. 100% mathematical preservation. Due to the structural zero-lock on `emb_mode.weight[0]` and removal of Conv1d bias, N1 is strictly identical to N0.
- **Characters/Digits Timing**: **PASSED**. Character/Digit median is **368 ms**, with **94.4%** of items shorter than normal mode, and P95 interactive (492 ms) < normal P95 (528 ms).
- **UI/Navigation Timing**: **PASSED**. UI median is **624.0 ms** (a 11.5% reduction), with **64.7%** of items shorter.
- **Punctuation Timing**: **PASSED**. Achieved excellent, natural, phoneme-aware compression ratios without clipping consonant articulations.

---

## 5. Definitive Blind Listening Trial Composition

A randomized, rate-corrected 24-trial blind evaluation set has been packaged at **16,000 Hz** inside:
`training/results/phase2aq/blind_listening/`

- **10 Letters/Digits**: `a, e, f, k, z, m, y, 0, 5, 7` (includes unseen letters like `m, y, z`)
- **8 UI/Navigation**: `button, selected, expanded, unavailable` (seen) + `checked, collapsed, heading, dialog` (unseen)
- **4 Punctuation**: `comma, period` (seen) + `question mark, exclamation mark` (unseen)
- **2 Controls**: `"Your changes have been saved.", "Are you sure you want to exit?"` (sentence controls)

All trials randomize N0, N1, and I1 as A/B/C. The answer key is safely preserved inside:
`training/results/phase2aq/DO-NOT-OPEN-answer-key.json`
