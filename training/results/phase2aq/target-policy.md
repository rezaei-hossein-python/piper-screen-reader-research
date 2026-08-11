# Phase 2AQ Target Policy: Phoneme-Class Aware Duration Compression

Date: 2026-08-11

## 1. Motivation

The previous brute-force duration compression policy (universally scaling predicted normal durations by a constant factor of 0.5) resulted in aggressive, robotic, and unacceptable interactive voice quality on letters, digits, and UI micro-speech.

The **Natural Interactive Prosody** hypothesis states that different phonetic classes tolerate different levels of compression before articulation collapses.

---

## 2. Phoneme Class Classification & Speed Rules

We categorized all default Piper VITS phoneme tokens and established the following rules:

### A. Silence, Boundaries, and PAD (Strong Reduction: 50% – 70%)
- **Phonemes**: `_`, `^`, `$`, ` `, `PAD`, `BOS`, `EOS`.
- **Policy**: Allowed deep compression since these elements do not contain articulation.
- **Rule**: 
  - T1: `dur * 0.7`
  - T2 (Standard): `dur * 0.5`
  - T3: `dur * 0.3`
- **Floor**: 1 frame.

### B. Vowels (Moderate Reduction: 10% – 40%)
- **Phonemes**: standard vowels (`a`, `e`, `i`, `o`, `u`, `Ã¦`, `É™`, `É›`, etc.) and diphthongs (`eÉª`, `aÉª`, `oÊŠ`).
- **Policy**: Moderate vowel sustain reduction represents natural concise prosody.
- **Rule**:
  - T1: `dur * 0.9`
  - T2 (Standard): `dur * 0.8`
  - T3: `dur * 0.6`
- **Floor**: 2 frames (protects phoneme boundary transition).

### C. Stops / Plosives (Strict Protection: 0% – 10% Reduction)
- **Phonemes**: `p`, `b`, `t`, `d`, `k`, `g`, `Ê”`, etc.
- **Policy**: Closing and release phases must be fully preserved to avoid severe slurring or clipping.
- **Rule**:
  - T1/T2 (Standard): `dur * 1.0` (0% reduction / fully protected)
  - T3: `dur * 0.9`
- **Floor**: 2 frames.

### D. Fricatives (High Protection: 0% – 20% Reduction)
- **Phonemes**: `f`, `v`, `s`, `z`, `Êƒ`, `Ê’`, `Î²`, `Î¸`, etc.
- **Policy**: Fricative noise must maintain sufficient duration to avoid popping or identity loss.
- **Rule**:
  - T1: `dur * 1.0`
  - T2 (Standard): `dur * 0.9`
  - T3: `dur * 0.8`
- **Floor**: 2 frames.

### E. Nasals, Liquids, and Glides (Moderate Protection: 5% – 30% Reduction)
- **Phonemes**: nasals (`m`, `n`), liquids (`l`, `r`), glides (`j`, `w`).
- **Policy**: Consonantal transitions must remain clear.
- **Rule**:
  - T1: `dur * 0.95`
  - T2 (Standard): `dur * 0.85`
  - T3: `dur * 0.70`
- **Floor**: 2 frames.

### F. Stress & Control Marks (High Protection)
- **Phonemes**: `Ëˆ` (primary stress), `ËŒ` (secondary stress), `Ë` (length sign).
- **Rule**: 
  - T1: `dur * 1.0`
  - T2 (Standard): `dur * 0.9`
  - T3: `dur * 0.8`
- **Floor**: 1 frame.
