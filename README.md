# Piper screen-reader research

Isolated R&D on adapting Piper/VITS for low-latency, high-quality screen-reader
speech. This repository does **not** modify or integrate with the production NVDA
Piper add-on. Phase 2S remains the accepted production baseline in the separate
[`nvda-piper-driver`](https://github.com/rezaei-hossein-python/nvda-piper-driver)
repository.

## Problem

High-quality neural TTS voices often produce utterance durations unsuitable for
immediate screen-reader character and navigation feedback. Global speed-up
(`length_scale`) shortens everything uniformly and degrades consonant clarity and
naturalness. Screen readers need a **compact interactive style** without
sacrificing speaker identity or reading quality.

## Production baseline

Phase 2S adaptive onset shaping is preserved in the production add-on. This
repository records why inference-time duration manipulation alone is insufficient
as a general policy and why the next step is **learned interactive-mode
conditioning** during training or fine-tuning.

## Research question

Can Piper/VITS itself be adapted—via a learned `speech_mode` condition—to
generate both normal reading speech and compact interactive speech from one
speaker identity?

## Major findings (Phase 2AD–2AM)

| Phase | Finding |
|---|---|
| **2AD/2AE** | ONNX duration boundary discovered; per-token override before alignment is technically possible with a graph rewrite |
| **2AE baseline** | First listening set invalidated by missing `normalize_audio=True`; corrected in 2AG |
| **2AG** | One- and two-frame separator reductions preserve voice quality on a four-item gate |
| **2AH–2AJ** | Bounded duration-policy ladder (P0–P6, V1–V6, A0–A8); V6 rejected perceptually; A5 (E1+E2+E3 boundary edits) emerges as strongest modified candidate |
| **2AK** | A5 yields ~18% median duration reduction across 38 items but fails clear-majority quality gate (Original 10, A5 4 of 14 valid trials) |
| **2AL** | Repeated listening shows **stable item-dependent preference** (S/W/button→A5; R/U/0→Original); stochastic ONNX variation does not explain the split |
| **2AM** | **Outcome C**: no credible identity-free structural selector between Original and A5; stronger hand-written inference heuristics are not justified |

Detailed chronology: `results/summaries/piper-duration-path.md`, `results/phase2am/`.

## Current conclusion

> Inference-time duration manipulation is technically viable but not sufficiently
> generalizable as a fixed or simple structure-routed policy.

All Phase 2AD–2AM artifacts (override tooling, A5 implementation, graph
analysis, listening findings, selector failure) are retained as evidence for this
transition.

## New direction (Phase 2AN+)

Train or fine-tune Piper to learn a dedicated **interactive speech mode** while
preserving normal mode and one speaker identity. See `training/` for the
feasibility and architecture design phase.

## Repository layout

```text
corpus/           Evaluation text for inference experiments
experiments/      ONNX duration override and graph inspection (Phase 2AD–2AM)
locks/            Pinned source hashes and artifact identities (no weights copied)
results/          Findings, measurements, and summaries (no raw audio tracked)
scripts/          Local reproduction helpers
tests/            Regression tests for inference research
training/         Screen-reader-conditioned Piper design (Phase 2AN+)
```

## Pinned source

- Piper: [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) v1.5.0
  at `e2a4b1fa1c502bbb97e729a5b34a6af565007843` (GPL-3.0-or-later)
- VITS reference: [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
  at `2e561ba58618d021b5b8323d3765880f7e0ecfdb` (MIT)

Upstream checkouts are reference-only and not tracked. Identities are recorded
in `locks/source.lock.json`.

## Related repositories

- **Production add-on**: NVDA Piper driver (Phase 2S baseline; read-only for this work)
- **Broader neural TTS R&D**: [`nvda-fast-neural-tts-research`](https://github.com/rezaei-hossein-python/nvda-fast-neural-tts-research) — FastSpeech2, Matcha, vocoder, and architecture history leading to Piper

## License and artifacts

Research code and documentation in this repository follow the project's chosen
license. Voice weights, ONNX models, generated audio, virtual environments, and
private blind-test keys are **not** tracked. See `.gitignore` and `locks/`.
