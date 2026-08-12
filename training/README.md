# Phase 2AN — Screen-reader-conditioned Piper training research

This area begins a **new research story** after Phase 2AM Outcome C. It does not
modify Phase 2AD–2AM inference experiments in place.

Phase 2AN is initially a **training feasibility and architecture design phase**.
No expensive training runs are authorized without an explicit follow-on phase.

## Hypothesis

One Piper/VITS speaker can learn two prosody/timing regimes controlled by a
discrete condition:

| Mode | Behavior |
|---|---|
| **normal** | Ordinary duration and prosody; high-quality reading |
| **interactive** | Compact boundary timing, reduced pauses/tails, preserved consonant articulation, same speaker identity |

Interactive mode must be **learned**, not implemented as global waveform
acceleration or post-hoc time compression.

## Directory layout

```text
training/
  README.md                          This file
  ROADMAP.md                         Completed / current / future phases
  screen-reader-conditioned-piper-architecture.md   Master architecture document
  source-analysis/                   Piper/VITS training graph map
  dataset-design/                    Dual-mode dataset strategy
  conditioning/                      Mode-conditioning injection options
  experiments/                       Minimal prototype experiment design
  scripts/                           Future training helpers (empty until needed)
  results/                           Future training measurements (empty)
  locks/                             Training-specific source/asset locks
```

## Hard requirements

1. **Same voice**: normal and interactive modes share one speaker identity.
2. **Normal mode preservation**: interactive training must not materially regress reading quality.
3. **No production changes**: Phase 2S and the NVDA add-on remain untouched.
4. **No blind Sonic ground truth**: do not train on time-compressed waveforms as interactive labels.

## Outcome (Phase 2AN)

**Outcome A — fine-tuning feasible.** A public Lessac-low PyTorch checkpoint
exists; mode conditioning at the duration predictor is architecturally clean;
licensing permits fine-tuning from the public checkpoint for research. Current
local hardware lacks GPU; execution deferred to a GPU environment.

See `screen-reader-conditioned-piper-architecture.md` for full analysis.
