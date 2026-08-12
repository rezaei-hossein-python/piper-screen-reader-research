# Mode conditioning options

Three mechanisms were identified before source analysis; this document records
tradeoffs after reviewing `models.py` and `export_onnx.py`.

## A. Learned mode embedding (preferred candidate)

Add `nn.Embedding(2, gin_channels)` for `speech_mode ∈ {0=normal, 1=interactive}`.

Broadcast to `[B, gin_channels, 1]` and combine with speaker embedding `g` when
present. For Lessac (single speaker), mode embedding **becomes the sole global
condition vector** passed as `g` to `dp`, and optionally to `flow` and `dec`.

| Injection site | Effect | Risk |
|---|---|---|
| **dp only** | Different timing; acoustic path unchanged if `g=None` elsewhere | Preferred first hypothesis |
| enc_p + dp | Text-dependent and duration-dependent style | Higher risk to normal mode |
| dp + flow + dec | Full prosody shift | May alter timbre; harder to preserve identity |

## B. Duration-specific conditioning (preferred first experiment)

Extend `StochasticDurationPredictor` / `DurationPredictor` to accept mode
conditioning **only** via existing `gin_channels` pathway:

```python
# Conceptual — not implemented
g_mode = self.emb_mode(speech_mode).unsqueeze(-1)  # [B, gin, 1]
logw = self.dp(x, x_mask, g=g_mode, reverse=True, noise_scale=noise_scale_w)
```

**Rationale:** Training forward pass already detaches `x` before `dp` and uses
teacher alignment `w` for SDP loss. Mode-specific duration targets can be
learned without changing mel reconstruction targets in normal mode, if batches
are explicitly labeled by mode.

**Feasibility: YES** — `gin_channels` plumbing exists; Lessac config must set
`gin_channels > 0` even for single-speaker (currently 0). Fine-tuning from
checkpoint requires widening `emb_g` / adding `emb_mode` and initializing new
weights near zero.

## C. Fine-tuning with explicit duration targets

Provide mode-specific target durations derived from:

- Teacher normal alignments (from frozen Lessac baseline)
- Shortened interactive alignments (alignment editing, not Sonic compression)

Loss: standard SDP flow loss + optional auxiliary MSE on `logw` toward
mode-specific targets.

**Tradeoff:** Requires curated dual-alignment labels per utterance. More
supervision than embedding alone but may converge faster on tiny corpus.

## D. Rejected for Phase 2AN

| Approach | Reason |
|---|---|
| Global `length_scale` only | Already rejected in Phase 2Y; not mode-aware |
| Post-export ONNX override | Phase 2AM showed policy limits; not a training substitute |
| Waveform time compression as label | Phase 2AN explicitly forbids Sonic-style ground truth |

## Preferred architecture (Phase 2AN conclusion)

**Option B first:** mode embedding injected **only into `dp`** via `gin_channels`.

Secondary ablation if insufficient: add same embedding to `flow` (not `dec` initially).

Implementation prerequisites (future phase, not executed here):

1. Set `gin_channels` in model config (match checkpoint architecture or expand).
2. Add `emb_mode` parameter; load Lessac checkpoint with strict=False for new keys.
3. Pass `speech_mode` in training batch and `infer`.
4. Extend `export_onnx.py` with optional `speech_mode` input (default 0).

See `../screen-reader-conditioned-piper-architecture.md` § ONNX interface.
