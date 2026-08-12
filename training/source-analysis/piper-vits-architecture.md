# Piper/VITS training architecture map (piper1-gpl v1.5.0)

Pinned commit: `e2a4b1fa1c502bbb97e729a5b34a6af565007843`

Primary source files:

| Component | Path |
|---|---|
| Model definition | `src/piper/train/vits/models.py` |
| Training loop / losses | `src/piper/train/vits/lightning.py` |
| ONNX export | `src/piper/train/export_onnx.py` |
| Config (quality tiers) | `src/piper/train/vits/config.py` |

## Inference data flow (`SynthesizerTrn.infer`)

```text
phoneme IDs (input)
  → TextEncoder enc_p
      → x, m_p, logs_p, x_mask
  → speaker embedding g (if n_speakers > 1; Lessac low is single-speaker, g=None)
  → DurationPredictor or StochasticDurationPredictor dp
      → logw  (SDP: reverse=True, noise_scale_w)
      → w = exp(logw) * x_mask * length_scale
      → w_ceil = ceil(w)
      → y_lengths, y_mask
  → commons.generate_path(w_ceil, attn_mask) → attn
  → expand m_p, logs_p via attn
  → z_p = m_p + noise * exp(logs_p) * noise_scale
  → flow reverse (ResidualCouplingBlock)
  → Generator dec → waveform
```

Lessac low uses **`use_sdp=True`** (StochasticDurationPredictor), not the
deterministic DurationPredictor.

## Training data flow (`SynthesizerTrn.forward`)

```text
phoneme IDs + mel spectrogram + lengths
  → enc_p → prior stats
  → enc_q → posterior z
  → flow forward → z_p
  → monotonic_align.maximum_path → attn, w (ground-truth durations from alignment)
  → dp loss: SDP flow-matching on w, or MSE on logw for deterministic dp
  → expand prior, slice segment, dec → mel/wave
  → adversarial + feature + mel L1 + KL + duration losses
```

## Module roles

| Module | Role | Conditioning today |
|---|---|---|
| **enc_p** (TextEncoder) | Phoneme → hidden prior | None |
| **dp** (SDP or DurationPredictor) | Predict/log-duration per token | Optional `g` via `gin_channels` Conv1d cond |
| **flow** | Latent mapping prior ↔ posterior | Optional `g` |
| **dec** (Generator) | Latent → waveform | Optional `g` |
| **emb_g** | Multi-speaker embedding | N/A for Lessac (n_speakers=1) |

## ONNX export surface (current)

`export_onnx.py` wraps `model_g.infer` with inputs:

- `input` — phoneme ID sequence
- `input_lengths`
- `scales` — `[noise_scale, length_scale, noise_scale_w]`
- `sid` — speaker id (omitted for single-speaker export)

Output: `output` waveform only. No duration tensor or mode input.

## Alignment with Phase 2AD inference research

The exported graph computes `w_ceil` internally at the single `Ceil` node
feeding `CumSum` (alignment). Phase 2AE proved override by replacing that path
in a rewritten graph. A trained mode condition would ideally make override
unnecessary by learning shorter `w_ceil` directly in interactive mode.

## Key observation for conditioning

Piper already supports **global** timing control via `length_scale` in `infer`.
That is not item-aware and affects all tokens uniformly — the Phase 2Y failure
mode. A discrete **mode embedding** can modulate duration prediction without
replacing `length_scale`, preserving backward-compatible defaults when
interactive mode is off.
