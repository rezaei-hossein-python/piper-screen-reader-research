# Piper duration path and Lessac graph boundary

## Pinned source

The research fork uses OHF-Voice `piper1-gpl` v1.5.0 at
`e2a4b1fa1c502bbb97e729a5b34a6af565007843` (GPL-3.0-or-later). The relevant
VITS reference is `jaywalnut310/vits` at
`2e561ba58618d021b5b8323d3765880f7e0ecfdb` (MIT).

## Source inference path

The source model path is:

```text
phoneme IDs
→ enc_p
→ duration predictor (dp)
→ exp(logw) * x_mask * length_scale
→ torch.ceil(w) = w_ceil
→ y_lengths / y_mask
→ commons.generate_path(w_ceil, attn_mask)
→ expand m_p and logs_p
→ flow reverse
→ decoder
→ PCM
```

The exact implementation is `src/piper/train/vits/models.py`, `SynthesizerTr.infer`.
The only public runtime control is the three-element `scales` input; its middle
value is the global `length_scale`. `train/export_onnx.py` exports only the
waveform output and does not expose duration tensors or an override input.

## Existing Lessac ONNX graph

The locked model is 63,201,294 bytes, opset 15, with 2,755 nodes. It has inputs
`input`, `input_lengths`, and `scales`, and only final waveform output `output`.
There is exactly one `Ceil` node:

```text
/Mul_1_output_0 → /Ceil → /Ceil_output_0
```

`/Ceil_output_0` feeds `/ReduceSum` (output length) and `/CumSum` (the alignment
path). It is not an input and is not externally writable. The existing upstream
`patch_voice_with_alignment.py` can mark this tensor as an additional graph
output, which proves duration extraction, but it cannot replace the tensor
before `/CumSum`.

An ephemeral in-memory graph-output proof ran the locked model through CPU
ONNX Runtime with a five-token content-free probe. It returned a valid
waveform and a positive duration vector (one observed vector was
`[1, 4, 1, 1, 8]`). This confirms extraction is real, not merely a node-name
inspection. The probe model was written only to the system temporary folder
and deleted immediately; no modified model was retained.

## Decision boundary

Predicted per-phoneme durations are internally present and extractable only by
an output patch. Selective override is not possible without a modified ONNX
export/graph (or a trainable PyTorch checkpoint and a research wrapper). The
current Phase 2H Lessac model is ONNX-only; no trainable/exportable Lessac
checkpoint was acquired in this first proof. No policy, waveform, or NVDA
integration was attempted.

## Implications

Option A (runtime override without re-export) is unavailable. Option B (a
research-only graph/export change that accepts a validated duration tensor) is
the smallest technically plausible next step, but it requires proving that the
new graph preserves all shape and decoder semantics. If that cannot be done
without a compatible trainable/exportable checkpoint, this branch must stop.
