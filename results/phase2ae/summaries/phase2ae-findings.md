# Phase 2AE — Piper ONNX duration override proof

## Scope

This is an isolated graph-control proof. The Phase 2S production add-on,
Phase 2H runtime/assets, and original Lessac ONNX file were not modified.

## Graph topology

The original Lessac graph is opset 15 with 2,755 nodes. It accepts `input`,
`input_lengths`, and `scales`, and emits only `output`. The single duration
boundary is:

```text
/Mul_1_output_0 → /Ceil → /Ceil_output_0
```

`/Ceil_output_0` is float32, rank 3, with symbolic shape
`[batch, 1, phonemes]`. It feeds both `/ReduceSum` and `/CumSum`.

## Rewrite

The generated graph preserves `/Ceil_output_0` and inserts:

```text
Where(duration_override_enabled,
      duration_override,
      /Ceil_output_0)
→ /Phase2AEDuration
```

Both original consumers are rewired to `/Phase2AEDuration`. The added inputs
are `duration_override` (float32 `[batch, 1, phonemes]`) and
`duration_override_enabled` (bool scalar). The generated outputs are
`output`, `/Ceil_output_0`, and `/Phase2AEDuration`.

Generated model SHA-256:

```text
a5697871afeff4fdfa5e8f515a4241a63843512a51e588bccaeb9cbd5f16e480
```

## Host validation

The research validator requires exact rank/shape, float32, finite values,
integer-equivalent nonnegative durations, active-token values >=1, zero padded
tokens, maximum 2,000 frames per token, and maximum 20,000 total frames. Any
failure rejects the override before inference.

## Proof results

With a fixed five-token content-free probe and noise scales set to zero:

| Proof | Result |
|---|---|
| Original vs rewritten, override disabled | byte-identical; max absolute difference 0.0 |
| Self-duration override | byte-identical; max absolute difference 0.0 |
| One-token change | duration vector token 1 reduced by one frame; PCM 3,328 → 3,072 samples |
| Multi-token change | two one-frame reductions; PCM 3,328 → 2,816 samples |

The output change is exactly 256 samples per frame. At 22,050 Hz this is
approximately 11.61 ms per frame, matching Piper's hop length.

## Warm CPU timing

Twenty measured iterations after five warm-ups on the isolated CPU environment:

| Mode | Median | P95 |
|---|---:|---:|
| Original | 24.87 ms | 73.35 ms |
| Rewritten, disabled | 21.83 ms | 75.76 ms |
| Rewritten, self override | 17.05 ms | 21.04 ms |
| Rewritten, modified override | 16.27 ms | 17.93 ms |

Graph load was approximately 1,782.8 ms original and 1,834.0 ms rewritten.
The selector adds no meaningful steady-state overhead in this probe.

## Bounded conservative policy probe

After the four graph proofs passed, a public 16-item probe used Piper's own
phonemizer and reverse `phoneme_id_map` to map IDs back to symbols. The policy
protected stops, fricatives, nasals, liquids/glides, and unknowns; reduced
vowels only when longer than four frames by a bounded 20%; and reduced silence
by 50%. No phoneme was deleted and the same host validator was applied.

Across the 16 items, all PCM was finite and structurally valid:

| Condition | Median duration | Minimum | Maximum |
|---|---:|---:|---:|
| Original Piper | 394.74 ms | 278.64 ms | 615.33 ms |
| Global `length_scale=0.4` | 197.37 ms | 139.32 ms | 417.96 ms |
| Conservative selective override | 301.86 ms | 220.59 ms | 557.28 ms |

This is a mechanism probe, not a final acoustic-quality result. A 12-item
opaque three-way listening set (original/global/selective) was generated under
`listening/blind-test`; its answer key is outside the user-facing directory
under the ignored Phase 2AE raw results.

## Decision

The graph-control mechanism is proven. A safe per-token duration override can
be injected after Piper duration prediction and before alignment generation
without retraining. The original prediction path remains available and the
disabled/self-duration controls are byte-identical. This phase does not yet
prove that a phoneme-aware policy sounds good; no broad policy or NVDA
integration was performed.
