# Phase 2AF — Piper duration-quality diagnosis

## Phase 2AE manual result

The user listened to all 12 Phase 2AE trials. All tested variants were judged
very weak in voice quality. Trial 011 A remained understandable, but its
quality was highly questionable. No shortened candidate was acceptable for
screen-reader use. The conservative policy is rejected as a product
candidate; the ONNX graph override remains a successful mechanism proof.

The revealed key shows that every trial included an original Piper sample.
However, the listening generator used direct ONNX output and applied only a
fixed 0.85 clip/headroom. It did not apply Piper's `normalize_audio=True`
per-utterance peak normalization. The accepted Phase 2S runtime uses
`SynthesisConfig()` with model settings `noise_scale=0.667`,
`length_scale=1`, `noise_w=0.8`, `normalize_audio=True`, and `volume=1.0`.
Therefore the user's weak judgment applies to an invalid comparison set and
cannot isolate duration damage.

## Blind-key disposition

The 12 trials were assigned as follows:

| Trial | Source | A | B | C |
|---:|---|---|---|---|
| 001 | A | selective | global | original |
| 002 | E | global | selective | original |
| 003 | F | original | selective | global |
| 004 | K | global | original | selective |
| 005 | P | selective | global | original |
| 006 | S | selective | original | global |
| 007 | T | selective | original | global |
| 008 | V | selective | original | global |
| 009 | X | selective | original | global |
| 010 | Z | selective | global | original |
| 011 | 1 | original | selective | global |
| 012 | 7 | global | selective | original |

## Authoritative baseline A/B/C/D

The fixed corpus was `A`, `F`, `S`, `T`, `7`, `button`, `selected`, and
`The page is ready.`. Paths were:

- A: exact Phase 2S Piper Python runtime;
- B: original ONNX with the same settings and Piper normalization;
- C: rewritten graph, override disabled, same normalization;
- D: rewritten graph with its own duration vector fed back, same normalization.

For the exact nonzero Phase 2S settings, A and B produced matching sample
counts and hashes in the controlled run. C and D follow the same stochastic
model but can select a different deterministic random stream because the graph
structure/output set differs. This is why a deterministic control was also
required.

With `noise_scale=0`, `noise_w=0`, `length_scale=1`, and normalization enabled,
all four paths were byte-identical for all eight items: equal sample counts,
equal hashes, and maximum absolute difference `0.0`.

In the nonzero Phase 2S setting, A and B had identical sample counts and hashes
in the controlled sequence. C and D had the same model semantics but a separate
graph-level random stream, so their stochastic duration samples differed. The
eight-item median PCM durations were A/B **664 ms** and C/D **576 ms**; this is
sampling variation, not evidence of compression. The deterministic control is
the authoritative equivalence result.

## Compression diagnosis (structural only)

Using deterministic inference, the six-item diagnostic corpus was `F`, `S`,
`T`, `A`, `7`, and `button`. One-frame changes reduced PCM by exactly 256
samples (16 ms at 16 kHz). A two-frame vowel reduction reduced it by 512
samples.

Baseline medians across the six items were:

| Quantity | Median |
|---|---:|
| PCM duration | 552 ms |
| Leading low-energy region | 46.34 ms |
| Trailing low-energy region | 129.19 ms |
| Speech-bearing estimate | 376.03 ms |

The structural probe found no malformed PCM, NaN/Inf, or decoder crash.
Alignment cumulative differences showed that changing one token shifts all
subsequent cumulative positions (alignment L1 values typically 3–14 frames).
Expanded pre-flow latent tensors lost one frame and had overlap cosine values
approximately 0.946–0.996, depending on class. This indicates alignment/latent
coupling, not proven audible degradation.

## Classification

**Result A — research baseline was wrong.** The Phase 2AE WAV generator did
not faithfully represent accepted Phase 2S loudness normalization, and the
original samples in the blind set were also weak. Duration compression must
not be blamed until a correctly normalized, tiny baseline listening comparison
is completed.

## Recommendation

Correct the isolated research audio conversion and perform only a tiny
normalized baseline sanity check. Do not tune duration policies, integrate
with NVDA, retrain Piper, or begin Phase 2AG until that check passes.
