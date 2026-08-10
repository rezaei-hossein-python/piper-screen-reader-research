# Phase 2AG corrected Piper selective-duration listening gate

This is a four-item perceptual gate using the corrected Phase 2AF pipeline. It
does not modify NVDA, Phase 2S, or the original Lessac model.

## Fixed configuration

`noise_scale=0.667`, `length_scale=1.0`, `noise_w=0.8`,
`normalize_audio=True`, volume `1.0`, mono, 16 kHz. The original Lessac model
and config remain hash-verified. The deterministic Phase 2AE equivalence proof
was rerun: original versus override-disabled and self-duration override were
byte-identical with maximum absolute difference `0.0`.

## Corpus and edits

The four items are `F`, `S`, `A`, and `button`. Token IDs are mapped back to
phonemes using Piper's phonemizer and reverse `phoneme_id_map`. The selected
region was a sufficiently long separator/silence token; no speech consonant
was shortened.

| Item | Original duration vector | Edited token | C1 vector | C2 vector | O/C1/C2 duration |
|---|---|---:|---|---|---:|
| F | `[2,2,3,4,7,2,9,2,7]` | 3 (`_`) | `[2,2,3,3,7,2,9,2,7]` | `[2,2,3,2,7,2,9,2,7]` | 608/592/576 ms |
| S | `[12,1,7,1,2,4,4,4,6]` | 5 (`_`) | `[12,1,7,1,2,3,4,4,6]` | `[12,1,7,1,2,2,4,4,6]` | 656/640/624 ms |
| A | `[10,1,5,3,6,4,2,4,3]` | 3 (`_`) | `[10,1,5,2,6,4,2,4,3]` | `[10,1,5,1,6,4,2,4,3]` | 608/592/576 ms |
| button | `[2,1,1,1,1,2,1,2,3,1,3,5,2,5,8]` | 11 (`_`) | `[2,1,1,1,1,2,1,2,3,1,3,4,2,5,8]` | `[2,1,1,1,1,2,1,2,3,1,3,3,2,5,8]` | 608/592/576 ms |

Each frame is 256 samples, or 16 ms at 16 kHz. C1 removes one frame and C2
removes two frames from the same eligible region. All samples passed finite,
mono/16-kHz PCM, normalization, nonempty, and no-clipping checks.

## Blinded handoff

The user-facing set contains exactly 12 WAVs in
`listening/phase2ag-corrected`, randomized as four opaque A/B/C trials. The
answer key is private under ignored `results/phase2ag-DO-NOT-OPEN-answer-key.json`.
No perceptual conclusion is recorded until the user reports the listening
result.
