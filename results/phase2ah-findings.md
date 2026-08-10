# Phase 2AH pre-listening findings

Phase 2AH searches for a bounded cumulative Piper duration envelope. It uses
the corrected Phase 2AF configuration and the validated Phase 2AE graph
override. No NVDA or Phase 2S production code is involved.

## Policies

P0 is original. P1 reduces the first eligible separator by one frame. P2
reduces it by up to two frames. P3 reduces every eligible separator by one
frame. P4 reduces the terminal separator by up to two frames. P5 reduces one
long vowel by one frame. P6 combines P3 with one long-vowel frame reduction.
All policies are host-validated, preserve active-token minimums, and never
modify consonants or unknown tokens.

Across 24 items, the duration summaries were:

| Policy | Median | P95 | Maximum | Median saved | Median reduction | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 576 ms | 832 ms | 1168 ms | 0 ms | 0% | 1.74/s |
| P1 | 560 ms | 816 ms | 1152 ms | 16 ms | 2.78% | 1.79/s |
| P2 | 552 ms | 816 ms | 1152 ms | 16 ms | 3.18% | 1.81/s |
| P3 | 488 ms | 768 ms | 992 ms | 64 ms | 12.70% | 2.05/s |
| P4 | 544 ms | 816 ms | 1152 ms | 32 ms | 5.41% | 1.84/s |
| P5 | 576 ms | 816 ms | 1152 ms | 16 ms | 2.38% | 1.74/s |
| P6 | 480 ms | 752 ms | 976 ms | 80 ms | 14.96% | 2.08/s |

All generated PCM was finite, valid, mono 16 kHz, normalized, and unclipped.
The warm research-only inference comparison showed no meaningful selector
overhead: original median 38.7 ms, rewritten disabled 29.7 ms, and rewritten
self-override 27.7 ms on the fixed `button` probe (20 warm iterations).

Candidate Q is P1, the conservative one-frame separator policy. Candidate R is
P6, the strongest bounded policy that remained structurally clean. The eight
strategic items are blinded in the ignored listening output; no perceptual
conclusion is recorded until manual listening.
