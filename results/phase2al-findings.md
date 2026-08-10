# Phase 2AL — stochastic repeatability of frozen A5

Phase 2AL did not change A5. The exact policy is E1 first PAD/separator, E2
internal PAD/BOS/EOS reductions, and E3 terminal PAD/EOS reduction.

The ONNX graph contains two internal `RandomNormalLike` nodes (`/dp/RandomNormalLike`
and `/RandomNormalLike`) and exposes no random tensor or seed input. Common
random-number pairing was therefore not technically feasible without changing
graph semantics. The harness generated five paired Original/A5 realizations
per item, randomized whether Original or A5 ran first, and recorded duration,
RMS, peak, zero-crossing rate, and spectral centroid.

Corpus: S, R, 5, dialog, U, W, K, M, 0, button, expanded, unavailable.
Generated: 60 paired realizations and 120 raw WAVs. Overall median duration was
624 ms Original versus 520 ms A5, a median saving of 112 ms; the 95th-percentile
pairwise saving was 272 ms. The raw output contains five independent runs per
item, not five deterministic copies.

Across items, median within-item duration standard deviation was 41.2 ms for
Original and 45.7 ms for A5. Median RMS standard deviation was 0.0126 versus
0.0153; median spectral-centroid standard deviation was 30.0 Hz versus 46.2 Hz.
These metrics show meaningful stochastic realization variation, but they do
not substitute for listening.

Automatic validation passed all 120 renders: same phonemes/settings within each
pair, consonants untouched, valid active-token floors, finite normalized mono
16-kHz PCM, and no clipping or runtime errors.

The fixed blinded subset contains 18 trials/36 WAVs. Every one of the 12 items
appears at least once; S, R, U, W, 0, and button appear twice. Scoring asks for
overall preference, quality, pronunciation, and any problem. The answer key
is private and must not be decoded until the user returns the scoring sheet.

## Decoded Phase 2AL manual result

The user supplied 17 valid preferences and one rejected realization. The private
key decoded as follows:

| Trial | Item | Run | A | B | Selected |
|---:|---|---:|---|---|---|
| 01 | S | 1 | Original | A5 | A5 |
| 02 | R | 1 | A5 | Original | Original |
| 03 | 5 | 1 | Original | A5 | Original |
| 04 | dialog | 1 | Original | A5 | A5 |
| 05 | U | 1 | Original | A5 | Original |
| 06 | W | 1 | Original | A5 | A5 |
| 07 | K | 1 | A5 | Original | Original |
| 08 | M | 1 | Original | A5 | REJECT — both unacceptable |
| 09 | 0 | 1 | A5 | Original | Original |
| 10 | button | 1 | Original | A5 | A5 |
| 11 | expanded | 1 | Original | A5 | A5 |
| 12 | unavailable | 1 | Original | A5 | Original |
| 13 | S | 2 | A5 | Original | A5 |
| 14 | R | 2 | Original | A5 | Original |
| 15 | U | 2 | Original | A5 | Original |
| 16 | W | 2 | Original | A5 | A5 |
| 17 | 0 | 2 | Original | A5 | Original |
| 18 | button | 2 | Original | A5 | A5 |

Trial 08 was `M`; Original was also unacceptable, so it is a shared
item/realization failure and is not counted against A5 as a preference.

Among 17 valid trials: Original 9, A5 8, Same 0. The six repeated items were
fully stable: S consistently A5, R consistently Original, U consistently
Original, W consistently A5, 0 consistently Original, and button consistently
A5. Stability was 6/6 items (100%); there were no flips and no rejected repeat.

Phase 2AK had Original 10/A5 4 among 14 valid trials. Phase 2AL moved to
Original 9/A5 8 among 17 valid trials, so stochastic realization and sampling
materially changed the aggregate balance. However, the repeated-item subset
did not flip, indicating that measurable item/phoneme structure dominates the
two-run preference outcome more than unconstrained random preference noise.

The duration advantage remains substantial: across the repeated pool Original
median/P95 was 624/912 ms, A5 was 520/720 ms, with 112 ms median saving (~18%).
The variability metrics (duration SD approximately 41.2/45.7 ms, RMS SD
0.0126/0.0153, and material spectral-centroid variation) support controlled
repeat testing but do not by themselves explain perceptual quality.

## Decision

A5 is perceptually competitive in this repeatability study but does not meet
the clear-majority acceptance gate: A5 was selected in 47.1% of valid trials,
with no Same responses, and one shared rejected item. The same Lessac voice plus
A5 interactive-mode hypothesis remains promising but is not validated for a
fixed general policy. Retain A5 as research-only; future work should target
item/phoneme-structure adaptation or a focused quality-scored repeat gate. Do
not integrate A5 into NVDA.
