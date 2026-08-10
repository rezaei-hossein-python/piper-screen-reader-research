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
