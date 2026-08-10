# Phase 2AM structural selector analysis

Primary stable groups: A5 = S, W, button; Original = R, U, 0. M is excluded. Secondary observations are used only for falsification.

## Strongest univariate thresholds

| Feature | Direction | Threshold | Correct stable items |
|---|---|---:|---:|
| boundary_frames | ge | 24 | 5/6 |
| internal_pad_frames | ge | 13 | 5/6 |
| mean_displacement_before_speech | ge | 2.25 | 5/6 |
| stop_count | ge | 2 | 5/6 |
| terminal_boundary_frames | le | 9 | 5/6 |
| total_frames | ge | 43 | 5/6 |
| active_tokens | le | 9 | 4/6 |
| boundary_ratio | le | 0.5854 | 4/6 |
| displacement_before_consonant | ge | 3 | 4/6 |
| displacement_before_stress | ge | 5 | 4/6 |
| e2_frames | ge | 6 | 4/6 |
| e3_frames | ge | 4 | 4/6 |

No single measured feature cleanly separates the six repeated items. Candidate thresholds that appear perfect on the six-item split contradict secondary observations and are rejected as overfit.

## Leave-one-item-out

| Held out | Feature | Train fit | Predicted | Observed |
|---|---|---:|---|---|
| S | stop_count | 5/5 | Original | A5 |
| 0 | boundary_frames | 5/5 | A5 | Original |
| button | total_frames | 5/5 | Original | A5 |
| W | active_tokens | 4/5 | Original | A5 |
| U | vowel_count | 4/5 | A5 | Original |
| R | terminal_boundary_frames | 5/5 | A5 | Original |

## Decision

Outcome C: no credible structural selector is justified. Stable A5 and Original groups overlap in boundary ratio, E1/E2/E3 savings, edit count, displacement before speech/consonants/stressed vowels, and phoneme-class counts. Secondary observations further falsify rules that memorize the six repeated items. Defaulting to Original when uncertain would route too many structurally overlapping cases to Original to preserve a demonstrated general speed benefit, while routing A5 lacks sufficient precision evidence. No new policy or listening set was generated.

## Secondary falsification

Using the strongest primary threshold candidate, `A5 if median boundary_frames
>= 24`, predictions were: 5→Original (observed Original, match), dialog→A5
(observed A5, match), K→Original (observed Original, match), expanded→A5
(observed A5, match), unavailable→A5 (observed Original, mismatch). The rule
matches 4/5 secondary observations but fails unavailable; it is not promoted.
