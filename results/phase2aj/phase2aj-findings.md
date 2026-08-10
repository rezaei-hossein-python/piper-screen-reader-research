# Phase 2AJ automatic findings

Corpus: 15 items (seven primary usable Phase 2AI items plus eight confirmation items). `Y` is analyzed separately and excluded from policy scoring.

| Policy | Median | P75 | P90 | P95 | Max | Median/P95 saved | Median % reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 672.0 | 736.0 | 883.2 | 1004.8 | 1184.0 | 0.0/0.0 ms | 0.00% |
| A1 | 656.0 | 720.0 | 867.2 | 988.8 | 1168.0 | 16.0/16.0 ms | 2.38% |
| A2 | 560.0 | 640.0 | 774.4 | 843.2 | 944.0 | 96.0/184.0 ms | 13.79% |
| A3 | 624.0 | 688.0 | 835.2 | 956.8 | 1136.0 | 48.0/48.0 ms | 7.14% |
| A4 | 640.0 | 720.0 | 867.2 | 984.0 | 1152.0 | 16.0/32.0 ms | 3.03% |
| A5 | 544.0 | 608.0 | 742.4 | 811.2 | 912.0 | 128.0/204.8 ms | 17.78% |
| A6 | 560.0 | 632.0 | 774.4 | 838.4 | 928.0 | 112.0/188.8 ms | 15.56% |
| A7 | 608.0 | 688.0 | 835.2 | 952.0 | 1120.0 | 48.0/64.0 ms | 8.89% |
| A8 | 544.0 | 600.0 | 742.4 | 806.4 | 896.0 | 144.0/209.6 ms | 21.21% |

A0 is Original; A1 is exact V1; A8 is exact V6. Candidate M is A5 (V1 + internal boundary + terminal optimization). Candidate F is A6 (V1 + internal boundary + one long-vowel reduction). A5 retains more V6 duration savings than A6 automatically, while A6 isolates whether one vowel edit adds useful value. A deterministic adaptive rule is not justified by seven preference observations; no item identity is used.

Automatic validation passed for all 135 renders. All edit families preserve token sequence, active-token minimums, consonants, finite normalized mono 16-kHz PCM, and alignment safety. No Phase 2AJ policy is perceptually validated until the next blind gate.

## Decoded Phase 2AJ manual result

The user reported valid preferences for all eight trials; there were no
unacceptable/NONE responses. The private key decoded as follows:

| Trial | Item | A | B | C | Selected | Decoded selection |
|---:|---|---|---|---|---|---|
| 1 | S | A6 | Original | A5 | C | A5 |
| 2 | U | A6 | Original | A5 | C | A5 |
| 3 | W | A6 | Original | A5 | C | A5 |
| 4 | 0 | Original | A5 | A6 | A | Original |
| 5 | exclamation mark | A5 | A6 | Original | C | Original |
| 6 | expanded | A6 | Original | A5 | B | Original |
| 7 | unavailable | Original | A6 | A5 | A | Original |
| 8 | button | A6 | A5 | Original | B | A5 |

Preference counts were Original 4, A5 4, and A6 0. A5 is therefore the
leading modified experimental policy because it received four selections and
has the stronger automatic duration result (median 544 ms, P95 811.2 ms in
this ablation run). It is not perceptually validated: it tied Original and
was not preferred on exclamation mark, expanded, or unavailable. A6 has no
positive selection evidence and is not promoted.

Compared with Phase 2AI, S changed from Original to A5; U and W changed from
V1 to A5; 0 remained Original; exclamation mark and expanded changed from V6
to Original; unavailable changed from V1 to Original. The new button
confirmation selected A5. These are item-level observations, not causal proof
for or against any edit family. No pronunciation-degradation flags were
reported, but the simple preference question does not establish normal-reading
voice preservation beyond the selected items.

## Hypothesis status and recommendation

The combined evidence supports the central hypothesis provisionally: the same
Lessac Piper voice can retain recognizable, preferred quality for some
interactive items while removing substantial phoneme-aware duration occupancy.
It does not yet establish a single universally safe fast policy. A5 is the
leading research candidate; Original remains equally preferred overall, and
A6 is rejected as a candidate because it received zero selections. The next
authorized phase should test Original versus A5 on a broader, explicitly
quality-flagged interactive corpus, with item-adaptive evidence considered only
after that gate. No NVDA integration or Phase 2AK execution is authorized by
this result.
