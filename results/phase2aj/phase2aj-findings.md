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
