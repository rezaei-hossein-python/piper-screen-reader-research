# Phase 2AJ automatic findings

Corpus: 15 items (seven primary usable Phase 2AI items plus eight confirmation items). `Y` is analyzed separately and excluded from policy scoring.

| Policy | Median | P75 | P90 | P95 | Max | Median/P95 saved | Median % reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 640.0 | 744.0 | 998.4 | 1043.2 | 1088.0 | 0.0/0.0 ms | 0.00% |
| A1 | 624.0 | 728.0 | 982.4 | 1027.2 | 1072.0 | 16.0/16.0 ms | 2.50% |
| A2 | 576.0 | 640.0 | 867.2 | 912.0 | 912.0 | 96.0/164.8 ms | 14.29% |
| A3 | 592.0 | 696.0 | 950.4 | 1000.0 | 1056.0 | 48.0/48.0 ms | 7.50% |
| A4 | 608.0 | 720.0 | 966.4 | 1016.0 | 1072.0 | 32.0/32.0 ms | 3.12% |
| A5 | 544.0 | 616.0 | 841.6 | 884.8 | 896.0 | 112.0/180.8 ms | 18.92% |
| A6 | 560.0 | 632.0 | 851.2 | 900.8 | 912.0 | 96.0/176.0 ms | 16.18% |
| A7 | 576.0 | 688.0 | 934.4 | 988.8 | 1056.0 | 64.0/64.0 ms | 8.33% |
| A8 | 528.0 | 608.0 | 825.6 | 873.6 | 896.0 | 128.0/192.0 ms | 19.44% |

A0 is Original; A1 is exact V1; A8 is exact V6. Candidate M is A5 (V1 + internal boundary + terminal optimization). Candidate F is A6 (V1 + internal boundary + one long-vowel reduction). A5 retains more V6 duration savings than A6 automatically, while A6 isolates whether one vowel edit adds useful value. A deterministic adaptive rule is not justified by seven preference observations; no item identity is used.

Automatic validation passed for all 135 renders. All edit families preserve token sequence, active-token minimums, consonants, finite normalized mono 16-kHz PCM, and alignment safety. No Phase 2AJ policy is perceptually validated until the next blind gate.
