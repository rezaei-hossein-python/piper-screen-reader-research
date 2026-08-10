# Phase 2AI automatic findings

Corpus: 54 interactive utterances (26 letters, 10 digits, 6 spoken punctuation names, 12 UI micro-utterances). Candidate B is V6 pending listening.

| Policy | Median | P90 | P95 | Max | Median saved | P95 saved | Median reduction | Median modifications | Incremental ms/mod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 560.0 | 776.0 | 837.6 | 1264.0 | 0.0 | 0.0 | 0.0% | 0.0 | 0.0 |
| V1 | 544.0 | 760.0 | 821.6 | 1248.0 | 16.0 | 16.0 | 2.9% | 1.0 | 16.0 |
| V2 | 472.0 | 646.4 | 747.2 | 1040.0 | 80.0 | 133.6 | 15.1% | 5.0 | 16.0 |
| V3 | 464.0 | 646.4 | 731.2 | 1040.0 | 96.0 | 133.6 | 15.8% | 6.0 | 16.0 |
| V4 | 464.0 | 646.4 | 725.6 | 1040.0 | 96.0 | 133.6 | 15.8% | 6.0 | 16.0 |
| V5 | 448.0 | 614.4 | 693.6 | 1024.0 | 128.0 | 160.0 | 20.6% | 6.0 | 16.0 |
| V6 | 448.0 | 614.4 | 693.6 | 1024.0 | 128.0 | 160.0 | 20.6% | 6.0 | 0.0 |

V1 is exactly Phase 2AH P1. V2 adds only audited PAD/BOS/EOS occupancy. V3 adds one long-vowel frame; V4 repeats that bounded reduction across eligible long vowels. V5 adds terminal PAD/EOS optimization. V6 equals V5 because no further independently justified class was available; this is the diminishing-return stop.

All consonants, unknown/other speech-bearing tokens, stress, length and diacritic controls remain unchanged. No global scalar or PCM truncation is used.

Automatic validation: PASS (378 renders). Warm `button` inference: original median 25.7 ms; V6 median 22.4 ms.
