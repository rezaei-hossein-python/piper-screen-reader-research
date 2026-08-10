# Phase 2AI unknown-token audit

The Phase 2AH `protected/unknown` pool is fully explained. It was 370 frames, but it was not a single semantic class. The legacy classifier missed Unicode IPA and Piper control tokens.

## Exact 370-frame reconstruction

| Correct class | Frames |
|---|---:|
| boundary/silence | 178 |
| stress/control marker | 92 |
| vowel | 72 |
| liquid | 6 |
| other speech-bearing | 6 |
| nasal | 6 |
| fricative | 5 |
| stop | 3 |
| punctuation/boundary | 2 |
| **Total** | **370** |

Symbols and frames: `$`=94, `^`=84, `ˈ`=78, `ɪ`=28, `ɛ`=23, `ː`=11, `ɑ`=9, `ə`=8, `ɹ`=6, `ŋ`=6, `ʃ`=5, `ᵻ`=4, `ʌ`=3, `ʔ`=3, `̩`=3, `ʲ`=2, ` `=2, `ɐ`=1.

`^` (ID 1) is BOS, `$` (ID 2) is EOS, and `_` (ID 0) is Piper PAD inserted before and after phonemes. Stress/length/diacritic tokens remain protected. IPA tokens are speech-bearing and are now classified by manner.

## Expanded-corpus active-token audit

| ID | Symbol | Class | Occurrences | Frames |
|---:|---|---|---:|---:|
| 0 | `_` | padding | 364 | 831 |
| 1 | `^` | boundary/silence | 54 | 111 |
| 2 | `$` | boundary/silence | 54 | 294 |
| 3 | ` ` | punctuation/boundary | 3 | 6 |
| 14 | `a` | vowel | 6 | 17 |
| 15 | `b` | stop | 4 | 4 |
| 17 | `d` | stop | 11 | 18 |
| 18 | `e` | vowel | 7 | 17 |
| 19 | `f` | fricative | 3 | 12 |
| 20 | `h` | fricative | 1 | 1 |
| 21 | `i` | vowel | 12 | 33 |
| 22 | `j` | glide | 4 | 8 |
| 23 | `k` | stop | 16 | 32 |
| 24 | `l` | liquid | 10 | 20 |
| 25 | `m` | nasal | 6 | 10 |
| 26 | `n` | nasal | 13 | 38 |
| 27 | `o` | vowel | 2 | 4 |
| 28 | `p` | stop | 4 | 7 |
| 31 | `s` | fricative | 13 | 44 |
| 32 | `t` | stop | 13 | 27 |
| 33 | `u` | vowel | 5 | 12 |
| 34 | `v` | fricative | 4 | 8 |
| 35 | `w` | glide | 3 | 7 |
| 38 | `z` | fricative | 2 | 5 |
| 39 | `æ` | vowel | 4 | 12 |
| 44 | `ŋ` | nasal | 2 | 3 |
| 50 | `ɐ` | vowel | 2 | 2 |
| 51 | `ɑ` | vowel | 5 | 13 |
| 54 | `ɔ` | vowel | 1 | 1 |
| 59 | `ə` | vowel | 12 | 18 |
| 61 | `ɛ` | vowel | 16 | 44 |
| 66 | `ɡ` | stop | 1 | 1 |
| 74 | `ɪ` | vowel | 18 | 44 |
| 88 | `ɹ` | liquid | 7 | 16 |
| 96 | `ʃ` | fricative | 6 | 22 |
| 100 | `ʊ` | vowel | 2 | 10 |
| 102 | `ʌ` | vowel | 5 | 10 |
| 108 | `ʒ` | fricative | 2 | 3 |
| 109 | `ʔ` | stop | 1 | 3 |
| 119 | `ʲ` | other speech-bearing | 1 | 3 |
| 120 | `ˈ` | stress/control marker | 56 | 157 |
| 121 | `ˌ` | stress/control marker | 1 | 1 |
| 122 | `ː` | stress/control marker | 21 | 44 |
| 126 | `θ` | fricative | 1 | 1 |
| 128 | `ᵻ` | other speech-bearing | 3 | 11 |
| 144 | `̩` | stress/control marker | 1 | 1 |

## Frame occupancy

| Class | Frames |
|---|---:|
| padding | 831 |
| boundary/silence | 405 |
| vowel | 237 |
| stress/control marker | 203 |
| fricative | 96 |
| stop | 92 |
| nasal | 51 |
| liquid | 36 |
| glide | 15 |
| other speech-bearing | 14 |
| punctuation/boundary | 6 |
