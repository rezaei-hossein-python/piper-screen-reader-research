# Phase 2AJ Y baseline analysis

Phonemes: `['w', 'ˈ', 'a', 'ɪ']`

IDs: `[1, 0, 35, 0, 120, 0, 14, 0, 74, 0, 2]`

Predicted durations: `[2, 1, 1, 5, 1, 5, 5, 1, 7, 3, 5]`

| Index | Token | Class | Frames |
|---:|---|---|---:|
| 0 | `^` | boundary/silence | 2 |
| 1 | `_` | padding | 1 |
| 2 | `w` | glide | 1 |
| 3 | `_` | padding | 5 |
| 4 | `ˈ` | stress/control marker | 1 |
| 5 | `_` | padding | 5 |
| 6 | `a` | vowel | 5 |
| 7 | `_` | padding | 1 |
| 8 | `ɪ` | vowel | 7 |
| 9 | `_` | padding | 3 |
| 10 | `$` | boundary/silence | 5 |

All four paths used the locked model/config and the Phase 2S scales (0.667, 1.0, 0.8) with normalization. The original graph does not expose a duration output, and separate ONNX sessions are stochastic, so cross-path duration/PCM identity cannot be claimed. The self-duration path uses the supplied predicted vector exactly; graph structure and token sequence are unchanged. Since Original, V1 and V6 were all unacceptable for Y, classify this as an independent Lessac/eSpeak pronunciation or item-level baseline issue, not evidence for changing the duration policy.
