# Phase 2AK — frozen A5 Original-versus-A5 quality gate

Phase 2AK freezes A5 exactly as Phase 2AJ Candidate M: E1 first PAD/separator,
E2 internal PAD/BOS/EOS boundary reductions, and E3 terminal PAD/EOS reduction.
There is no vowel, consonant, adaptive, global-scaling, or item-specific edit.

The corpus contains 38 fixed items: 18 characters, 7 digits, 5 punctuation
names, and 8 UI/navigation utterances. `Y` is excluded from scoring as the
known baseline pronunciation case. Authoritative settings are noise scale
0.667, length scale 1.0, noise_w 0.8, normalize_audio=True, volume 1.0, and
16-kHz mono output.

| Category | Original median/P75/P90/P95/max | A5 median/P75/P90/P95/max | A5 median/P95 saved | Median reduction |
|---|---|---|---|---:|
| Characters | 520/624/710.4/800/800 ms | 392/488/604.8/672/672 ms | 96/144 ms | 19.72% |
| Digits | 624/720/748.8/758.4/768 ms | 480/600/614.4/619.2/624 ms | 128/155.2 ms | 20.83% |
| Punctuation | 592/896/1078.4/1139.2/1200 ms | 496/752/896/944/992 ms | 144/195.2 ms | 17.33% |
| UI/navigation | 576/668/824/852/880 ms | 456/548/675.2/697.6/720 ms | 128/154.4 ms | 20.83% |

Automatic validation passed all 76 renders: same phoneme sequence/settings,
only duration-plan difference, consonants untouched, active-token minimums,
valid alignment, finite normalized mono 16-kHz PCM, no clipping or runtime
errors.

The longest A5 items were exclamation mark (992 ms), question mark (752 ms),
unavailable (720 ms), F (672 ms), M (672 ms), expanded (656 ms), 2 (624 ms),
5 (608 ms), 9 (592 ms), and U (576 ms). Their remaining occupancy is dominated
by speech-bearing phonemes plus protected stress/control and boundary tokens;
this phase does not modify those classes.

The fixed blind gate contains 16 trials and 32 WAVs: S, U, W, A, K, M, R, 0,
5, exclamation mark, comma, expanded, unavailable, button, heading, and
dialog. Each trial has opaque A/B assignment of Original and A5. The answer
key is private and must not be decoded until the user returns the scoring sheet.
