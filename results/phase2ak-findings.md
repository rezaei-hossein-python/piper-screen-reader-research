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

## Decoded Phase 2AK manual result

The user supplied valid preferences for 14 trials and rejected two trials as
unacceptable for both variants. The private key decoded as follows:

| Trial | Item | A | B | Selected result |
|---:|---|---|---|---|
| 01 | S | A5 | Original | A5 |
| 02 | U | A5 | Original | Original |
| 03 | W | Original | A5 | Original |
| 04 | A | A5 | Original | REJECT — both unacceptable |
| 05 | K | Original | A5 | Original |
| 06 | M | Original | A5 | Original |
| 07 | R | A5 | Original | A5 |
| 08 | 0 | Original | A5 | Original |
| 09 | 5 | A5 | Original | A5 |
| 10 | exclamation mark | A5 | Original | REJECT — both unacceptable |
| 11 | comma | Original | A5 | Original |
| 12 | expanded | Original | A5 | Original |
| 13 | unavailable | A5 | Original | Original |
| 14 | button | A5 | Original | Original |
| 15 | heading | A5 | Original | Original |
| 16 | dialog | Original | A5 | A5 |

Among the 14 valid trials, Original was preferred 10 times and A5 four times.
Trials 04 (`A`) and 10 (`exclamation mark`) are separate baseline/item
failures: Original was also unacceptable, so neither rejection is attributed
to A5. They may reflect Lessac pronunciation, frontend/corpus behavior, or
stochastic generation variation and require separate diagnosis.

Overlap with Phase 2AJ was mixed: S remained A5; U and W changed from A5 to
Original; 0 remained Original; exclamation mark changed from A5 preference to
shared rejection; expanded changed from A5 preference to Original; unavailable
changed from A5 preference to Original; and button changed from A5 preference
to Original. This is not a stable quality advantage for A5.

## Decision

Outcome **C**: A5 fails as a general interactive policy. It retains a material
automatic advantage—characters 520→392 ms median and 800→672 ms P95; digits
624→480 ms median; punctuation 592→496 ms; UI 576→456 ms—but A5 was preferred
on only 28.6% of valid trials, far below the clear-majority acceptance gate.
No systematic A5 quality advantage is demonstrated. A5 must not advance to
isolated NVDA testing; retain it as a research artifact only.
