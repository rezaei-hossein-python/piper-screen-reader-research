# Interactive-mode dataset strategy

Phase 2AN designs dataset representation only. **No large dataset is created.**

## Core principle

Shorter text alone does not teach interactive **style**. The model must see the
same lexical content—or paired content—under two timing/prosody regimes:

```text
NORMAL:       "The settings dialog is open."
INTERACTIVE:  "button" / "S" / "seven" / "selected"
```

Interactive items are naturally short, but training must still learn **how** they
are spoken compactly, not merely **that** they are short.

## Proposed example structure

Pipe-delimited Piper training format (audio path | transcript | speaker | mode):

```text
wavs/norm_001.wav|The settings dialog is open.|0|normal
wavs/int_001.wav|button|0|interactive
```

Mode is a new metadata field requiring dataloader and collate changes.

## Supervision sources (investigated, not built)

| Source | Pros | Cons |
|---|---|---|
| **Duplicated utterances, normal vs interactive timing** | Clear mode label; same speaker | Requires interactive recordings or synthetic alignments |
| **Duration-target manipulation** | Uses existing Lessac audio; alignment editing | Risk of unnatural prosody if targets too aggressive |
| **Teacher-generated compact variants** | Could use corrected A5-style plans as weak teacher | Phase 2AM showed item-dependent quality; not universal teacher |
| **Alignment-derived targets** | Precise frame control | Labor-intensive; needs validation listening |
| **Fine-tune from Lessac checkpoint + new interactive clips** | Preserves identity | Requires new recording or high-quality synthesis |
| **Natural concise tokens from original corpus** | Authentic | May not cover full UI vocabulary |

## Phase 2AN prohibition

Do **not** create interactive targets by:

- Sonic/time compression of normal recordings
- Global resampling speed change
- Post-hoc ONNX A5 render as sole ground truth

Acceptable future directions:

- Carefully modified alignments with resynthesis from Lessac checkpoint
- Duration-conditioned teacher with listening validation
- Targeted new recording (later phase only)

## Minimal prototype corpus (Phase 2M design target)

Initial vocabulary (from Phase 2AN spec):

```text
A E F K R S U W 0 5 7
button selected expanded unavailable
```

Expand only after normal-mode regression checks pass.

## Normal-mode preservation data

Fine-tuning must include **normal-mode Lessac-like utterances** (from original
training metadata or held-out sentences) so interactive specialization does not
collapse reading prosody. Suggested split for first experiment:

- 80% batches mixed normal (existing Lessac-style)
- 20% interactive-labeled compact items

Exact ratios subject to listening gates in a future phase.

## LESSAC original dataset

The Lessac Blizzard 2013 corpus is **research-licensed** (not public domain).
Piper's public checkpoint bundle includes `dataset.jsonl.gz` (transcript/audio
metadata) on Hugging Face — sufficient for fine-tune **if** the researcher
already holds or obtains a Blizzard license for the underlying audio.

For the minimal prototype, **new synthetic or recorded interactive clips** may be
smaller than re-downloading the full Blizzard corpus.

See licensing section in `../screen-reader-conditioned-piper-architecture.md`.
