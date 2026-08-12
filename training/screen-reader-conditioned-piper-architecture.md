# Screen-reader-conditioned Piper architecture

Phase 2AN master document. No training was executed.

---

## 1. Problem

Screen-reader feedback requires sub-400 ms perceptual units for characters,
digits, and navigation tokens while preserving neural voice quality. Production
Phase 2S improves onset but cannot shorten underlying Piper utterance durations
without quality tradeoffs.

## 2. Previous inference findings (Phase 2AD–2AM)

- **Proven:** ONNX graph accepts per-token duration override before alignment.
- **A5 policy:** ~18% median duration reduction (E1+E2+E3 boundary edits).
- **Quality:** Item-dependent stable preference (2AL); not a universal win.
- **Selector:** Phase 2AM Outcome C — no identity-free structural Original/A5 router.
- **Conclusion:** Stronger inference heuristics are not justified.

These findings motivate **learned** interactive prosody instead of post-hoc edits.

## 3. Architecture target

```text
Same speaker / same voice identity

NORMAL mode:        text → normal timing/prosody → standard Piper speech
INTERACTIVE mode:   text → compact timing/prosody → shorter screen-reader speech
```

## 4. Conditioning options

| Option | Description | Status |
|---|---|---|
| A | Learned mode embedding (normal=0, interactive=1) | Primary |
| B | Duration-predictor-only injection via `gin_channels` | **Preferred first** |
| C | Fine-tune with explicit dual duration targets | Complementary supervision |

Details: `conditioning/mode-conditioning-options.md`

## 5. Preferred conditioning point

**Duration predictor only** (`StochasticDurationPredictor` for Lessac low).

Mode embedding broadcast as `g` into `dp` while `flow` and `dec` remain unconditioned
on mode in the first experiment. This minimizes timbre drift while allowing
different `logw → w_ceil → alignment` paths.

Required source changes (future):

- Add `emb_mode: Embedding(2, gin_channels)`
- Ensure `gin_channels > 0` in config for single-speaker fine-tune
- Thread `speech_mode` through `VitsModel` batch, `infer`, and ONNX export

## 6. Training losses

Existing Piper/VITS losses (from `lightning.py`):

| Loss | Purpose |
|---|---|
| `loss_dur` | SDP flow-matching / duration predictor |
| `loss_mel` | Spectral reconstruction (L1) |
| `loss_kl` | Latent KL |
| `loss_gen`, `loss_fm`, `loss_disc` | Adversarial |

Interactive mode learning relies on **mode-labeled batches** so `loss_dur` sees
shorter teacher alignments in interactive mode while `loss_mel` still anchors
natural spectra. Optional auxiliary `logw` MSE toward edited alignments for
interactive clips.

Normal-mode batches use unmodified Lessac alignments to preserve reading behavior.

## 7. Dataset strategy

See `dataset-design/interactive-mode-strategy.md`. Key points:

- Dual-mode labels required; short text alone insufficient
- No Sonic/time-compression ground truth
- Minimal 17-token prototype before expansion
- Blizzard 2013 Lessac corpus is research-licensed; HF checkpoint bundle includes metadata

## 8. Same-speaker requirement (hard)

One selected voice for both modes. Not a split "reading voice" vs "typing voice."
Mode embedding must not replace speaker embedding; for Lessac (single speaker),
mode becomes the global condition vector for duration only.

## 9. ONNX / export implications

Proposed additional input:

```text
input, input_lengths, scales, speech_mode
```

| Property | Value |
|---|---|
| dtype | `int64` scalar or length-1 tensor |
| default | `0` (normal) |
| `0` | Must match existing Lessac ONNX behavior within stochastic tolerance |
| `1` | Interactive compact timing |

Export changes in `export_onnx.py`:

- Add dummy `speech_mode` to trace
- Include in `input_names` when non-default export is built

## 10. Backward compatibility

**Desired invariant:** `speech_mode = 0` → existing Piper behavior.

Compatibility options if extra input breaks generic Piper:

| Strategy | Notes |
|---|---|
| Optional input with initializer default 0 | Best if ONNX Runtime accepts omitted optional inputs |
| Separate export graphs | `lessac-normal.onnx` + `lessac-dual.onnx` |
| Runtime wrapper | Research-only; not for Phase 2S |

Standard Piper passes three inputs today (`input`, `input_lengths`, `scales`).
Adding a fourth requires add-on changes in a **future** integration phase only.

## 11. Licensing

| Asset | License | Redistribution |
|---|---|---|
| Piper code (piper1-gpl) | GPL-3.0-or-later | Fork/modify under GPL |
| Piper checkpoints (HF dataset) | MIT | Public checkpoints redistributable per HF |
| Lessac ONNX (piper-voices) | MIT | Model card cites Blizzard source |
| Lessac Blizzard 2013 audio | Research license | **Not** freely commercial; manual license |
| LJSpeech | N/A for Lessac | Lessac is **not** LJSpeech-trained |
| Derived fine-tuned weights | Uncertain for public release | Do not publish weights without legal review; MODEL_CARD required |

**Distinction:** code (GPL), dataset (Blizzard research), checkpoint (MIT on HF),
derived model (project-specific; treat as non-redistributable until reviewed).

Fine-tuning from public checkpoint for **local research** is consistent with
Piper documentation; publishing derived weights is a separate decision.

## 12. Compute estimate

Current development machine (Phase 2AN survey):

| Resource | Value |
|---|---|
| RAM | ~16 GB |
| GPU | None detected (`nvidia-smi` unavailable) |
| CPU | Single socket (details vary) |

Piper training documentation recommends NVIDIA GPU (historically 24–48 GB for
full training; fine-tune reports success from ~8 GB VRAM with small batches).

| Task | CPU | GPU |
|---|---|---|
| Minimal fine-tune (17-token + normal holdout) | Impractical (multi-day+) | **Required** — est. 2–8 GB VRAM, hours not days |
| Full Lessac retrain | Not feasible | 24+ GB VRAM, days |
| ONNX export | CPU OK | N/A |

**Phase 2AN does not start training on current hardware.**

## 13. Minimal prototype

Vocabulary: `A E F K R S U W 0 5 7 button selected expanded unavailable`

Design: `experiments/minimal-prototype-design.md`

## 14. Acceptance gates

### Normal mode

- No material regression vs baseline Lessac in naturalness, identity, pronunciation, reading quality

### Interactive mode

| Gate | Target |
|---|---|
| Initial median (chars/digits) | ≤ 350 ms |
| Stretch median | ≤ 250–300 ms |
| Quality | Intentionally concise, not accelerated |
| Latency | CPU inference within Piper-class range |

## 15. Open risks

| Risk | Mitigation |
|---|---|
| Mode embedding alters timbre | Restrict conditioning to `dp` only |
| Interactive clips overfit | Mixed normal batches; held-out listening |
| SDP stochasticity masks duration gains | Fixed seeds for eval; compare median over repeats |
| Checkpoint expansion breaks load | `strict=False` load; zero-init new embedding |
| GPL implications for NVDA integration | Separate legal review before production |
| Blizzard license for new recordings | Use alignment resynth from licensed checkpoint first |

---

## Phase 2AN outcome

**Outcome A — fine-tuning feasible**

- Public Lessac-low checkpoint exists and matches production ONNX speaker
- Architecture supports dp-only mode conditioning with modest source changes
- Licensing permits research fine-tune from HF checkpoint
- GPU environment required for execution (not available locally)

**Exact next experiment:** Minimal fine-tune per `experiments/minimal-prototype-design.md` on GPU, after implementing `emb_mode` and preparing dual-mode training clips.

**Blocking assets for immediate local execution:** GPU; interactive-labeled audio or validated alignment-resynth targets (not Sonic-compressed).

---

## References

- `source-analysis/piper-vits-architecture.md`
- `../results/summaries/piper-duration-path.md`
- `../results/phase2am/phase2am-findings.md`
- [rhasspy/piper-checkpoints lessac/low](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/lessac/low)
- [piper1-gpl v1.5.0](https://github.com/OHF-Voice/piper1-gpl/tree/v1.5.0)
