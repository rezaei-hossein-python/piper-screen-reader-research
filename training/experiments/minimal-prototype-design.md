# Minimal prototype experiment design (not executed)

## Objective

Prove on a tiny vocabulary:

```text
same speaker + normal mode preserved + interactive mode shorter
```

## Vocabulary (initial)

```text
A E F K R S U W 0 5 7
button selected expanded unavailable
```

Do not expand to A–Z or full sentences until gates pass.

## Prerequisites

1. Download `epoch=2307-step=558536.ckpt` from
   [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/lessac/low)
2. GPU environment (see architecture doc § compute)
3. Source modifications for `emb_mode` + dataloader mode field
4. Interactive-labeled training clips (recorded or alignment-resynthesized)

## Training procedure (sketch)

```text
1. Load Lessac-low checkpoint (SDP, 16 kHz, low quality tier)
2. Add emb_mode (2 × gin_channels); initialize near zero
3. Enable gin_channels on dp only (first experiment)
4. Fine-tune ≤ 50 epochs on mixed batches:
     - normal: sample from Lessac metadata sentences
     - interactive: minimal corpus clips with compact alignment targets
5. Export ONNX with speech_mode input (default 0)
6. Measure durations on corpus/characters.txt subset
7. Blind listening: normal vs baseline Lessac; interactive vs Phase 2S targets
```

## Hyperparameters (starting point, not validated)

| Parameter | Value |
|---|---|
| Base checkpoint | Lessac low epoch=2307 |
| Batch size | 8–16 (VRAM dependent) |
| Learning rate | 1e-4 – 2e-4 (fine-tune) |
| Max epochs | 20–50 with early stopping on normal-mode mel loss |
| Precision | fp32 (Piper default) |

## Acceptance gates (see architecture doc)

### Normal mode

Perceptually comparable to baseline Lessac ONNX on reading sentences and on
prototype vocabulary spoken in normal mode.

### Interactive mode

| Metric | Initial target | Stretch |
|---|---|---|
| Character/digit median duration | ≤ 350 ms | ≤ 250–300 ms |
| Quality | Concise, not sped-up | — |
| CPU inference | Piper-class latency | — |

## What this experiment does not prove

- Full UI vocabulary coverage
- Multilingual or Persian paths
- Production NVDA integration
- Superiority over Phase 2S onset shaping

## Stop conditions

- Normal-mode mel loss regression > 10% vs baseline on held-out sentences
- Interactive mode preferred over A5 but still fails quality on > 30% of prototype items
- Mode embedding causes audible timbre shift in normal mode

## Next phase authorization

Execute only after explicit Phase 2AO (or equivalent) approval with GPU access
and interactive training assets ready.
