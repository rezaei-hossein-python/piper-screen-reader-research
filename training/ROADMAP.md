# Piper screen-reader research roadmap

## Completed — inference-time optimization (Phase 2AD–2AM)

- ONNX duration boundary discovery and override proof
- Normalization correction (Phase 2AG)
- Bounded policy ladder through A5
- Broad Original-vs-A5 listening gates (2AK, 2AL)
- Structural selector analysis — **Outcome C** (Phase 2AM)
- Conclusion: inference manipulation is viable but not generalizable as a fixed policy

All tooling, measurements, and findings retained under `experiments/` and
`results/`.

## Completed — learned interactive-mode feasibility (Phase 2AN)

- Map Piper/VITS training architecture (piper1-gpl v1.5.0)
- Evaluate mode-conditioning injection points
- Audit Lessac checkpoint availability and licensing
- Design minimal dual-mode prototype corpus and acceptance gates
- **No training execution** in Phase 2AN

Deliverables: `training/screen-reader-conditioned-piper-architecture.md` and
subordinate design documents.

## Completed — preflight shape/gradient validation (Phase 2AO)

- Perform local CPU preflight checks and mathematical shape/gradient validation of learned duration-predictor-only mode conditioning
- Discovered and surgically bypassed gradient-cutoff (`g = torch.detach(g)`) blocker in VITS StochasticDurationPredictor
- Confirmed 100% gradient flow from the duration loss back to `emb_mode.weight` (gradient norm: **0.997**)
- Verified model backward and forward correctness on CPU with a mock monotonic alignment

Deliverables: `training/results/phase2ao/phase2ao-findings.md` and `training/experiments/phase2ao/test_conditioning_shapes.py`.

## Future — only if Phase 2AO+ gates pass

| Phase | Scope |
|---|---|
| Trained prototype | Minimal fine-tune proving normal preserved + interactive shorter |
| Isolated NVDA integration | Separate from Phase 2S; not authorized until prototype passes gates |
| Production architecture | ADR and add-on design; Phase 2S remains baseline until accepted |
| Multilingual / Persian | Out of scope until English Lessac prototype succeeds |

## Explicit non-goals (current)

- Stronger inference heuristics or structural Original/A5 routing
- NVDA runtime, cache, WavePlayer, or scheduler changes
- Publishing model weights or ONNX artifacts in this repository
- Automatic multi-day CPU training
