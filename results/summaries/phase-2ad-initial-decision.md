# Phase 2AD initial decision

The Piper source audit succeeded: VITS duration prediction and alignment are
the right conceptual intervention point. The locked Lessac ONNX graph exposes
neither the duration tensor nor an override input. The upstream alignment-output
patch can expose `w_ceil` for inspection only; it cannot inject modified values.

Therefore the first inference-only proof is **blocked by the model/export
boundary**. No conservative/balanced/aggressive policy was run, no listening
set was generated, and no production code was changed. A future narrowly scoped
research step may evaluate a modified export or obtain a compatible trainable
Lessac checkpoint, but it must not silently alter the Phase 2S model or begin
retraining.
