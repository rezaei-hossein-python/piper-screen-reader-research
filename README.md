# Piper screen-reader duration research

This is an isolated, local-only Phase 2AD research repository. It does not
modify or integrate with `C:\projects\nvda piper addon`, NVDA, or the Phase 2S
cache. It investigates whether Piper/VITS per-phoneme durations can be
selectively changed before alignment generation while retaining the original
decoder and speaker.

The pinned Piper source is in `upstream/piper`; the pinned architectural VITS
reference is in `references/vits`. The existing Lessac ONNX model remains in
the protected Phase 2H asset directory and is referenced by hash rather than
copied here.

Initial work is inference-only: inspect the graph, prove duration boundaries,
and stop if the available ONNX graph cannot accept a safe override. No
retraining, NVDA integration, add-on, or production change is authorized.
