"""Small deterministic PCM metrics used by Phase 2AE proofs."""

from __future__ import annotations

import hashlib

import numpy as np


def metrics(samples: np.ndarray, sample_rate: int = 22050) -> dict[str, object]:
	if samples.size == 0 or samples.dtype.kind not in "fi":
		raise ValueError("invalid PCM array")
	if not np.isfinite(samples).all():
		raise ValueError("PCM contains NaN/Inf")
	return {
		"samples": int(samples.size),
		"duration_ms": float(samples.size * 1000.0 / sample_rate),
		"peak": float(np.max(np.abs(samples))),
		"rms": float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))),
		"sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
	}
