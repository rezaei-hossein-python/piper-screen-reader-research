"""Host-side duration validation and fixed-voice ONNX probing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DurationLimits:
	max_total_frames: int = 20000
	max_frames_per_token: int = 2000


def validate_override(override: np.ndarray, predicted: np.ndarray, limits: DurationLimits = DurationLimits()) -> np.ndarray:
	if override.dtype != np.float32:
		raise ValueError("duration override must be float32")
	if predicted.ndim != 3 or override.shape != predicted.shape:
		raise ValueError("duration override shape mismatch")
	if not np.isfinite(override).all():
		raise ValueError("duration override contains non-finite values")
	if not np.equal(override, np.rint(override)).all():
		raise ValueError("duration override must be integer-equivalent")
	if (override < 0).any() or (override > limits.max_frames_per_token).any():
		raise ValueError("duration override is outside its bounds")
	active = predicted > 0
	if (override[active] < 1).any():
		raise ValueError("active tokens require at least one frame")
	if (override[~active] != 0).any():
		raise ValueError("padding tokens must remain zero")
	if int(np.sum(override)) > limits.max_total_frames:
		raise ValueError("duration override exceeds total-frame bound")
	return override


def modest_single_token_change(predicted: np.ndarray, index: int, delta: int) -> np.ndarray:
	result = predicted.astype(np.float32, copy=True)
	if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] != 1:
		raise ValueError("expected [1, 1, phonemes] duration tensor")
	value = int(result[0, 0, index])
	if value < 1 or value + delta < 1:
		raise ValueError("token is inactive or change would make it invalid")
	result[0, 0, index] = value + delta
	return result
