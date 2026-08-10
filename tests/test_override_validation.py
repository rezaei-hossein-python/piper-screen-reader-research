from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("duration_probe", ROOT / "experiments/onnx-duration-override/duration_probe.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_valid_override_and_single_change() -> None:
	predicted = np.array([[[2, 3, 0, 1]]], dtype=np.float32)
	override = predicted.copy()
	assert MODULE.validate_override(override, predicted).shape == predicted.shape
	changed = MODULE.modest_single_token_change(predicted, 1, -1)
	assert changed.tolist() == [[[2.0, 2.0, 0.0, 1.0]]]


@pytest.mark.parametrize(
	"bad",
	[
		np.array([[[2.5, 3, 0, 1]]], dtype=np.float32),
		np.array([[[2, -1, 0, 1]]], dtype=np.float32),
		np.array([[[2, 3, 1, 1]]], dtype=np.float32),
		np.array([[[2, 0, 0, 1]]], dtype=np.float32),
	]
)
def test_invalid_override_rejected(bad: np.ndarray) -> None:
	predicted = np.array([[[2, 3, 0, 1]]], dtype=np.float32)
	with pytest.raises(ValueError):
		MODULE.validate_override(bad, predicted)
