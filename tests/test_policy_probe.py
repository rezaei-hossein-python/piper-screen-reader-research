from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("policy_probe", ROOT / "experiments/onnx-duration-override/policy_probe.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_conservative_policy_protects_consonants() -> None:
	result = MODULE.conservative(["t", "e", "sil"], [2, 10, 8])
	assert result == [2, 8, 4]


def test_unknown_is_unchanged() -> None:
	assert MODULE.conservative(["?"], [7]) == [7]
