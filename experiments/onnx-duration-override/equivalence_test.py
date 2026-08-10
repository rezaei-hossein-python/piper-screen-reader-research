"""Run original/rewritten ONNX equivalence and override proofs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _load_probe(root: Path):
	spec = importlib.util.spec_from_file_location("duration_probe", root / "duration_probe.py")
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def _inputs(enabled: bool, override: np.ndarray) -> dict[str, np.ndarray]:
	return {
		"input": np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
		"input_lengths": np.array([5], dtype=np.int64),
		"scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
		"duration_override": override,
		"duration_override_enabled": np.array(enabled, dtype=np.bool_),
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("original", type=Path)
	parser.add_argument("rewritten", type=Path)
	args = parser.parse_args()
	probe = _load_probe(Path(__file__).parent)
	original = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
	rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
	base = rewritten.run(None, _inputs(False, np.ones((1, 1, 5), dtype=np.float32)))
	original_output = original.run(None, {
		"input": np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
		"input_lengths": np.array([5], dtype=np.int64),
		"scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
	})[0]
	disabled_equal = bool(np.array_equal(original_output, base[0]))
	disabled_max_abs = float(np.max(np.abs(original_output - base[0])))
	predicted = base[1].astype(np.float32)
	probe.validate_override(predicted, predicted)
	self_run = rewritten.run(None, _inputs(True, predicted))
	self_equal = bool(np.array_equal(base[0], self_run[0]))
	self_max_abs = float(np.max(np.abs(base[0] - self_run[0])))
	changed = probe.modest_single_token_change(predicted, 1, -1)
	probe.validate_override(changed, predicted)
	changed_run = rewritten.run(None, _inputs(True, changed))
	multiple = changed.copy()
	multiple[0, 0, 3] -= 1
	probe.validate_override(multiple, predicted)
	multiple_run = rewritten.run(None, _inputs(True, multiple))
	print(json.dumps({
		"rewritten_outputs": [list(value.shape) for value in base],
		"predicted_duration": predicted.reshape(-1).tolist(),
		"proof1_disabled_byte_identical": disabled_equal,
		"proof1_disabled_max_abs": disabled_max_abs,
		"proof2_self_duration_byte_identical": self_equal,
		"proof2_self_duration_max_abs": self_max_abs,
		"disabled_pcm_samples": int(base[0].size),
		"self_override_pcm_samples": int(self_run[0].size),
		"single_token_pcm_samples": int(changed_run[0].size),
		"multi_token_pcm_samples": int(multiple_run[0].size),
	}, indent=2))


if __name__ == "__main__":
	main()
