from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def inputs(enabled: bool = False, override: np.ndarray | None = None) -> dict[str, np.ndarray]:
	details = {
		"input": np.array([[0, 1, 2, 3, 4]], np.int64),
		"input_lengths": np.array([5], np.int64),
		"scales": np.array([0.0, 1.0, 0.0], np.float32),
	}
	if override is not None:
		details["duration_override"] = override
		details["duration_override_enabled"] = np.array(enabled, np.bool_)
	return details


def stats(session: ort.InferenceSession, request: dict[str, np.ndarray]) -> dict[str, float]:
	for _ in range(5):
		session.run(None, request)
	values: list[float] = []
	for _ in range(20):
		started = time.perf_counter()
		session.run(None, request)
		values.append((time.perf_counter() - started) * 1000.0)
	values.sort()
	return {
		"median_ms": statistics.median(values),
		"p95_ms": values[int(0.95 * (len(values) - 1))],
		"min_ms": values[0],
		"max_ms": values[-1],
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("original", type=Path)
	parser.add_argument("rewritten", type=Path)
	args = parser.parse_args()
	started = time.perf_counter()
	original = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
	original_load = (time.perf_counter() - started) * 1000.0
	started = time.perf_counter()
	rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
	rewritten_load = (time.perf_counter() - started) * 1000.0
	predicted = rewritten.run(None, inputs(False, np.ones((1, 1, 5), np.float32)))[1].astype(np.float32)
	modified = predicted.copy()
	modified[0, 0, 1] -= 1
	print(json.dumps({
		"original_load_ms": original_load,
		"rewritten_load_ms": rewritten_load,
		"original": stats(original, inputs()),
		"rewritten_disabled": stats(rewritten, inputs(False, np.ones((1, 1, 5), np.float32))),
		"rewritten_self_override": stats(rewritten, inputs(True, predicted)),
		"rewritten_modified_override": stats(rewritten, inputs(True, modified)),
	}, indent=2))


if __name__ == "__main__":
	main()
