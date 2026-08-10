"""Tiny deterministic diagnostic of one-token duration perturbations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from piper import PiperVoice


ITEMS = ["F", "S", "T", "A", "7", "button"]
OUTPUTS = {
	"/CumSum_output_0": ["batch", 1, "phonemes"],
	"/Reshape_output_0": ["alignment_flat"],
	"/Squeeze_output_0": ["batch", "time", "phonemes"],
	"/MatMul_output_0": ["batch", "time", 192],
	"/MatMul_1_output_0": ["batch", "time", 192],
}


def load_module(path: Path, name: str):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def add_debug_outputs(source: Path) -> tuple[Path, list[str]]:
	model = onnx.load(source, load_external_data=False)
	existing = {value.name for value in model.graph.output}
	added: list[str] = []
	for name, shape in OUTPUTS.items():
		if name not in existing:
			model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
			added.append(name)
	fd, path = tempfile.mkstemp(prefix="phase2af-diagnostic-", suffix=".onnx")
	os.close(fd)
	onnx.save(model, path)
	return Path(path), added


def inputs(ids: list[int], enabled: bool, override: np.ndarray) -> dict[str, np.ndarray]:
	return {
		"input": np.asarray([ids], np.int64),
		"input_lengths": np.asarray([len(ids)], np.int64),
		"scales": np.asarray([0.0, 1.0, 0.0], np.float32),
		"duration_override": override,
		"duration_override_enabled": np.asarray(enabled, np.bool_),
	}


def pcm_metrics(samples: np.ndarray, sample_rate: int = 16000) -> dict[str, object]:
	values = samples.reshape(-1).astype(np.float32)
	peak = float(np.max(np.abs(values)))
	if peak > 1e-8:
		values = values / peak
	energy = np.abs(values)
	threshold = max(0.002, float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) * 0.08)
	first = next((i for i, value in enumerate(energy) if value >= threshold), len(values))
	last = next((i for i, value in enumerate(energy[::-1]) if value >= threshold), len(values))
	return {
		"samples": int(values.size),
		"duration_ms": float(values.size * 1000 / sample_rate),
		"leading_ms": float(first * 1000 / sample_rate),
		"trailing_ms": float(last * 1000 / sample_rate),
		"peak": float(np.max(np.abs(values))),
		"rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
		"finite": bool(np.isfinite(values).all()),
		"sha256": hashlib.sha256(values.tobytes()).hexdigest(),
	}


def tensor_difference(base: np.ndarray, changed: np.ndarray) -> dict[str, float]:
	a = base.astype(np.float64).reshape(-1)
	b = changed.astype(np.float64).reshape(-1)
	if a.shape != b.shape:
		overlap = min(a.size, b.size)
		if overlap == 0:
			return {"shape_changed": 1.0, "base_elements": float(a.size), "changed_elements": float(b.size)}
		delta = b[:overlap] - a[:overlap]
		denominator = float(np.linalg.norm(a[:overlap]) * np.linalg.norm(b[:overlap]))
		return {
			"shape_changed": 1.0,
			"base_elements": float(a.size),
			"changed_elements": float(b.size),
			"overlap_l1": float(np.sum(np.abs(delta))),
			"overlap_l2": float(np.linalg.norm(delta)),
			"overlap_max_abs": float(np.max(np.abs(delta))),
			"overlap_cosine": float(np.dot(a[:overlap], b[:overlap]) / denominator) if denominator else 1.0,
		}
	delta = b - a
	denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
	return {
		"shape_changed": 0.0,
		"l1": float(np.sum(np.abs(delta))),
		"l2": float(np.linalg.norm(delta)),
		"max_abs": float(np.max(np.abs(delta))),
		"cosine": float(np.dot(a, b) / denominator) if denominator else 1.0,
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type=Path)
	parser.add_argument("rewritten", type=Path)
	parser.add_argument("config", type=Path)
	parser.add_argument("policy", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	policy = load_module(args.policy, "phase2af_policy")
	validator = load_module(args.policy.with_name("duration_probe.py"), "phase2af_validator")
	voice = PiperVoice.load(str(args.model), str(args.config))
	reverse = {value[0]: key for key, value in voice.config.phoneme_id_map.items()}
	diagnostic_path, _ = add_debug_outputs(args.rewritten)
	session = ort.InferenceSession(str(diagnostic_path), providers=["CPUExecutionProvider"])
	rows = []
	try:
		for text in ITEMS:
			phonemes = voice.phonemize(text)[0]
			ids = voice.phonemes_to_ids(phonemes)
			labels = [reverse.get(value, "?") for value in ids]
			placeholder = np.ones((1, 1, len(ids)), np.float32)
			base_outputs = session.run(None, inputs(ids, False, placeholder))
			base_duration = base_outputs[1].astype(np.float32)
			base = {
				"text_id": text,
				"phonemes": phonemes,
				"labels": labels,
				"ids": ids,
				"durations": base_duration.reshape(-1).astype(int).tolist(),
				"pcm": pcm_metrics(base_outputs[0]),
			}
			variants: dict[str, dict[str, object]] = {}
			for kind in ("silence", "vowel", "stop", "fricative", "nasal", "liquid_glide"):
				indices = [i for i, (label, duration) in enumerate(zip(labels, base_duration.reshape(-1))) if policy.classify(label) == kind and duration > 1]
				if not indices:
					continue
				index = indices[0]
				modified = base_duration.copy()
				modified[0, 0, index] -= 1
				validator.validate_override(modified, base_duration)
				outputs = session.run(None, inputs(ids, True, modified))
				variants[f"{kind}_minus1"] = {
					"index": index,
					"label": labels[index],
					"class": kind,
					"durations": modified.reshape(-1).astype(int).tolist(),
					"pcm": pcm_metrics(outputs[0]),
					"pcm_difference": tensor_difference(base_outputs[0], outputs[0]),
					"alignment_difference": tensor_difference(base_outputs[3], outputs[3]),
					"expanded_mean_difference": tensor_difference(base_outputs[6], outputs[6]),
				}
				if kind == "vowel" and int(base_duration[0, 0, index]) > 2:
					modified = base_duration.copy()
					modified[0, 0, index] -= 2
					validator.validate_override(modified, base_duration)
					outputs = session.run(None, inputs(ids, True, modified))
					variants["vowel_minus2"] = {
						"index": index,
						"label": labels[index],
						"class": kind,
						"durations": modified.reshape(-1).astype(int).tolist(),
						"pcm": pcm_metrics(outputs[0]),
						"pcm_difference": tensor_difference(base_outputs[0], outputs[0]),
						"alignment_difference": tensor_difference(base_outputs[3], outputs[3]),
						"expanded_mean_difference": tensor_difference(base_outputs[6], outputs[6]),
					}
			base["variants"] = variants
			rows.append(base)
	finally:
		diagnostic_path.unlink(missing_ok=True)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
