"""Run the bounded conservative policy on public critical utterances."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice


ITEMS = ["A", "E", "F", "K", "P", "S", "T", "V", "X", "Z", "1", "7", "comma", "question mark", "button", "selected"]


def load_module(path: Path, name: str):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def session_inputs(ids: list[int], scale: float, enabled: bool = False, override: np.ndarray | None = None):
	values: dict[str, np.ndarray] = {
		"input": np.asarray([ids], dtype=np.int64),
		"input_lengths": np.asarray([len(ids)], dtype=np.int64),
		"scales": np.asarray([0.0, scale, 0.0], dtype=np.float32),
	}
	if override is not None:
		values["duration_override"] = override
		values["duration_override_enabled"] = np.asarray(enabled, dtype=np.bool_)
	return values


def pcm_metric(samples: np.ndarray, sample_rate: int = 22050) -> dict[str, object]:
	flat = samples.reshape(-1).astype(np.float64)
	return {
		"samples": int(flat.size),
		"duration_ms": float(flat.size * 1000.0 / sample_rate),
		"peak": float(np.max(np.abs(flat))),
		"rms": float(np.sqrt(np.mean(flat * flat))),
		"finite": bool(np.isfinite(flat).all()),
		"sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type=Path)
	parser.add_argument("rewritten", type=Path)
	parser.add_argument("config", type=Path)
	parser.add_argument("policy", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	voice = PiperVoice.load(str(args.model), str(args.config))
	policy = load_module(args.policy, "policy_probe")
	validator = load_module(Path(__file__).with_name("duration_probe.py"), "duration_probe")
	reverse = {value[0]: key for key, value in voice.config.phoneme_id_map.items()}
	original = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
	rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
	rows = []
	for item in ITEMS:
		phonemes = voice.phonemize(item)[0]
		ids = voice.phonemes_to_ids(phonemes)
		labels = [reverse.get(value, "?") for value in ids]
		base = original.run(None, session_inputs(ids, 1.0))[0]
		global_pcm = original.run(None, session_inputs(ids, 0.4))[0]
		prediction = rewritten.run(None, session_inputs(ids, 1.0, False, np.ones((1, 1, len(ids)), np.float32)))[1]
		selective = np.asarray(policy.conservative(labels, prediction.reshape(-1).astype(int).tolist()), dtype=np.float32).reshape(1, 1, len(ids))
		validator.validate_override(selective, prediction.astype(np.float32))
		selective_pcm = rewritten.run(None, session_inputs(ids, 1.0, True, selective))[0]
		rows.append({
			"item_id": item,
			"phoneme_count": len(ids),
			"phoneme_classes": [policy.classify(label) for label in labels],
			"predicted_durations": prediction.reshape(-1).astype(int).tolist(),
			"selective_durations": selective.reshape(-1).astype(int).tolist(),
			"normal": pcm_metric(base),
			"global_0_4": pcm_metric(global_pcm),
			"selective": pcm_metric(selective_pcm),
		})
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
