"""Generate a small opaque original/global/selective listening set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice


ITEMS = ["A", "E", "F", "K", "P", "S", "T", "V", "X", "Z", "1", "7"]


def load_module(path: Path, name: str):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
	values = np.clip(samples.reshape(-1), -0.85, 0.85)
	pcm = (values * 32767.0).astype("<i2").tobytes()
	with wave.open(str(path), "wb") as output:
		output.setnchannels(1)
		output.setsampwidth(2)
		output.setframerate(sample_rate)
		output.writeframes(pcm)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type=Path)
	parser.add_argument("rewritten", type=Path)
	parser.add_argument("config", type=Path)
	parser.add_argument("policy", type=Path)
	parser.add_argument("output", type=Path)
	parser.add_argument("answer_key", type=Path)
	args = parser.parse_args()
	voice = PiperVoice.load(str(args.model), str(args.config))
	policy = load_module(args.policy, "phase2ae_policy")
	validator = load_module(args.policy.with_name("duration_probe.py"), "phase2ae_validator")
	reverse = {value[0]: key for key, value in voice.config.phoneme_id_map.items()}
	original = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
	rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
	rng = random.Random(20260809)
	args.output.mkdir(parents=True, exist_ok=True)
	key: dict[str, object] = {}
	for index, item in enumerate(ITEMS, 1):
		phonemes = voice.phonemize(item)[0]
		ids = voice.phonemes_to_ids(phonemes)
		labels = [reverse.get(value, "?") for value in ids]
		base_inputs = {"input": np.asarray([ids], np.int64), "input_lengths": np.asarray([len(ids)], np.int64), "scales": np.asarray([0.0, 1.0, 0.0], np.float32)}
		base = original.run(None, base_inputs)[0]
		global_inputs = dict(base_inputs)
		global_inputs["scales"] = np.asarray([0.0, 0.4, 0.0], np.float32)
		global_pcm = original.run(None, global_inputs)[0]
		probe_inputs = dict(base_inputs)
		probe_inputs["duration_override"] = np.ones((1, 1, len(ids)), np.float32)
		probe_inputs["duration_override_enabled"] = np.asarray(False, np.bool_)
		prediction = rewritten.run(None, probe_inputs)[1]
		selective = np.asarray(policy.conservative(labels, prediction.reshape(-1).astype(int).tolist()), np.float32).reshape(1, 1, len(ids))
		validator.validate_override(selective, prediction.astype(np.float32))
		selective_inputs = dict(base_inputs)
		selective_inputs["duration_override"] = selective
		selective_inputs["duration_override_enabled"] = np.asarray(True, np.bool_)
		selective_pcm = rewritten.run(None, selective_inputs)[0]
		conditions = [("original", base), ("global", global_pcm), ("selective", selective_pcm)]
		rng.shuffle(conditions)
		trial = f"trial-{index:03d}"
		mapping = {}
		for suffix, (condition, samples) in zip(("a", "b", "c"), conditions):
			write_wav(args.output / f"{trial}-{suffix}.wav", samples)
			mapping[suffix] = condition
		key[trial] = {"source_item": item, "assignment": mapping}
	args.answer_key.parent.mkdir(parents=True, exist_ok=True)
	args.answer_key.write_text(json.dumps(key, indent=2), encoding="utf-8")
	(args.output / "instructions.txt").write_text(
		"For each trial, listen to A, B, and C in any order. Record which is most intelligible/natural and whether the selective version sounds worthwhile. Do not open the answer key.\n",
		encoding="utf-8",
	)


if __name__ == "__main__":
	main()
