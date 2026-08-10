"""Compare accepted Piper normalization against four small research paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper import SynthesisConfig, PiperVoice


ITEMS = ["A", "F", "S", "T", "7", "button", "selected", "The page is ready."]


def normalize(samples: np.ndarray) -> np.ndarray:
	values = samples.reshape(-1).astype(np.float32)
	peak = float(np.max(np.abs(values))) if values.size else 0.0
	if peak >= 1e-8:
		values = values / peak
	return np.clip(values, -1.0, 1.0).astype(np.float32)


def metrics(samples: np.ndarray, sample_rate: int) -> dict[str, object]:
	values = normalize(samples)
	energy = np.abs(values)
	threshold = max(0.002, float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) * 0.08)
	first = next((index for index, value in enumerate(energy) if value >= threshold), len(values))
	last = next((index for index, value in enumerate(energy[::-1]) if value >= threshold), len(values))
	return {
		"sample_rate": sample_rate,
		"samples": int(values.size),
		"duration_ms": float(values.size * 1000 / sample_rate),
		"peak": float(np.max(np.abs(values))),
		"rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
		"leading_ms": float(first * 1000 / sample_rate),
		"trailing_ms": float(last * 1000 / sample_rate),
		"finite": bool(np.isfinite(values).all()),
		"sha256": hashlib.sha256(values.tobytes()).hexdigest(),
	}


def ids_for(voice: PiperVoice, text: str) -> tuple[list[str], list[int]]:
	phonemes = voice.phonemize(text)[0]
	return phonemes, voice.phonemes_to_ids(phonemes)


def ort_inputs(ids: list[int], scales: list[float], enabled: bool, override: np.ndarray) -> dict[str, np.ndarray]:
	return {
		"input": np.asarray([ids], dtype=np.int64),
		"input_lengths": np.asarray([len(ids)], dtype=np.int64),
		"scales": np.asarray(scales, dtype=np.float32),
		"duration_override": override,
		"duration_override_enabled": np.asarray(enabled, dtype=np.bool_),
	}


def original_inputs(ids: list[int], scales: list[float]) -> dict[str, np.ndarray]:
	return {
		"input": np.asarray([ids], dtype=np.int64),
		"input_lengths": np.asarray([len(ids)], dtype=np.int64),
		"scales": np.asarray(scales, dtype=np.float32),
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type=Path)
	parser.add_argument("rewritten", type=Path)
	parser.add_argument("config", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	voice = PiperVoice.load(str(args.model), str(args.config))
	original = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
	rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
	config = json.loads(args.config.read_text(encoding="utf-8"))
	inference = config["inference"]
	scales = [inference["noise_scale"], inference["length_scale"], inference["noise_w"]]
	rows = []
	for text in ITEMS:
		phonemes, ids = ids_for(voice, text)
		# Path A: exact Piper Python runtime, including its per-utterance normalization.
		chunks = list(voice.synthesize(text, syn_config=SynthesisConfig()))
		audio_a = np.concatenate([chunk.audio_float_array for chunk in chunks])
		# Paths B/C/D: same model settings, then the same normalization operation.
		placeholder = np.ones((1, 1, len(ids)), dtype=np.float32)
		b = original.run(None, original_inputs(ids, scales))[0]
		c_outputs = rewritten.run(None, ort_inputs(ids, scales, False, placeholder))
		c = c_outputs[0]
		predicted = c_outputs[1].astype(np.float32)
		d = rewritten.run(None, ort_inputs(ids, scales, True, predicted))[0]
		rows.append({
			"text_id": text,
			"phonemes": phonemes,
			"phoneme_ids": ids,
			"settings": {"noise_scale": scales[0], "length_scale": scales[1], "noise_w": scales[2], "normalize_audio": True, "volume": 1.0},
			"A_phase2s_runtime": metrics(audio_a, voice.config.sample_rate),
			"B_original_onnx_normalized": metrics(b, voice.config.sample_rate),
			"C_rewritten_disabled_normalized": metrics(c, voice.config.sample_rate),
			"D_rewritten_self_duration_normalized": metrics(d, voice.config.sample_rate),
			"predicted_durations": predicted.reshape(-1).tolist(),
		})
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
