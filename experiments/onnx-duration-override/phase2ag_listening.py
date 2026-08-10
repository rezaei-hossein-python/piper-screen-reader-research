"""Generate the four-item corrected Phase 2AG blinded listening gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice


ITEMS = ["F", "S", "A", "button"]
SCALES = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)


def normalized(values: np.ndarray) -> np.ndarray:
    result = values.reshape(-1).astype(np.float32)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak >= 1e-8:
        result = result / peak
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, values: np.ndarray, sample_rate: int) -> None:
    pcm = (normalized(values) * 32767.0).round().astype("<i2").tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def metrics(values: np.ndarray, sample_rate: int) -> dict[str, object]:
    audio = normalized(values)
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "samples": int(audio.size),
        "duration_ms": float(audio.size * 1000.0 / sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "finite": bool(np.isfinite(audio).all()),
        "clipped": bool(np.any(np.abs(audio) > 1.0)),
        "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("rewritten", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("answer_key", type=Path)
    args = parser.parse_args()

    voice = PiperVoice.load(str(args.model), str(args.config))
    rewritten = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
    reverse = {value[0]: key for key, value in voice.config.phoneme_id_map.items()}
    rng = random.Random(20260809)
    args.output.mkdir(parents=True, exist_ok=True)
    key: dict[str, object] = {}
    rows: list[dict[str, object]] = []

    for item in ITEMS:
        phonemes = voice.phonemize(item)[0]
        ids = voice.phonemes_to_ids(phonemes)
        labels = [reverse.get(value, "?") for value in ids]
        inputs = {
            "input": np.asarray([ids], dtype=np.int64),
            "input_lengths": np.asarray([len(ids)], dtype=np.int64),
            "scales": SCALES,
        }
        # Use one override-disabled rewritten invocation as the authoritative
        # original for this trial. It exposes the exact predicted vector used
        # by that waveform, avoiding a different stochastic duration draw from
        # a second ONNX session while retaining the proven original path.
        probe = dict(inputs)
        probe["duration_override"] = np.ones((1, 1, len(ids)), dtype=np.float32)
        probe["duration_override_enabled"] = np.asarray(False, dtype=np.bool_)
        disabled_outputs = rewritten.run(None, probe)
        base = disabled_outputs[0]
        predicted = disabled_outputs[1].astype(np.float32)
        durations = predicted.reshape(-1).astype(np.int64)

        # Prefer a silence separator only if it has enough occupancy; otherwise
        # use the first sufficiently long vowel nucleus.  These are diagnostic,
        # not production policy rules.
        candidates = [i for i, label in enumerate(labels) if label == "_" and durations[i] >= 3]
        vowel_labels = set("aeiouɐɑɒəɛɜɪɔɵʊʌɐɜɞɪʉɘɤøœɶ")
        candidates += [i for i, label in enumerate(labels) if label in vowel_labels and durations[i] >= 3]
        if not candidates:
            raise RuntimeError(f"no safe diagnostic token for {item}: {labels} {durations.tolist()}")
        token = candidates[0]
        available = int(durations[token] - 1)
        reduction2 = 2 if available >= 2 else 1

        variants: list[tuple[str, np.ndarray, int]] = [("original", base, 0)]
        for name, reduction in (("c1", 1), ("c2", reduction2)):
            modified = durations.copy()
            modified[token] -= reduction
            override = modified.astype(np.float32).reshape(1, 1, -1)
            changed = dict(inputs)
            changed["duration_override"] = override
            changed["duration_override_enabled"] = np.asarray(True, dtype=np.bool_)
            audio = rewritten.run(None, changed)[0]
            variants.append((name, audio, reduction))

        shuffled = list(variants)
        rng.shuffle(shuffled)
        trial = f"trial-{len(rows) + 1:02d}"
        assignment: dict[str, str] = {}
        variant_metrics: dict[str, object] = {}
        for suffix, (name, audio, reduction) in zip(("a", "b", "c"), shuffled):
            filename = f"{trial}-{suffix}.wav"
            write_wav(args.output / filename, audio, voice.config.sample_rate)
            assignment[suffix] = name
            variant_metrics[name] = {"reduction_frames": reduction, **metrics(audio, voice.config.sample_rate)}
        key[trial] = {"source_item": item, "assignment": assignment}
        rows.append({
            "item": item,
            "phonemes": phonemes,
            "labels": labels,
            "original_durations": durations.tolist(),
            "modified_token": token,
            "modified_label": labels[token],
            "c1_durations": (durations - np.eye(1, len(durations), token, dtype=np.int64).reshape(-1)).tolist(),
            "c2_durations": (durations - np.eye(1, len(durations), token, dtype=np.int64).reshape(-1) * reduction2).tolist(),
            "variants": variant_metrics,
        })

    (args.output / "instructions.txt").write_text(
        "For each trial, listen to A, B, and C. Record which sounds best and whether the other versions are indistinguishable, slightly worse, or clearly worse. Do not open the answer key.\n",
        encoding="utf-8",
    )
    args.answer_key.parent.mkdir(parents=True, exist_ok=True)
    args.answer_key.write_text(json.dumps(key, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "phase2ag-measurements.json").write_text(json.dumps({"settings": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8, "normalize_audio": True, "volume": 1.0, "sample_rate": voice.config.sample_rate}, "items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
