"""Measure the bounded Phase 2AH Piper duration-policy ladder."""

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

from duration_probe import validate_override


ITEMS = [
    "A", "E", "F", "K", "M", "R", "S", "T", "X",
    "1", "5", "7", "9", "comma", "period", "question mark", "exclamation mark",
    "button", "selected", "checked", "edit", "link", "heading", "save",
]
SCALES = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)
VOWELS = set("aeiouɐɑɒəɛɜɪɔɵʊʌɐɞɘɤøœɶ")


def normalize(values: np.ndarray) -> np.ndarray:
    result = values.reshape(-1).astype(np.float32)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak >= 1e-8:
        result = result / peak
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, values: np.ndarray, sample_rate: int) -> None:
    pcm = (normalize(values) * 32767.0).round().astype("<i2").tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def metric(values: np.ndarray, sample_rate: int) -> dict[str, object]:
    audio = normalize(values)
    return {
        "sample_rate": sample_rate,
        "samples": int(audio.size),
        "duration_ms": float(audio.size * 1000 / sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "finite": bool(np.isfinite(audio).all()),
        "clipped": bool(np.any(np.abs(audio) > 1.0)),
        "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
    }


def indices(labels: list[str], durations: np.ndarray) -> tuple[list[int], list[int]]:
    separators = [i for i, label in enumerate(labels) if label == "_" and int(durations[i]) >= 2]
    vowels = [i for i, label in enumerate(labels) if label in VOWELS and int(durations[i]) >= 2]
    return separators, vowels


def apply_policy(durations: np.ndarray, separators: list[int], vowels: list[int], policy: str) -> np.ndarray:
    result = durations.astype(np.int64, copy=True)
    if policy == "p0":
        return result
    if policy in {"p1", "p2"} and separators:
        result[separators[0]] -= 1 if policy == "p1" else min(2, int(result[separators[0]]) - 1)
    elif policy == "p3":
        for index in separators:
            result[index] -= 1
    elif policy == "p4" and separators:
        index = separators[-1]
        result[index] -= min(2, int(result[index]) - 1)
    elif policy == "p5" and vowels:
        index = max(vowels, key=lambda item: int(result[item]))
        result[index] -= 1
    elif policy == "p6":
        for index in separators:
            result[index] -= 1
        if vowels:
            index = max(vowels, key=lambda item: int(result[item]))
            result[index] -= 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("rewritten", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("listening", type=Path)
    parser.add_argument("answer_key", type=Path)
    args = parser.parse_args()
    voice = PiperVoice.load(str(args.model), str(args.config))
    session = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
    reverse = {value[0]: key for key, value in voice.config.phoneme_id_map.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.listening.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260810)
    policy_names = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]
    all_rows: list[dict[str, object]] = []

    for item in ITEMS:
        phonemes = voice.phonemize(item)[0]
        ids = voice.phonemes_to_ids(phonemes)
        labels = [reverse.get(value, "?") for value in ids]
        inputs = {
            "input": np.asarray([ids], dtype=np.int64),
            "input_lengths": np.asarray([len(ids)], dtype=np.int64),
            "scales": SCALES,
            "duration_override": np.ones((1, 1, len(ids)), dtype=np.float32),
            "duration_override_enabled": np.asarray(False, dtype=np.bool_),
        }
        base_outputs = session.run(None, inputs)
        original = base_outputs[0]
        predicted = base_outputs[1].astype(np.float32).reshape(-1)
        separators, vowels = indices(labels, predicted)
        row: dict[str, object] = {"item": item, "phonemes": phonemes, "labels": labels, "original_durations": predicted.astype(int).tolist(), "separator_indices": separators, "vowel_indices": vowels, "policies": {}}
        generated: dict[str, np.ndarray] = {"p0": original}
        for policy in policy_names[1:]:
            modified = apply_policy(predicted, separators, vowels, policy)
            override = modified.astype(np.float32).reshape(1, 1, -1)
            validate_override(override, predicted.reshape(1, 1, -1))
            changed = dict(inputs)
            changed["duration_override"] = override
            changed["duration_override_enabled"] = np.asarray(True, dtype=np.bool_)
            generated[policy] = session.run(None, changed)[0]
            row["policies"][policy] = {"durations": modified.tolist(), "frames_removed": int(predicted.sum() - modified.sum())}
        for policy, audio in generated.items():
            row["policies"].setdefault(policy, {})
            modified = predicted if policy == "p0" else np.asarray(row["policies"][policy]["durations"])
            row["policies"][policy].update(metric(audio, voice.config.sample_rate))
            row["policies"][policy]["milliseconds_removed"] = float((predicted.sum() - modified.sum()) * 256 * 1000 / 16000)
        # Keep compact measurements for every policy; Q/R are selected below.
        all_rows.append(row)

    # Q is the strongest one-frame-first policy; R is the cleanest strongest
    # policy with additional savings. Both remain bounded and separator/vowel
    # only. The measurements, not listening preference, determine their scope.
    q_policy, r_policy = "p1", "p6"
    chosen_items = ["F", "S", "A", "7", "question mark", "button", "selected", "heading"]
    key: dict[str, object] = {}
    for row in all_rows:
        if row["item"] not in chosen_items:
            continue
        item = row["item"]
        variants = [("original", row["policies"]["p0"]), ("q", row["policies"][q_policy]), ("r", row["policies"][r_policy])]
        # Re-render selected WAVs from fresh graph calls using the same plans.
        ids = voice.phonemes_to_ids(row["phonemes"])
        base_inputs = {"input": np.asarray([ids], np.int64), "input_lengths": np.asarray([len(ids)], np.int64), "scales": SCALES, "duration_override": np.ones((1, 1, len(ids)), np.float32), "duration_override_enabled": np.asarray(False, np.bool_)}
        audio_map = {"original": session.run(None, base_inputs)[0]}
        for name, policy in (("q", q_policy), ("r", r_policy)):
            override = np.asarray(row["policies"][policy]["durations"], dtype=np.float32).reshape(1, 1, -1)
            request = dict(base_inputs); request["duration_override"] = override; request["duration_override_enabled"] = np.asarray(True, np.bool_)
            audio_map[name] = session.run(None, request)[0]
        shuffled = [(name, audio_map[name]) for name, _ in variants]
        rng.shuffle(shuffled)
        trial = f"trial-{len(key)+1:02d}"
        assignment = {}
        for suffix, (name, audio) in zip(("a", "b", "c"), shuffled):
            filename = f"{trial}-{suffix}.wav"
            write_wav(args.listening / filename, audio, voice.config.sample_rate)
            assignment[suffix] = name
        key[trial] = {"source_item": item, "assignment": assignment}
    (args.output / "phase2ah-measurements.json").write_text(json.dumps({"settings": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8, "normalize_audio": True, "sample_rate": 16000}, "q_policy": q_policy, "r_policy": r_policy, "items": all_rows}, indent=2), encoding="utf-8")
    args.answer_key.parent.mkdir(parents=True, exist_ok=True)
    args.answer_key.write_text(json.dumps(key, indent=2), encoding="utf-8")
    (args.listening / "instructions.txt").write_text("For each trial, listen to A, B, and C. Record best: A/B/C and whether the others are indistinguishable, slightly worse, or clearly worse. Flag any pronunciation improvement or degradation.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
