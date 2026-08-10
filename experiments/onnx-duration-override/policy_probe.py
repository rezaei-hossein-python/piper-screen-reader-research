"""Conservative language-neutral duration policy for a small proof corpus."""

from __future__ import annotations

import math


VOWELS = set("aeiouɑæəɛɪɔʊʌɒɜɞɐɨɵɤ")
STOPS = set("pbtdkgʔ")
FRICATIVES = set("fvszʃʒθðh")
AFFRICATES = {"tʃ", "dʒ"}
NASALS = set("mnŋ")
LIQUIDS = set("lrwj")


def classify(symbol: str) -> str:
	if symbol in {"", "_", "sil", "sp", "PAD"}:
		return "silence"
	if symbol in AFFRICATES:
		return "affricate"
	if symbol in VOWELS:
		return "vowel"
	if symbol in STOPS:
		return "stop"
	if symbol in FRICATIVES:
		return "fricative"
	if symbol in NASALS:
		return "nasal"
	if symbol in LIQUIDS:
		return "liquid_glide"
	return "unknown"


def conservative(symbols: list[str], durations: list[int]) -> list[int]:
	if len(symbols) != len(durations):
		raise ValueError("symbol/duration length mismatch")
	result: list[int] = []
	for symbol, duration in zip(symbols, durations):
		if duration < 0:
			raise ValueError("negative duration")
		kind = classify(symbol)
		if kind == "vowel" and duration > 4:
			result.append(max(1, math.floor(duration * 0.8)))
		elif kind == "silence":
			result.append(0 if duration == 0 else max(1, math.floor(duration * 0.5)))
		else:
			result.append(duration)
	return result
