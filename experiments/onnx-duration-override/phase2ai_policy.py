"""Unicode-safe Piper/eSpeak token classification and Phase 2AI policies."""

from __future__ import annotations

from dataclasses import dataclass


BOUNDARY = {"^", "$"}
PADDING = {"_"}
PUNCTUATION = {" ", "!", "'", "(", ")", ",", "-", ".", ":", ";", "?", "#", '"'}
STRESS_CONTROL = {"ˈ", "ˌ", "ː", "ˑ", "˞", "ʰ", "ˤ", "↓", "↑", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "̧", "̃", "̪", "̯", "̩", "̺", "̻"}
VOWELS = set("aeiouyæøœɐɑɒɔɘəɚɛɜɞɤɨɪɯɵɶʉʊʌ")
STOPS = set("pbtdkqɡɢʔcɟʈɖɓɗɠʛ")
FRICATIVES = set("fvszθðʃʒçʝxɣχʁħʕhɦɸβɕʑʂʐɧɬɮ")
AFFRICATES = {"ʦ", "ʣ", "ʧ", "ʤ", "ʨ", "ʥ", "ʈ͡ʂ", "ɖ͡ʐ"}
NASALS = set("mnŋɲɳɴɱ")
LIQUIDS = set("lrɹɻɽɾʀʁɭɺ")
GLIDES = set("wjɥɰʋ")


def classify_token(symbol: str) -> str:
    if symbol in PADDING:
        return "padding"
    if symbol in BOUNDARY:
        return "boundary/silence"
    if symbol in PUNCTUATION:
        return "punctuation/boundary"
    if symbol in STRESS_CONTROL:
        return "stress/control marker"
    if symbol in VOWELS:
        return "vowel"
    if symbol in STOPS:
        return "stop"
    if symbol in FRICATIVES:
        return "fricative"
    if symbol in AFFRICATES:
        return "affricate"
    if symbol in NASALS:
        return "nasal"
    if symbol in LIQUIDS:
        return "liquid"
    if symbol in GLIDES:
        return "glide"
    # A token present in an eSpeak phoneme stream is conservatively speech-bearing
    # until source evidence proves otherwise.
    return "other speech-bearing"


@dataclass(frozen=True)
class Token:
    index: int
    symbol: str
    token_id: int
    frames: int


def _dec(result: list[int], index: int, amount: int = 1) -> None:
    result[index] -= min(amount, max(0, result[index] - 1))


def apply_policy(tokens: list[Token], policy: str) -> tuple[list[int], list[int]]:
    """Return duration plan and modified indices; never touch consonants/unknowns."""
    result = [token.frames for token in tokens]
    modified: list[int] = []

    def reduce(index: int, amount: int = 1) -> None:
        before = result[index]
        _dec(result, index, amount)
        if result[index] != before and index not in modified:
            modified.append(index)

    pads = [t.index for t in tokens if classify_token(t.symbol) == "padding" and t.frames >= 2]
    vowels = [t.index for t in tokens if classify_token(t.symbol) == "vowel" and t.frames >= 4]
    bos_eos = [t.index for t in tokens if t.symbol in BOUNDARY and t.frames >= 2]

    if policy == "p0":
        return result, modified
    # V1 is exactly Phase 2AH P1: first eligible PAD/separator, one frame.
    if pads:
        reduce(pads[0])
    if policy == "v1":
        return result, modified
    # V2: all learned PAD separators plus explicit BOS/EOS boundaries, one frame.
    for index in pads:
        if index not in modified:
            reduce(index)
    for index in bos_eos:
        reduce(index)
    if policy == "v2":
        return result, modified
    # V3: one sufficiently long vowel, one frame.
    if vowels:
        reduce(max(vowels, key=lambda i: result[i]))
    if policy == "v3":
        return result, modified
    # V4: repeat the same bounded operation for every sufficiently long vowel.
    for index in vowels:
        if index not in modified:
            reduce(index)
    if policy == "v4":
        return result, modified
    # V5: one more frame from demonstrated terminal non-speech occupancy only.
    terminal = [t.index for t in tokens if t.symbol == "$" or (t.symbol == "_" and t.index > len(tokens) - 4)]
    for index in terminal:
        reduce(index)
    if policy in {"v5", "v6"}:
        return result, modified
    raise ValueError(f"unknown policy: {policy}")
