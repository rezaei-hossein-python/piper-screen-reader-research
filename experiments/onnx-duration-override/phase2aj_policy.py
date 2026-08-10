"""Atomic Phase 2AJ Piper duration edit families."""
from __future__ import annotations
from phase2ai_policy import Token, classify_token


def eligible(tokens: list[Token]) -> dict[str, list[int]]:
    pads = [t.index for t in tokens if classify_token(t.symbol) == "padding" and t.frames >= 2]
    vowels = [t.index for t in tokens if classify_token(t.symbol) == "vowel" and t.frames >= 4]
    boundaries = [t.index for t in tokens if t.symbol in {"^", "$"} and t.frames >= 2]
    terminal = [t.index for t in tokens if t.symbol == "$" or (t.symbol == "_" and t.index > len(tokens) - 4)]
    return {"pads": pads, "boundaries": boundaries, "vowels": vowels, "terminal": terminal}


def apply_families(tokens: list[Token], families: set[str]) -> tuple[list[int], list[int], dict[str, list[int]]]:
    result = [t.frames for t in tokens]
    fired = {name: [] for name in ("E1", "E2", "E3", "E4", "E5")}
    groups = eligible(tokens)

    def reduce(name: str, index: int) -> None:
        if result[index] > 1:
            result[index] -= 1
            fired[name].append(index)

    if "E1" in families and groups["pads"]:
        reduce("E1", groups["pads"][0])
    if "E2" in families:
        for index in groups["pads"]:
            if index not in fired["E1"]:
                reduce("E2", index)
        for index in groups["boundaries"]:
            reduce("E2", index)
    if "E3" in families:
        for index in groups["terminal"]:
            reduce("E3", index)
    if "E4" in families and groups["vowels"]:
        reduce("E4", max(groups["vowels"], key=lambda i: result[i]))
    if "E5" in families:
        for index in groups["vowels"]:
            if index not in fired["E4"]:
                reduce("E5", index)
    changed = [i for i, (a, b) in enumerate(zip((t.frames for t in tokens), result)) if a != b]
    return result, changed, fired


POLICY_FAMILIES = {
    "a0": set(), "a1": {"E1"}, "a2": {"E1", "E2"}, "a3": {"E1", "E3"},
    "a4": {"E1", "E4"}, "a5": {"E1", "E2", "E3"}, "a6": {"E1", "E2", "E4"},
    "a7": {"E1", "E3", "E4"}, "a8": {"E1", "E2", "E3", "E4", "E5"},
}
