from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("phase2ai_policy", ROOT / "experiments/onnx-duration-override/phase2ai_policy.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_actual_lessac_symbols_are_classified() -> None:
    expected = {
        "_": "padding", "^": "boundary/silence", "$": "boundary/silence",
        "ˈ": "stress/control marker", "ː": "stress/control marker",
        "ɪ": "vowel", "ɛ": "vowel", "ʌ": "vowel", "ə": "vowel",
        "ʃ": "fricative", "ŋ": "nasal", "ɹ": "liquid", "w": "glide",
    }
    assert {symbol: MODULE.classify_token(symbol) for symbol in expected} == expected


def test_v1_is_exact_first_pad_policy() -> None:
    tokens = [MODULE.Token(i, s, i, f) for i, (s, f) in enumerate(zip("^_a_$", [4, 3, 5, 4, 5]))]
    plan, changed = MODULE.apply_policy(tokens, "v1")
    assert plan == [4, 2, 5, 4, 5]
    assert changed == [1]


def test_v6_never_modifies_consonant_or_stress() -> None:
    symbols = ["^", "_", "t", "_", "ˈ", "_", "ɛ", "_", "$", "_"]
    tokens = [MODULE.Token(i, s, i, 5) for i, s in enumerate(symbols)]
    plan, changed = MODULE.apply_policy(tokens, "v6")
    assert 2 not in changed and 4 not in changed
    assert plan[2] == 5 and plan[4] == 5
    assert all(value >= 1 for value in plan)


def test_cumulative_stages_do_not_repeat_earlier_reduction() -> None:
    symbols = ["^", "_", "ɛ", "_", "a", "_", "$"]
    tokens = [MODULE.Token(i, s, i, 5) for i, s in enumerate(symbols)]
    v1, _ = MODULE.apply_policy(tokens, "v1")
    v2, _ = MODULE.apply_policy(tokens, "v2")
    v3, _ = MODULE.apply_policy(tokens, "v3")
    v4, _ = MODULE.apply_policy(tokens, "v4")
    assert v1[1] == 4 and v2[1] == 4
    assert v3[2] == 4 and v4[2] == 4
    assert v4[4] == 4
