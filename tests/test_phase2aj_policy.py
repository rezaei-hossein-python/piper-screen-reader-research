from __future__ import annotations
import importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
for name in ("phase2ai_policy", "phase2aj_policy"):
    path = ROOT / f"experiments/onnx-duration-override/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
aj = sys.modules["phase2aj_policy"]

def test_atomic_families_cover_v6_without_consonants():
    symbols = ["^", "_", "t", "_", "ɛ", "_", "$", "_"]
    tokens = [aj.Token(i, s, i, 5) for i, s in enumerate(symbols)]
    plan, changed, fired = aj.apply_families(tokens, aj.POLICY_FAMILIES["a8"])
    assert plan[2] == 5
    assert set(fired) == {"E1", "E2", "E3", "E4", "E5"}
    assert all(plan[i] >= 1 for i in range(len(plan)))
