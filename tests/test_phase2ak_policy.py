from __future__ import annotations
import importlib.util,sys
from pathlib import Path
root=Path(__file__).parents[1]
for name in ("phase2ai_policy","phase2aj_policy"):
 spec=importlib.util.spec_from_file_location(name,root/f"experiments/onnx-duration-override/{name}.py");assert spec and spec.loader;mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
def test_a5_is_frozen_e1_e2_e3():
 from phase2aj_policy import Token,apply_families
 tokens=[Token(i,s,i,5) for i,s in enumerate(["^","_","t","_","ɛ","_","$"])]
 plan,_,fired=apply_families(tokens,{"E1","E2","E3"});assert plan[2]==5;assert all(fired[k] is not None for k in ("E1","E2","E3"))
