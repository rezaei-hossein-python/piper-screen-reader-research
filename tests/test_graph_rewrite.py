from __future__ import annotations

import importlib.util
from pathlib import Path

import onnx


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("rewrite_graph", ROOT / "experiments/onnx-duration-override/rewrite_graph.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selector_contract_on_generated_graph() -> None:
	model_path = ROOT / "models/generated/lessac-duration-override.onnx"
	if not model_path.exists():
		return
	model = onnx.load(model_path, load_external_data=False)
	assert "duration_override" in {item.name for item in model.graph.input}
	assert "duration_override_enabled" in {item.name for item in model.graph.input}
	assert "/Ceil_output_0" in {item.name for item in model.graph.output}
	assert "/Phase2AEDuration" in {item.name for item in model.graph.output}
	selectors = [node for node in model.graph.node if node.name == "/Phase2AE/SelectDuration"]
	assert len(selectors) == 1
	assert selectors[0].op_type == "Where"
	assert selectors[0].input == ["duration_override_enabled", "duration_override", "/Ceil_output_0"]


def test_original_topology_is_not_assumed_silently() -> None:
	assert MODULE.PREDICTED == "/Ceil_output_0"
	assert MODULE.EFFECTIVE == "/Phase2AEDuration"
