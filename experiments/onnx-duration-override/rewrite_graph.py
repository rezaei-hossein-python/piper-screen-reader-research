"""Rewrite a Piper VITS graph with a host-validated duration selector."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import TensorProto, helper


PREDICTED = "/Ceil_output_0"
EFFECTIVE = "/Phase2AEDuration"
OVERRIDE = "duration_override"
ENABLED = "duration_override_enabled"


def _value_info(name: str, elem_type: int, shape: list[object] | None) -> onnx.ValueInfoProto:
	return helper.make_tensor_value_info(name, elem_type, shape)


def rewrite(input_path: Path, output_path: Path) -> dict[str, object]:
	model = onnx.load(input_path, load_external_data=False)
	ceil_nodes = [node for node in model.graph.node if node.op_type == "Ceil"]
	if len(ceil_nodes) != 1 or ceil_nodes[0].output[0] != PREDICTED:
		raise ValueError("expected the pinned single /Ceil_output_0 boundary")
	consumers = [i for i, node in enumerate(model.graph.node) if PREDICTED in node.input]
	if not consumers:
		raise ValueError("duration tensor has no downstream consumers")
	graph_inputs = {item.name for item in model.graph.input}
	if OVERRIDE not in graph_inputs:
		model.graph.input.append(_value_info(OVERRIDE, TensorProto.FLOAT, ["batch_size", 1, "phonemes"]))
	if ENABLED not in graph_inputs:
		model.graph.input.append(_value_info(ENABLED, TensorProto.BOOL, []))
	selector = helper.make_node(
		"Where", [ENABLED, OVERRIDE, PREDICTED], [EFFECTIVE], name="/Phase2AE/SelectDuration"
	)
	first_consumer = min(consumers)
	nodes = list(model.graph.node)
	for node in nodes:
		for index, input_name in enumerate(node.input):
			if input_name == PREDICTED:
				node.input[index] = EFFECTIVE
	nodes.insert(first_consumer, selector)
	model.graph.node.clear()
	model.graph.node.extend(nodes)
	output_names = {item.name for item in model.graph.output}
	if PREDICTED not in output_names:
		model.graph.output.append(_value_info(PREDICTED, TensorProto.FLOAT, ["batch_size", 1, "phonemes"]))
	if EFFECTIVE not in output_names:
		model.graph.output.append(_value_info(EFFECTIVE, TensorProto.FLOAT, ["batch_size", 1, "phonemes"]))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	onnx.checker.check_model(model)
	onnx.save(model, output_path)
	return {
		"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
		"output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
		"predicted_tensor": PREDICTED,
		"effective_tensor": EFFECTIVE,
		"selector": "Where(enabled, override, predicted)",
		"original_consumers_rewired": len(consumers),
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("input", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	print(rewrite(args.input, args.output))


if __name__ == "__main__":
	main()
