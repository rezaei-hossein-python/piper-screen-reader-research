"""Content-free inspection of the locked Piper ONNX duration boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def inspect(model_path: Path) -> dict[str, object]:
	model = onnx.load(model_path, load_external_data=False)
	ceil_nodes = [node for node in model.graph.node if node.op_type == "Ceil"]
	if len(ceil_nodes) != 1:
		raise ValueError(f"expected one Ceil node, found {len(ceil_nodes)}")
	ceil_output = ceil_nodes[0].output[0]
	consumers = [node.op_type for node in model.graph.node if ceil_output in node.input]
	return {
		"bytes": model_path.stat().st_size,
		"sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
		"opset": [item.version for item in model.opset_import],
		"node_count": len(model.graph.node),
		"inputs": [item.name for item in model.graph.input],
		"outputs": [item.name for item in model.graph.output],
		"ceil_output": ceil_output,
		"ceil_consumers": consumers,
	}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type=Path)
	args = parser.parse_args()
	print(json.dumps(inspect(args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
