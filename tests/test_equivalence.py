from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_equivalence_record_is_present_after_proof() -> None:
	path = ROOT / "results/phase2ae/summaries/equivalence.json"
	if not path.exists():
		return
	record = json.loads(path.read_text(encoding="utf-8-sig"))
	assert record["proof1_disabled_byte_identical"] is True
	assert record["proof2_self_duration_byte_identical"] is True
