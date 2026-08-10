from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_phase2af_summary_classifies_baseline_mismatch() -> None:
	text = (ROOT / "results/phase2af-findings.md").read_text(encoding="utf-8")
	assert "Result A" in text
	assert "normalize_audio=True" in text


def test_phase2ae_assignments_include_original_in_every_trial() -> None:
	key = ROOT / "results/phase2ae/raw/DO-NOT-OPEN-before-scoring-answer-key.json"
	if not key.exists():
		return
	data = json.loads(key.read_text(encoding="utf-8-sig"))
	assert len(data) == 12
	for trial in data.values():
		assert "original" in trial["assignment"].values()
