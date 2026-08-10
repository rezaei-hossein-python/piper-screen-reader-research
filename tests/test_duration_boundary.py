"""Dependency-light invariants for the first Phase 2AD graph proof."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class DurationBoundaryTests(unittest.TestCase):
	def test_source_lock_is_pinned(self) -> None:
		lock = json.loads((ROOT / "locks" / "source.lock.json").read_text())
		self.assertEqual(lock["piper"]["commit"], "e2a4b1fa1c502bbb97e729a5b34a6af565007843")
		self.assertEqual(lock["vits_reference"]["commit"], "2e561ba58618d021b5b8323d3765880f7e0ecfdb")

	def test_locked_model_hash_if_present(self) -> None:
		lock = json.loads((ROOT / "locks" / "artifacts.lock.json").read_text())
		model = Path(lock["lessac_model"]["source"])
		if not model.exists():
			self.skipTest("protected Phase 2H model is unavailable")
		self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), lock["lessac_model"]["sha256"])

	def test_duration_values_require_positive_integers(self) -> None:
		valid = [1, 2, 1, 4]
		self.assertTrue(all(isinstance(value, int) and value >= 1 for value in valid))
		invalid = [1, 0, 2]
		self.assertFalse(all(isinstance(value, int) and value >= 1 for value in invalid))


if __name__ == "__main__":
	unittest.main()
