"""Verify the bundled red-phase checkpoint and its model inputs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cpl_profile import RED_PHASE, ProfileError, normalize


class TestRedPhaseProfile(unittest.TestCase):
    def test_uses_bundled_checkpoint(self):
        checkpoint = RED_PHASE.checkpoint
        self.assertEqual(checkpoint.parent.name, "assets")
        self.assertEqual(checkpoint.name, "best.ckpt")
        self.assertTrue(checkpoint.is_file())

    def test_expected_model_inputs(self):
        self.assertEqual(RED_PHASE.is_red, 1)
        self.assertEqual(RED_PHASE.gray_fixed, 5.0)
        self.assertEqual(RED_PHASE.thickness_choices, (30.0, 48.0, 80.0))
        self.assertAlmostEqual(normalize("gray", RED_PHASE.gray_fixed), 1.0)
        self.assertAlmostEqual(normalize("thickness", 48.0), 0.36)
        self.assertAlmostEqual(normalize("stretch", 100.0), 33.0 / 133.0)

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaisesRegex(ProfileError, "viscosity"):
            normalize("viscosity", 1.0)

    def test_out_of_range_parameter_is_rejected(self):
        with self.assertRaisesRegex(ProfileError, "gray"):
            normalize("gray", 5.5)

    def test_audit_metadata_records_phase_and_training_ranges(self):
        payload = RED_PHASE.to_json()
        self.assertEqual(payload["profile"], "red")
        self.assertEqual(payload["is_red"], 1)
        self.assertEqual(payload["gray_search"], [5.0, 5.0])
        self.assertEqual(payload["normalization"], {
            "gray": [1.0, 5.0],
            "thickness": [30.0, 80.0],
            "stretch": [67.0, 200.0],
        })


if __name__ == "__main__":
    unittest.main()
