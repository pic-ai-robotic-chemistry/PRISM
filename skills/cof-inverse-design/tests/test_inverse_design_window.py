"""Tests for the peak-wavelength window in the inverse-design objective.

The window exists because an intensity-only objective leaves the wavelength
free: the previous blue run landed at 502 nm purely because nothing scored it.
The constraint is enforced by *ordering* -- every out-of-window candidate scores
strictly worse than every in-window one -- rather than by a hand-tuned penalty
weight, because a penalty strong enough to dominate a large intensity term
turns the surrogate landscape into a cliff. These tests pin that ordering.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import inverse_design as idm  # noqa: E402

PARAM_NAMES = ["thickness", "stretch"]
POINT = [48.0, 100.0]


class StubModel:
    """A stand-in whose spectrum peaks exactly where the test asks."""

    def __init__(self, peak_wl: float, peak_int: float):
        self.peak_wl = peak_wl
        self.peak_int = peak_int

    def predict_spectrum(self, color_idx, gray, thickness, stretch, is_red):
        import torch

        spectrum = torch.zeros(1, len(idm.WAVELENGTHS), dtype=torch.float32)
        idx = int(np.argmin(np.abs(idm.WAVELENGTHS - self.peak_wl)))
        spectrum[0, idx] = self.peak_int
        return spectrum


def objective_for(peak_wl, peak_int, window=None, target_int=50000.0, target_wl=None):
    objective, history = idm.make_objective(
        StubModel(peak_wl, peak_int), "B", PARAM_NAMES,
        target_wl, target_int, 0.01, "cpu", wl_window=window,
    )
    return objective, history


class TestWindowCeiling(unittest.TestCase):
    def test_ceiling_is_finite_and_positive(self):
        self.assertGreater(idm.window_ceiling(None, 50000.0, 0.01), 0.0)

    def test_ceiling_dominates_every_achievable_in_window_loss(self):
        """The ordering only holds if the ceiling really is above the in-window range."""
        target_int, weight = 50000.0, 0.01
        ceiling = idm.window_ceiling(None, target_int, weight)
        worst_in_window = max(
            weight * (peak - target_int) ** 2
            for peak in (-idm.INTENSITY_CEILING_MDEG, -12000.0, 0.0, 12000.0, idm.INTENSITY_CEILING_MDEG)
        )
        self.assertGreater(ceiling, worst_in_window)


class TestWindowOrdering(unittest.TestCase):
    WINDOW = (450.0, 495.0)

    def test_in_window_point_keeps_the_plain_intensity_loss(self):
        objective, history = objective_for(480.0, 6000.0, self.WINDOW)
        loss = objective(POINT)
        expected = 0.01 * (6000.0 - 50000.0) ** 2
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertTrue(history[-1]["in_window"])

    def test_out_of_window_point_is_penalised_above_every_in_window_point(self):
        ceiling = idm.window_ceiling(None, 50000.0, 0.01)
        inside, _ = objective_for(480.0, 6000.0, self.WINDOW)
        outside, history = objective_for(502.0, 6100.0, self.WINDOW)
        self.assertGreater(outside(POINT), ceiling)
        self.assertLess(inside(POINT), ceiling)
        self.assertFalse(history[-1]["in_window"])

    def test_a_stronger_out_of_window_candidate_still_loses(self):
        """This is the exact blue case: 6100 mdeg at 502 nm vs 6000 at 480 nm."""
        inside, _ = objective_for(480.0, 6000.0, self.WINDOW)
        outside, _ = objective_for(502.0, 6100.0, self.WINDOW)
        self.assertLess(inside(POINT), outside(POINT), "the stronger but off-colour point won")

    def test_penalty_grows_with_distance_from_the_window(self):
        near, _ = objective_for(497.0, 6000.0, self.WINDOW)
        far, _ = objective_for(540.0, 6000.0, self.WINDOW)
        self.assertLess(near(POINT), far(POINT))

    def test_boundaries_are_inclusive(self):
        for edge in self.WINDOW:
            objective, history = objective_for(float(edge), 6000.0, self.WINDOW)
            objective(POINT)
            self.assertTrue(history[-1]["in_window"], f"{edge} nm should count as inside")

    def test_no_window_reproduces_the_previous_behaviour(self):
        """Without a window the wavelength is free, which is what caused the drift."""
        objective, history = objective_for(502.0, 6100.0, None)
        self.assertAlmostEqual(objective(POINT), 0.01 * (6100.0 - 50000.0) ** 2, places=6)
        self.assertTrue(history[-1]["in_window"])

    def test_window_also_applies_when_a_wavelength_target_is_set(self):
        objective, history = objective_for(502.0, 6000.0, self.WINDOW, target_wl=480.0)
        self.assertGreater(objective(POINT), idm.window_ceiling(480.0, 50000.0, 0.01))
        self.assertFalse(history[-1]["in_window"])


class TestWindowCli(unittest.TestCase):
    def test_window_is_optional(self):
        args = idm.parse_args(["--color", "B", "-t", "480"])
        self.assertIsNone(args.wl_window)

    def test_window_is_parsed(self):
        args = idm.parse_args(["--color", "B", "-t", "480", "--wl-window", "450", "495"])
        self.assertEqual(args.wl_window, [450.0, 495.0])

    def test_inverted_window_is_rejected(self):
        with self.assertRaises(SystemExit):
            idm.parse_args(["--color", "B", "-t", "480", "--wl-window", "495", "450"])

    def test_degenerate_window_is_rejected(self):
        with self.assertRaises(SystemExit):
            idm.parse_args(["--color", "B", "-t", "480", "--wl-window", "480", "480"])

    def test_window_outside_the_modelled_range_is_rejected(self):
        with self.assertRaises(SystemExit):
            idm.parse_args(["--color", "B", "-t", "480", "--wl-window", "300", "495"])

    def test_window_requires_two_values(self):
        with self.assertRaises(SystemExit):
            idm.parse_args(["--color", "B", "-t", "480", "--wl-window", "450"])


if __name__ == "__main__":
    unittest.main()
