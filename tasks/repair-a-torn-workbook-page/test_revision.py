"""Regression checks for the seam boundary and the local compression window."""

from collections import deque
import unittest

from replay import BURNISH_N, BURNISH_WINDOW, Trial
from scene import TAPE_STEP, TEAR_START
from score import repaired


class SeamTests(unittest.TestCase):
    def repair_at_width(self, width_mm):
        return {
            "liner_bonds": [False] * 3,
            "tear_gap_mm": [abs(width_mm)] * 8,
            "tear_width_mm": [width_mm] * 8,
            "tape_local": [
                [0., TEAR_START + index * TAPE_STEP, 0., 1., 1., 1.]
                for index in range(4)
            ],
            "adhesion": {"left": [True] * 4, "right": [True] * 4},
        }

    def test_edges_may_meet(self):
        for width_mm in (0., -.0000000000001, .03, .099, 2.0):
            with self.subTest(width_mm=width_mm):
                self.assertTrue(repaired(self.repair_at_width(width_mm)))

    def test_overlap_and_loosely_spaced_edges_fail(self):
        for width_mm in (-.001, -.1, 2.001):
            with self.subTest(width_mm=width_mm):
                self.assertFalse(repaired(self.repair_at_width(width_mm)))


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.region = (2, "left")
        self.trial = Trial.__new__(Trial)
        self.trial.compression_windows = {self.region: deque(maxlen=BURNISH_WINDOW)}
        self.trial.compression_integrals = {self.region: 0.}

    def step(self, force_n):
        return self.trial.compression_mean(self.region, force_n)

    def test_light_pressure_does_not_accumulate(self):
        for _ in range(2 * BURNISH_WINDOW):
            self.assertLess(self.step(.014), BURNISH_N)

    def test_spike_does_not_replace_the_press(self):
        for force_n in [30.] + [BURNISH_N / 2] * (BURNISH_WINDOW - 1):
            self.assertLess(self.step(force_n), BURNISH_N)

    def test_normal_press_must_last_a_whole_window(self):
        for _ in range(BURNISH_WINDOW - 1):
            self.assertLess(self.step(1.1 * BURNISH_N), BURNISH_N)
        self.assertGreaterEqual(self.step(1.1 * BURNISH_N), BURNISH_N)
        self.assertEqual(self.step(0.), 0.)


if __name__ == "__main__":
    unittest.main()
