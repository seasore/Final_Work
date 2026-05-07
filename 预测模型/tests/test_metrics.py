# -*- coding: utf-8 -*-
"""Unit tests for evaluation metrics in train.py — known-value reproduction."""
import os
import sys
import unittest

# Import train module from parent directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train import (  # noqa: E402
    jiSuanMAE,
    jiSuanMAPE,
    jiSuanMSE,
    jiSuanR2,
    jiSuanSMAPE,
    jiSuanWAPE,
)


class TestMetricsKnownValues(unittest.TestCase):
    def test_mse_perfect(self):
        self.assertAlmostEqual(jiSuanMSE([1.0, 2.0], [1.0, 2.0]), 0.0, places=10)

    def test_mse_hand_computed(self):
        # y=[1,3], yhat=[1,2] -> errors 0,-1 -> MSE = 0.5
        self.assertAlmostEqual(jiSuanMSE([1, 2], [1, 3]), 0.5, places=10)

    def test_mae(self):
        self.assertAlmostEqual(jiSuanMAE([0, 10], [1, 7]), 2.0, places=10)

    def test_mape(self):
        # true [100, 100], pred [99, 101] -> |1|+|1| / 200 * 100 = 1%
        self.assertAlmostEqual(jiSuanMAPE([99, 101], [100, 100]), 1.0, places=6)

    def test_r2_perfect(self):
        self.assertAlmostEqual(jiSuanR2([1, 2, 3], [1, 2, 3]), 1.0, places=6)

    def test_wape(self):
        # |10-8|+|20-22| = 4, sum|y|=30 -> 4/30*100
        self.assertAlmostEqual(jiSuanWAPE([8, 22], [10, 20]), 400 / 30, places=6)

    def test_smape_symmetry(self):
        a, b = jiSuanSMAPE([1, 2], [2, 1]), jiSuanSMAPE([2, 1], [1, 2])
        self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
