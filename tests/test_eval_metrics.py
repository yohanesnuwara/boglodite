import unittest

import numpy as np

from boglodite_eval.metrics import categorical_metrics, nrms_percent, pearson


class EvalMetricTests(unittest.TestCase):
    def test_identical_continuous(self):
        a = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.assertAlmostEqual(nrms_percent(a, a), 0.0)
        self.assertAlmostEqual(pearson(a, a), 1.0)

    def test_categorical_perfect(self):
        a = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int8)
        m = categorical_metrics(a, a)
        self.assertAlmostEqual(m["voxel_agreement"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["mean_iou"], 1.0)
        self.assertAlmostEqual(m["cohen_kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
