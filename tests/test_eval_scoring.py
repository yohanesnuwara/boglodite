import tempfile
import unittest
from pathlib import Path

import numpy as np

from boglodite_eval.scoring import RunMetadata, score_run
from boglodite_eval.tasks import Task


class EvalScoringTests(unittest.TestCase):
    def test_faultseg_identical_is_correct(self):
        task = Task(
            id="FTEST", tool="faultseg", prompt="", orientation="inline", coordinate=150,
            candidate_files={"prediction": "fault.npy"}, reference_files={"prediction": "fault.npy"},
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); c = root / "c"; r = root / "r"; c.mkdir(); r.mkdir()
            arr = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
            np.save(c / "fault.npy", arr); np.save(r / "fault.npy", arr)
            result = score_run(task, c, r, RunMetadata("FTEST", "boglodite", 1))
            self.assertEqual(result["primary_outcome"], "correct")

    def test_faultseg_missing_is_overt(self):
        task = Task(
            id="FTEST", tool="faultseg", prompt="", orientation="inline", coordinate=150,
            candidate_files={"prediction": "fault.npy"}, reference_files={"prediction": "fault.npy"},
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); c = root / "c"; r = root / "r"; c.mkdir(); r.mkdir()
            np.save(r / "fault.npy", np.ones((4, 4), dtype=np.float32))
            result = score_run(task, c, r, RunMetadata("FTEST", "bare", 1))
            self.assertEqual(result["primary_outcome"], "overt_failure")

    def test_malenov_changed_classes_is_silent(self):
        task = Task(
            id="MTEST", tool="malenov", prompt="", orientation="inline", coordinate=130,
            candidate_files={"classes": "classes.npy"}, reference_files={"classes": "classes.npy"},
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); c = root / "c"; r = root / "r"; c.mkdir(); r.mkdir()
            ref = np.zeros((1, 10, 10), dtype=np.int8)
            cand = ref.copy(); cand[:, :5, :] = 1
            np.save(c / "classes.npy", cand); np.save(r / "classes.npy", ref)
            result = score_run(task, c, r, RunMetadata("MTEST", "bare", 1))
            self.assertEqual(result["primary_outcome"], "silent_failure")


if __name__ == "__main__":
    unittest.main()
