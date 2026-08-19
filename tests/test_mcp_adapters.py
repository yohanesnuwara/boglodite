"""Lightweight tests for the MCP execution adapters.

These tests intentionally do not require the F3 volume or trained models.
"""

from __future__ import annotations

import asyncio
import unittest

from boglodite_mcp import tools


class TestMCPAdapters(unittest.TestCase):
    def test_readiness_is_structured(self):
        result = tools.readiness()
        self.assertIn("faultseg", result)
        self.assertIn("malenov", result)
        self.assertIn("ready", result)
        self.assertIn("repo_root", result)

    def test_missing_prerequisites_fail_before_execution(self):
        missing = tools.REPO_ROOT / "__definitely_missing_boglodite_test_file__"
        result = asyncio.run(
            tools._run_adapter(
                name="test",
                command=["this-command-must-not-run"],
                expected_outputs=[],
                prerequisites=[missing],
            )
        )
        self.assertEqual(result["status"], "not_ready")
        self.assertEqual(result["tool"], "test")
        self.assertTrue(result["missing"])


if __name__ == "__main__":
    unittest.main()
