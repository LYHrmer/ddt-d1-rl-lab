"""Compile/run only the pure C++ argv contract; no ROS or model dependency."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Cli83(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("cli83 contract requires a C++17 compiler")
        cls.work = tempfile.TemporaryDirectory(prefix="cli83-cold-")
        cls.addClassCleanup(cls.work.cleanup)
        cls.binary = Path(cls.work.name) / "cli83_contract"
        subprocess.run(
            [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
             "-I", str(ROOT), str(ROOT / "tests/native/cli83_test.cpp"), "-o", str(cls.binary)],
            check=True, capture_output=True, text=True, timeout=60,
        )

    def test_complete_seven_pairs_and_rejections(self):
        result = subprocess.run(
            [str(self.binary), str(ROOT / "runtime83/ros_probe83.cpp")],
            check=True, capture_output=True, text=True, timeout=10,
        )
        print(result.stdout.strip(), flush=True)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "PASS")
        self.assertGreater(receipt["assertions"], 40)
        self.assertEqual(receipt["actor_calls"], 0)
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(receipt["physics_steps"], 0)
        self.assertFalse(receipt["ros_initialized"])


if __name__ == "__main__":
    unittest.main()
