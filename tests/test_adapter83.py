"""Compile actual new C++ integration against an inert protocol fixture only."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Adapter83(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("adapter83 native integration requires a C++17 compiler")
        cls.work = tempfile.TemporaryDirectory(prefix="adapter83-")
        cls.addClassCleanup(cls.work.cleanup)
        cls.binary = {}
        for name, source in (("fake", "tests/native/adapter83_fake.cpp"),
                             ("test", "tests/native/adapter83_test.cpp"),
                             ("probe", "runtime83/probe83.cpp")):
            target = Path(cls.work.name) / name
            subprocess.run(
                [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-pedantic",
                 "-ffp-contract=off", "-I", str(ROOT), str(ROOT / source), "-lcrypto",
                 "-pthread", "-o", str(target)],
                check=True, capture_output=True, text=True, timeout=60,
            )
            cls.binary[name] = target

    def test_real_core_protocol_and_reset_faults(self):
        result = subprocess.run(
            [str(self.binary["test"]), str(self.binary["fake"]),
             str(Path(self.work.name) / "exclusive_cases")],
            check=True, capture_output=True, text=True, timeout=20,
        )
        print(result.stdout.strip(), flush=True)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "PASS")
        self.assertGreater(receipt["assertions"], 80)
        self.assertTrue(receipt["fake_server_only"])
        self.assertEqual(receipt["actual_onnx_calls"], 0)
        self.assertEqual(receipt["physics_steps"], 0)
        self.assertTrue(receipt["non_executable_spawn_failure_checked"])

    def test_help_is_cold(self):
        result = subprocess.run([str(self.binary["probe"]), "--help"], check=True,
                                capture_output=True, text=True, timeout=2)
        self.assertIn("zero physics/hardware", result.stdout)


if __name__ == "__main__":
    unittest.main()
