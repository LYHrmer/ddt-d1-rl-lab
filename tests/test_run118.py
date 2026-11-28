from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools91.launch91 import digest, write
from tools118 import run


class BlindLaunch(unittest.TestCase):
    def test_native_command_uses_approved_bounds_and_one_registered_seed(self):
        command = run.guarded_command(run.NATIVE_STAGES[0], 1)
        self.assertEqual(run.SEEDS, (2087901246, 1309793742, 1423741533))
        self.assertEqual(command[-6:], ["-m", "tools118.run", "worker", "blind_2087901246", "--attempt", "1"])
        for setting in ("Nice=10", "MemoryHigh=2G", "MemoryMax=3G", "MemorySwapMax=0", "--setenv=CUDA_VISIBLE_DEVICES="):
            self.assertIn(setting, command)
        for name in run.THREADS:
            self.assertIn(f"--setenv={name}=1", command)

    def test_guarded_partial_can_never_be_resubmitted(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "evidence/uphill89").mkdir(parents=True)
            supervisor = root / "guard.py"
            supervisor.write_text("test supervisor")
            stage = run.NATIVE_STAGES[1]
            with patch.multiple(run, ROOT=root, BASE=root/"attempts", PRESSURE=supervisor, GUARD_SHA=digest(supervisor)), \
                 patch.object(run, "protocol", return_value="registered"), patch.object(run.subprocess, "run") as dispatch:
                out, guard, _ = run.locations(stage, 1)
                out.mkdir(parents=True)
                guard.mkdir()
                write(guard/"report.json", {"status": "INCOMPLETE", "child_started": True})
                (out/"partial.npz").write_bytes(b"partial trace preserved")
                self.assertFalse(run.launch(stage))
                dispatch.assert_not_called()
                self.assertEqual((out/"partial.npz").read_bytes(), b"partial trace preserved")
                self.assertFalse(run.locations(stage, 2)[0].exists())
