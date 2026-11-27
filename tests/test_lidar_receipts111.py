import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools111 import individual
from tools111.results import digest, install
from tools111.wheel_response import sustained_from


class NativeReceipts(unittest.TestCase):
    def test_direction_must_persist_until_the_end_of_the_measured_window(self):
        flags = np.array([True, True, False, True, True])
        self.assertEqual(sustained_from(flags, 1), 3)
        self.assertEqual(sustained_from(flags, 4), 4)
        self.assertIsNone(sustained_from(~flags, 1))

    def test_reservation_keeps_an_interrupted_case_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(individual,"BASE",base):
                for unit, arm in (("development_01","neither"),("individual_111103_speed_03","speed")):
                    trial = base/unit/f"111103_{arm}"
                    trial.mkdir(parents=True)
                    (trial/"reservation.json").write_text("{}")
                    with self.assertRaisesRegex(ValueError,"already reserved"):
                        individual.unstarted(111103,arm)
                individual.unstarted(111103,"legs")

    def test_publication_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base/"source", base/"public"/"trace"
            source.write_bytes(b"validated")
            install(source,target,digest(source))
            install(source,target,digest(source))
            source.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError,"preserve existing"):
                install(source,target,digest(source))
            self.assertEqual(target.read_bytes(),b"validated")
