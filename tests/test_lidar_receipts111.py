import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools111 import individual
from tools111.results import digest, install


class NativeReceipts(unittest.TestCase):
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
