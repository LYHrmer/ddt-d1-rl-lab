import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools117.archive import LIMIT
from tools118 import archive


class BlindArchiveTests(unittest.TestCase):
    def records(self):
        return [dict(seed=seed,stage_status='COMPLETE',complete_horizon=True,counts=dict(LIMIT),driving_pass=True,
                     gate_pass=True,reset={'fingerprint_sha256':str(seed)},motion={'failed':[]},
                     first_failed_section=None,lidar=dict(passed=True,speed_active=True,lift_active=True))
                for seed in archive.SEEDS]

    def test_only_three_full_distinct_blind_resets_can_pass(self):
        rows = self.records()
        self.assertTrue(archive.summarize(rows,[])['all_three_blind_passed'])
        with self.assertRaisesRegex(ValueError,'reused native reset'):
            archive.summarize(rows,[str(archive.SEEDS[0])])
        rows[1]['reset'] = rows[0]['reset']
        with self.assertRaisesRegex(ValueError,'reused native reset'):
            archive.summarize(rows,[])
        with self.assertRaisesRegex(ValueError,'seed order'):
            archive.summarize(self.records()[::-1],[])

    def test_completed_early_stop_and_unknown_partial_have_separate_counts(self):
        first, early, partial = self.records()
        early.update(complete_horizon=False,driving_pass=False,gate_pass=False,
                     counts=dict(decisions=25,servo=125,physics=250,packets=3,ray_queries=3120),
                     motion={'failed':['horizon','stopped','fall_free']})
        partial.update(stage_status='PARTIAL',counts=None)
        report = archive.summarize([first,early,partial],[])
        self.assertEqual((report['complete_stages'],report['full_horizons'],report['joint_passes']), (2,1,1))
        self.assertEqual(report['early_terminated_seeds'],[early['seed']])
        self.assertEqual(report['unknown_partial_seeds'],[partial['seed']])
        self.assertEqual((report['confirmed_counts']['physics'],report['counts_upper_bound']['physics']), (6250,12250))
        self.assertFalse(report['all_three_blind_passed'])

    def test_pre_admission_and_started_partial_remain_distinct_and_never_restart(self):
        stage = archive.NATIVE_STAGES[0]
        with TemporaryDirectory() as temp:
            local, public = Path(temp)/'evidence', Path(temp)/'public'
            local.mkdir()
            unit = archive.locations(stage,1)[2]
            one, two = f'{stage}_01', f'{stage}_02'
            (local/f'admission_{one}.json').write_text(json.dumps(dict(stage=stage,attempt=1,unit=unit,
                registration_sha256='registered',child_started=False,error='headroom')))
            broken = local/two
            broken.mkdir()
            (broken/'partial.npz').write_bytes(b'non-complete trace')
            def locations(name, number):
                tag = f'{name}_{number:02}'
                return local/tag, local/('guard_'+tag), 'd1-compound118-'+tag.replace('_','-')
            with patch.multiple(archive, BASE=local, PUBLIC=public, locations=locations):
                entries = archive.collect()
                ledger = {tag:archive.attempt(tag,e,'registered') for tag,e in entries.items()}
                stages = archive.stages_for(ledger)
                self.assertEqual(stages[stage],dict(status='PARTIAL_WITHOUT_GUARD',attempt=two,admission_failures=[one]))
                self.assertEqual(entries[two]['local_partial_sha256']['partial.npz'],archive.digest(broken/'partial.npz'))
                self.assertFalse((public/stage).exists())
                ledger[stage+'_03'] = dict(ledger[two],attempt=3)
                with self.assertRaisesRegex(ValueError,'retried after starting'):
                    archive.stages_for(ledger)


if __name__ == '__main__':
    unittest.main()
