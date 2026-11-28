import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools117 import archive
from tools117 import parking_video_check


class ResponseArchiveTests(unittest.TestCase):
    def setUp(self):
        self.attempts = {
            name+'_01':dict(stage=name, attempt=1, status='COMPLETE',
                            reserved_at_unix_ns=10*i+1, completed_at_unix_ns=10*i+2)
            for i,name in enumerate(archive.SPEED_STAGES)
        }
        self.cases = [dict(seed=seed, stage_status='COMPLETE', complete_horizon=True, counts=dict(archive.LIMIT),
                           driving_pass=True, gate_pass=True) for seed in archive.SEEDS]

    def test_only_parking_failure_opens_second_stage_after_all_three_complete(self):
        stages = archive.select(self.attempts)
        self.assertFalse(archive.gates(self.cases, self.attempts, stages))
        self.cases[2]['gate_pass'] = False
        self.assertTrue(archive.gates(self.cases, self.attempts, stages))
        name = archive.PARKING_STAGES[0]
        self.attempts[name+'_01'] = dict(stage=name,attempt=1,status='COMPLETE',reserved_at_unix_ns=13,
                                        completed_at_unix_ns=14)
        with self.assertRaisesRegex(ValueError, 'before prerequisites'):
            archive.gates(self.cases, self.attempts, archive.select(self.attempts))
        self.attempts[name+'_01'].update(reserved_at_unix_ns=23,completed_at_unix_ns=24)
        stages = archive.select(self.attempts)
        self.assertTrue(archive.gates(self.cases,self.attempts,stages))
        for k, value in [('driving_pass', False), ('complete_horizon', False), ('stage_status', 'PARTIAL')]:
            with self.subTest(k):
                with patch.dict(self.cases[1],{k:value}):
                    with self.assertRaisesRegex(ValueError, 'before its gate'):
                        archive.gates(self.cases,self.attempts,stages)

    def test_started_partial_is_counted_without_promotion_or_retry(self):
        name = archive.SPEED_STAGES[1]
        self.attempts[name+'_01'].update(status='PARTIAL',completed_at_unix_ns=None)
        self.attempts[name+'_02'] = dict(self.attempts[name+'_01'],attempt=2)
        with self.assertRaisesRegex(ValueError,'started stage repeated'):
            archive.select(self.attempts)
        self.cases[1].update(stage_status='PARTIAL',counts=None)
        self.cases[2].update(stage_status='NOT_STARTED',counts=None)
        total = archive.totals(self.cases)
        self.assertEqual((total['driving_passes'],total['joint_passes']), (1,1))
        self.assertEqual(total['unknown_partial_seeds'],[archive.SEEDS[1]])
        self.assertEqual((total['confirmed_counts']['physics'],total['counts_upper_bound']['physics']), (6000,12000))

    def test_pre_admission_retry_and_distinct_partial_directory(self):
        stage = archive.SPEED_STAGES[0]
        tag = stage + '_01'
        with TemporaryDirectory() as temp:
            root = Path(temp)
            local, public = root/'local', root/'reports'
            local.mkdir()
            unit = archive.locations(stage,1)[2]
            registration = '3'*64
            with patch.multiple(archive,BASE=local,PUBLIC=public):
                (local/('admission_'+tag+'.json')).write_text(json.dumps(dict(
                    stage=stage,attempt=1,unit=unit,registration_sha256=registration,child_started=False,error='memory')))
                interrupted = local/(stage+'_02')
                interrupted.mkdir()
                (interrupted/'partial.npz').write_bytes(b'recorded incomplete interval')
                entries = archive.collect()
                first = archive.check_attempt(tag,entries[tag],registration)
                self.assertEqual((first['status'],first['error']), ('PRE_ADMISSION_FAILURE','memory'))
                with self.assertRaisesRegex(ValueError,'unbound pre-admission'):
                    archive.check_attempt(tag,entries[tag],'wrong registration')
                other = entries[interrupted.name]
                self.assertEqual(other['status'],'PARTIAL_WITHOUT_GUARD')
                self.assertEqual(other['local_partial_sha256']['partial.npz'],archive.digest(interrupted/'partial.npz'))
                self.assertFalse((public/stage).exists())

    def test_optional_parking_video_requires_the_correct_guard_and_scoped_samples(self):
        base = archive.SOURCE_INDEX.parent/'video'
        guard = json.loads((base/'guard.json').read_text())
        samples = (base/'resources.jsonl').read_text()
        original_scope = guard['scope']
        scope = original_scope.rsplit('/',1)[0] + '/' + parking_video_check.UNIT + '.service'
        guard.update(scope=scope,command=['/usr/bin/python3','-B','-m','tools117.parking_video','--output',
                                         '/tmp/ddt/evidence/compound117_parking/video_01'])
        with TemporaryDirectory() as temp:
            path = Path(temp)
            with patch.multiple(parking_video_check, PUBLIC=path/'public', MOVIE=path/'movie.mp4'):
                self.assertIsNone(parking_video_check.verify())
                parking_video_check.MOVIE.touch()
                with self.assertRaisesRegex(ValueError,'incomplete parking'):
                    parking_video_check.verify()
            (path/'guard.json').write_text(json.dumps(guard))
            (path/'resources.jsonl').write_text(samples.replace(original_scope,scope))
            self.assertGreaterEqual(parking_video_check.check_guard(path)['minimum_effective_bytes'], 2 << 30)
            (path/'resources.jsonl').write_text(samples)
            with self.assertRaisesRegex(ValueError,'do not belong'):
                parking_video_check.check_guard(path)
            guard['command'][-1] = '/tmp/ddt/evidence/compound117_parking/video_02'
            (path/'guard.json').write_text(json.dumps(guard))
            with self.assertRaisesRegex(ValueError,'wrong parking video supervisor'):
                parking_video_check.check_guard(path)


if __name__ == '__main__':
    unittest.main()
