import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools116 import archive


class CompoundArchive(unittest.TestCase):
    def test_pre_admission_failure_and_single_public_movie(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, public, movie = root/'evidence', root/'reports', root/'assets/compound116.mp4'
            local.mkdir()
            first = dict(stage='wiring', attempt=1, unit='d1-compound116-wiring-01', child_started=False,
                         registration_sha256='registered', error='headroom timeout')
            (local/'admission_wiring_01.json').write_text(json.dumps(first))
            worker, guardian = local/'video_01', local/'guard_video_01'
            worker.mkdir()
            guardian.mkdir()
            (worker/'reservation.json').write_text('{}')
            (worker/movie.name).write_bytes(b'only the encoder output is stored in assets')
            (guardian/'report.json').write_text(json.dumps(dict(status='COMPLETE', child_started=True)))
            with patch.multiple(archive, BASE=local, PUBLIC=public, MOVIE=movie):
                entries = archive._collect()
                row = archive._ledger('wiring_01', entries['wiring_01'], 'registered')
                self.assertEqual((row['status'], row['error']), ('PRE_ADMISSION_FAILURE', 'headroom timeout'))
                with self.assertRaises(ValueError):
                    archive._ledger('wiring_01', entries['wiring_01'], 'other registration')
                self.assertEqual(entries['video_01']['files'][movie.name], archive.digest(movie))
                self.assertFalse((public/'video'/movie.name).exists())

    def test_failed_first_pair_closes_video_only_and_cannot_be_retried(self):
        stages = ('wiring', *archive.NATIVE_STAGES)
        outcomes = {}
        for i, stage in enumerate(stages):
            partial = stage == archive.NATIVE_STAGES[0]
            outcomes[stage+'_01'] = dict(stage=stage, attempt=1, status='PARTIAL' if partial else 'COMPLETE',
                                         reserved_at_unix_ns=10*i+1, completed_at_unix_ns=None if partial else 10*i+2)
        labels = archive._stage_order(outcomes)
        self.assertEqual(labels['video']['status'], 'NOT_STARTED')
        self.assertIn(archive.NATIVE_STAGES[0], labels['video']['reason'])
        self.assertEqual([labels[s]['status'] for s in archive.NATIVE_STAGES], ['PARTIAL','COMPLETE','COMPLETE'])
        tag = archive.NATIVE_STAGES[0]
        outcomes[tag+'_02'] = dict(outcomes[tag+'_01'], attempt=2, status='COMPLETE', completed_at_unix_ns=100)
        with self.assertRaises(ValueError):
            archive._stage_order(outcomes)

    def test_missing_counts_and_perception_gates_stay_separate_from_motion(self):
        first, second, third = archive.SEEDS
        good_motion = dict(seed=first, stage_status='COMPLETE', complete_horizon=True, counts=dict(archive.LIMIT),
                           motion=dict(failed=[]), motion_pass=True, lidar=dict(speed_active=True,lift_active=False),
                           lidar_pass=False, gate_pass=False, first_failed_section=None)
        partial = dict(seed=second, stage_status='PARTIAL', counts=None)
        skipped = dict(seed=third, stage_status='NOT_STARTED', counts=None)
        summary = archive.summarize([good_motion, partial, skipped])
        self.assertEqual([summary[k] for k in ('motion_passes','lidar_speed_passes','lidar_lift_passes','combined_passes')],
                         [1, 1, 0, 0])
        self.assertEqual(summary['failures_by_seed'][0]['failed'], ['pre_contact_lidar_lift'])
        self.assertEqual(summary['unknown_partial_native_seeds'], [second])
        self.assertEqual(summary['confirmed_native_counts']['physics'], 6000)
        self.assertEqual(summary['native_counts_upper_bound']['physics'], 12000)

    def test_stream_receipt_pins_first_pair_and_native_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            public, movie = root/'reports', root/'docs/assets/compound116.mp4'
            movie.parent.mkdir(parents=True)
            movie.write_bytes(b'synthetic stream receipt test')
            renderer = root/'tools116/video.py'
            renderer.parent.mkdir()
            renderer.write_bytes(b'synthetic renderer')
            source = public/archive.NATIVE_STAGES[0]/'report.json'
            source.parent.mkdir(parents=True)
            source.write_text('{}')
            seed = archive.SEEDS[0]
            before, after = 'a'*64, 'b'*64
            case = dict(stage_status='COMPLETE', seed=seed, reference_sha256=before, trace_sha256=after,
                        reference_motion={'ticks':600}, motion={'ticks':600})
            stream = dict(nb_read_frames='25', width=1280, height=720, codec_name='h264', pix_fmt='yuv420p',
                          r_frame_rate='25/1', duration='1.0')
            row = dict(status='COMPLETE', filename=movie.name, desktop_filename='demo.mp4', seed=seed,
                       trace_sha256=[before,after], last_ticks=[600,600], renderer_sha256=archive.digest(renderer),
                       native_report_sha256=archive.digest(source), sha256=archive.digest(movie), bytes=movie.stat().st_size,
                       cover_frames=0, physics=0, driver={'renderer':'llvmpipe'}, frames=25, fps=25, duration_s=1.,
                       stream=stream, shots=[dict(tick=600, first_frame=0, frames=25)])
            receipt = public/'video/report.json'
            receipt.parent.mkdir()
            receipt.write_text(json.dumps(dict(result=row)))
            with patch.multiple(archive, ROOT=root, PUBLIC=public, MOVIE=movie):
                self.assertEqual(archive._film({'status':'COMPLETE'},case)['file'], 'docs/assets/compound116.mp4')
                source.write_text('{"changed": true}')
                with self.assertRaises(ValueError):
                    archive._film({'status':'COMPLETE'},case)


if __name__ == '__main__':
    unittest.main()
