"""Standard-library regressions for strict archived diagnostic input seams."""
import io
import math
import struct
import unittest
import zipfile

from tools76.readback76 import f64_array, ordered, summary, validate_report


def npy(values, *, dtype='<f8', shape=(1, 2), fortran=False):
    header = repr({'descr': dtype, 'fortran_order': fortran, 'shape': shape}).encode() + b'\n'
    return b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + struct.pack('<' + 'd' * len(values), *values)


class ReadbackTest(unittest.TestCase):
    def archive(self, payload, name='value.npy'):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as handle:
            handle.writestr(name, payload)
        stream.seek(0)
        return zipfile.ZipFile(stream)

    def test_exact_raw_values_and_negative_zero_are_kept(self):
        with self.archive(npy([1e-20, -0.])) as handle:
            values = f64_array(handle, 'value', (1, 2))
        self.assertEqual(values[0], 1e-20)
        self.assertEqual(values[1].hex(), '-0x0.0p+0')

    def test_wrong_dtype_shape_order_or_nonfinite_is_rejected(self):
        for payload in (npy([1., 2.], dtype='<f4'), npy([1., 2.], shape=(2,)),
                        npy([1., 2.], fortran=True), npy([1., math.inf]), npy([1.])):
            with self.subTest(payload=payload), self.archive(payload) as handle, self.assertRaises(ValueError):
                f64_array(handle, 'value', (1, 2))

    def test_missing_member_and_large_member_are_rejected(self):
        with self.archive(npy([1., 2.]), name='other.npy') as handle, self.assertRaises(ValueError):
            f64_array(handle, 'value', (1, 2))
        with self.archive(b' ' * 65537) as handle, self.assertRaises(ValueError):
            f64_array(handle, 'value', (1, 2))

    def test_point_order_and_missing_evidence_fail_closed(self):
        grid = [{'seed': seed, 'tick': tick} for seed in (72301, 72302, 72303) for tick in (110, 260, 410)]
        ordered(grid, 'fresh')
        for wrong in (grid[:-1], list(reversed(grid))):
            with self.assertRaises(ValueError):
                ordered(wrong, 'fresh')
        for status in ('INCOMPLETE', 'PROVISIONAL'):
            with self.assertRaises(ValueError):
                validate_report({'schema': 'd1-yaw-weight72-result-v1', 'status': status,
                                 'verification_after': 'COMPLETE', 'comparison': {'status': status}})

    def test_overlapping_reasons_do_not_rewrite_unknown_or_false_original_gate(self):
        rows = [{'model_seed': 64001, 'arm': 'control', 'kind': 'body_vx',
                 'diagnosis': {'reasons': ['LOW', 'UNKNOWN'], 'original_valid_match': None}},
                {'model_seed': 64001, 'arm': 'control', 'kind': 'body_vx',
                 'diagnosis': {'reasons': ['LOW'], 'original_valid_match': False}}]
        result = summary(rows)[0]
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['reasons'], {'LOW': 2, 'UNKNOWN': 1})
        self.assertEqual(result['original_match_unknown'], 1)
        self.assertEqual(result['original_match_false'], 1)
        self.assertEqual(result['original_match_true'], 0)


if __name__ == '__main__':
    unittest.main()
