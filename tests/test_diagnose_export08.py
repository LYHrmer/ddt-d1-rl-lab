"""Tests for the pure part of tools08/diagnose_export08.py. No model, no physics."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import diagnose_export08 as diag  # noqa: E402


def test_summary_counts_elements_above_the_limit_and_scales_by_the_output():
    out = diag.summarise([0., 2e-6, -3e-6, 5e-7], [1., -4., 2., .5])
    assert out['max_abs_difference'] == pytest.approx(3e-6) and out['elements_above_limit'] == 2
    assert out['max_abs_unclipped_action'] == 4.
    assert out['float32_eps_times_peak'] == pytest.approx(np.finfo(np.float32).eps*4.)
    assert out['difference_over_eps_times_peak'] == pytest.approx(3e-6/(np.finfo(np.float32).eps*4.))


def test_the_limits_are_the_frozen_exporters():
    source = (ROOT/'tools/export_d1_ppo.py').read_text()
    assert 'if max_torch_ort > 1e-5 or max_sb3_torch > 1e-6:' in source
    assert (diag.TORCH_ORT_LIMIT, diag.SB3_TORCH_LIMIT) == (1e-5, 1e-6)


def test_a_zero_output_does_not_divide_by_zero():
    assert diag.summarise([0.], [0.])['difference_over_eps_times_peak'] is None
