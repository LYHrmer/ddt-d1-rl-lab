"""Compile and use the C++ implementation of qualified33 reference shaping."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from tools19 import cpp_actor19 as transport
from tools33.run_reference_shape33 import Runtime as PythonReference

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runtime34/actor_server34.cpp'
require, sha, atomic_json = transport.require, transport.sha, transport.atomic_json


def build_actor(output):
    """Only compilation; model sessions and physics are still zero."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    compiler = Path(shutil.which('g++')).resolve(strict=True)
    paths = [compiler, SOURCE, Path(__file__).resolve(), transport.ORT_LIBRARY, *transport.HEADERS]
    inputs = {str(p): sha(p) for p in paths}
    binary = output/'actor_server34'
    command = [str(compiler), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Wpedantic', '-ffp-contract=off',
               '-DD1_ACTOR34_SOURCE_SHA256="'+sha(SOURCE)+'"', str(SOURCE), '-I/usr/include',
               '-L/usr/lib', '-Wl,-rpath,/usr/lib', '-lonnxruntime', '-lcrypto', '-ldl', '-o', str(binary)]
    with (output/'stdout.log').open('xb') as out, (output/'stderr.log').open('xb') as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=120, check=False)
    require(result.returncode == 0, 'C++34 compilation failed')
    require(all(sha(p) == digest for p, digest in inputs.items()), '34 build input changed')
    proof = {'status': 'PASS', 'binary': str(binary), 'binary_sha256': sha(binary),
             'inputs_sha256': inputs, 'command': command, 'inference': 0, 'physics': 0}
    atomic_json(output/'build.json', proof)
    return proof


class CppReferenceActor(transport.CppActor19):
    """Version19 transport with two explicit, audited34 response diagnostics."""

    def _receive(self, timeout):
        value = super()._receive(timeout)
        if value.get('format') == 'd1-cpp-actor19-response-v1':
            import numpy as np
            require('effective_yaw_commands' in value and 'effective_input_sha256' in value,
                    '34 C++ preprocessing evidence missing')
            self.effective_yaw = np.asarray(value.pop('effective_yaw_commands'), dtype=np.float32)
            self.effective_sha = value.pop('effective_input_sha256')
            require(self.effective_yaw.shape == (11,) and np.isfinite(self.effective_yaw).all(),
                    '34 C++ effective command shape differs')
        return value

    def start(self):
        super().start()
        require(self.stats['ready'].get('reference_shaping') ==
                {'gain': 1.5, 'tau_s': .1, 'dt_s': .02, 'reset': 'new_process'}, '34 C++ state contract differs')
        return self


class Runtime(PythonReference):
    def __init__(self, original, folder, build):
        super().__init__(original)
        self.folder, self.build = Path(folder), build
        self.actor, self.extra = None, []
        self.cpp_attempts = self.cpp_returns = 0
        self.cpp_cleanup = True
        self.backend.update(kind='cpp34_reference_state_and_ort1.10_physical_driver',
                            python_ort_role='same_input_oracle_only', actor_calls_per_tick=2,
                            cpp_state_reset='new_process_per_episode')

    def make_env(self, name, sequence, delay, seed):
        require(self.actor is None and not self.extra, '34 preceding actor not closed')
        model = ROOT/'evidence/low_memory_delay25/execution_01/random/export/policy.onnx'
        self.actor = CppReferenceActor(self.build['binary'], model,
            output=self.folder/f'{name}_delay{delay}_seed{seed}_cpp',
            expected_binary_sha256=self.build['binary_sha256'], expected_model_sha256=sha(model),
            expected_source_sha256=sha(SOURCE), source=SOURCE, max_requests=600)
        self.actor.start()
        return super().make_env(name, sequence, delay, seed)

    def predict(self, obs):
        import numpy as np
        from tools33.reference_shape33 import YAW_COLUMNS
        oracle = super().predict(obs)
        self.cpp_attempts += 1
        actual = self.actor.predict(obs)
        self.cpp_returns += 1
        effective = obs.copy()
        effective[YAW_COLUMNS] = self.rows[-1][1]
        require(np.array_equal(self.actor.effective_yaw, effective[YAW_COLUMNS]) and
                self.actor.effective_sha == hashlib.sha256(effective.tobytes()).hexdigest(),
                'C++34 full effective input differs from independent Python33')
        error = float(np.max(abs(actual.astype(float)-oracle.astype(float))))
        require(error <= 1e-5, 'C++34 same-input action parity failed')
        self.rows[-1] = (*self.rows[-1][:2], actual.copy())
        self.extra.append((obs.copy(), oracle.copy(), actual.copy(), error))
        return actual

    def close_actor(self):
        if self.actor is None:
            return None
        report = self.actor.close()
        self.cpp_cleanup &= report['cleanup_complete']
        self.actor = None
        return report

    def verify(self, output, item, helpers):
        import numpy as np
        report = self.close_actor()
        require(report['status'] == 'CLOSED' and report['cleanup_complete'] and
                report['requests_attempted'] == report['requests_completed'] ==
                report['inference_attempts'] == report['inference_returns'] == item['policy_transitions'],
                'C++34 actual requests/cleanup mismatch')
        super().verify(output, item, helpers)
        path = output/(Path(item['trace_file']).stem+'_cpp_oracle.npz')
        arrays = {name: np.asarray([row[i] for row in self.extra])
                  for i, name in enumerate(('raw_actor627', 'python_oracle16', 'cpp_issued16', 'max_error'))}
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            require(all(np.array_equal(saved[k], arrays[k]) for k in arrays), '34 sidecar readback differs')
        item['cpp_reference34'] = {'sidecar': path.name, 'sha256': sha(path),
            'full_effective_inputs_bitwise_equal': True, 'max_action_error': float(arrays['max_error'].max()),
            'cpp_calls': report['inference_returns'], 'python_oracle_calls': len(self.extra),
            'transport': report, 'physical_action_driver': 'C++34'}
        self.extra.clear()
