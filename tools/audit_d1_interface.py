#!/usr/bin/env python3
"""Zero-physics parity audit against the original C++ FSM and ONNX inferrer.

This compiles unmodified production sources into a bounded offline harness.
It does not import MuJoCo, initialize ROS, publish commands, or train a policy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort
import yaml

WS = Path('/home/lyh/ddt_ros2_ws')
SRC = WS / 'src/ddt_ros2_control'
CTRL = SRC / 'controller/rl_controller'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(argv, *, stdin=None):
    return subprocess.run(['rtk', 'proxy', *map(str, argv)], input=stdin,
                          capture_output=True, text=True, timeout=120)


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def fixtures(default):
    result = []

    def add(op, label, gyro=(0., 0., 0.), quat=(1., 0., 0., 0.),
            command=(0., 0., 0.), q=None, dq=None, action=None):
        result.append({'op': op, 'label': label, 'gyro': list(gyro),
                       'quat_wxyz': list(quat), 'command': list(command),
                       'q': list(default if q is None else q),
                       'dq': list(np.zeros(16) if dq is None else dq),
                       **({'injected_action': list(action)} if action is not None else {})})

    add('enter', 'cold_zero')
    add('forward', 'zero_first_forward')
    q = default + np.linspace(-.2, .2, 16)
    q[[3, 7, 11, 15]] = [100., -200., 300., -400.]
    add('forward', 'tilt_and_wheel_position_mask', gyro=(1., -2., .5),
        quat=quaternion(.2, -.3, .4), command=(.2, -.3, .4),
        q=q, dq=np.linspace(-10., 10., 16))
    add('forward', 'commands_clip', gyro=(.5, .2, -.4),
        quat=quaternion(math.pi/2, 0., 0.), command=(3., -4., 2.))
    add('forward', 'gyro_memory', quat=quaternion(0., math.pi/2, 0.))
    add('forward', 'quaternion_sign_equivalence',
        quat=-np.array(quaternion(0., math.pi/2, 0.)))
    for i in range(12):
        add('forward', f'history_sentinel_{i:02}',
            gyro=(i/20, -i/30, i/40), command=(i/20, -.1, .15),
            quat=quaternion(.01*i, -.02*i, .03*i),
            q=default + i*np.linspace(-.001, .001, 16),
            dq=i*np.linspace(-.03, .03, 16))
    add('forward', 'unclipped_observations', gyro=(1000., -1000., 1500.),
        q=default + 120., dq=np.full(16, 5000.))
    add('injected_action', 'unclipped_action_command_mapping',
        q=default + np.linspace(-9., 9., 16), dq=np.linspace(-200., 200., 16),
        action=np.linspace(-100., 100., 16))
    add('enter', 'warm_reentry_stale_action_probe')
    add('forward', 'warm_first_forward')
    return result


def input_text(cases):
    lines = [str(len(cases))]
    for case in cases:
        values = sum([case[k] for k in ('gyro', 'quat_wxyz', 'command', 'q', 'dq')], [])
        values += case.get('injected_action', [])
        lines.append(case['op'] + ' ' + case['label'] + ' ' +
                     ' '.join(format(float(x), '.17g') for x in values))
    return '\n'.join(lines) + '\n'


class Reference:
    """Independent NumPy contract, checked against unmodified C++ execution."""
    def __init__(self, cfg, native_defaults, session):
        self.cfg = cfg
        self.defaults = native_defaults
        self.session = session
        self.action = np.zeros(16, np.float32)
        self.last_action = np.zeros(16, np.float32)
        self.gyro = np.zeros(3, np.float32)
        self.obs = np.zeros(57, np.float32)
        self.history = np.zeros((10, 57), np.float32)

    def observation(self, case):
        self.gyro = (np.float32(.97)*np.array(case['gyro'], np.float32)
                     + np.float32(.03)*self.gyro)
        # Rotate world down by the inverse quaternion using cross products,
        # independently of the C++ rotation-matrix implementation.
        quat = np.array(case['quat_wxyz'], np.float64)
        v = np.array([0., 0., -1.], np.float64)
        xyz = -quat[1:]
        t = 2*np.cross(xyz, v)
        gravity = (v + quat[0]*t + np.cross(xyz, t)).astype(np.float32)
        q = np.array(case['q'], np.float32)
        q[[3, 7, 11, 15]] = 0.
        error = q - np.array(self.cfg['default_joint_angles'], np.float32)
        command = np.clip(
            np.array(case['command'])*self.defaults['commands_gain'] + self.defaults['commands_comp'],
            self.cfg['min_commands'], self.cfg['max_commands']) * self.defaults['commands_scale']
        return np.concatenate((self.gyro*np.float32(self.cfg['ang_vel_scale']), gravity,
                               command.astype(np.float32), error*np.float32(self.cfg['dof_pos_scale']),
                               np.array(case['dq'], np.float32)*np.float32(self.cfg['dof_vel_scale']),
                               self.last_action)).astype(np.float32)

    def step(self, case):
        inputs, output = [], []
        if case['op'] == 'enter':
            self.gyro.fill(0)
            self.last_action.fill(0)
            self.obs = self.observation(case)
            self.history = np.tile(self.obs, (10, 1))
            # Matches the audited fact: original enter does NOT reset action_vec_.
        elif case['op'] == 'forward':
            self.obs = self.observation(case)
            inputs = [self.obs.copy(), self.history.copy().reshape(-1)]
            self.action = self.session.run(['nn_output'], {
                'nn_input0': self.obs.reshape(1, 57),
                'nn_input1': self.history.reshape(1, 10, 57)})[0].reshape(16)
            output = self.action.copy()
            self.last_action = self.action.copy()
            self.history = np.vstack((self.history[1:], self.obs))
        else:
            self.action = np.array(case['injected_action'], np.float32)
        # C++ computes scaled/position target as float, then stores into double.
        scaled = (self.action.astype(np.float64)*self.cfg['action_scales']).astype(np.float32)
        qd = (scaled + np.array(self.cfg['default_joint_angles'], np.float32)).astype(np.float64)
        qd[[3, 7, 11, 15]] = 0.
        qd_dot = np.zeros(16)
        qd_dot[[3, 7, 11, 15]] = scaled[[3, 7, 11, 15]]
        return {'obs_after': self.obs.copy(), 'history_after': self.history.reshape(-1).copy(),
                'action_after': self.action.copy(), 'onnx_output': np.array(output), 'inputs': inputs,
                'qd': qd, 'qd_dot': qd_dot, 'kp': np.array(self.cfg['joint_kp']),
                'kd': np.array(self.cfg['joint_kd']), 'tau_ff': np.zeros(16)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=WS/'evidence/interface_audit_03/golden_run_01')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f'Refusing to overwrite audit evidence: {out}')
    out.mkdir(parents=True)
    yaml_path = CTRL/'config/d1/controllers.yaml'
    config = yaml.safe_load(yaml_path.read_text())['/**']['d1_rl_controller']['ros__parameters']
    cfg = config['rl_flat_lab']
    assert (cfg['num_obs'], cfg['num_actions'], cfg['history_len'], cfg['control_type']) == (57, 16, 10, 'P_V')
    model = CTRL/cfg['policy_path']
    production_sources = [CTRL/'src/fsm/FSMState_RL.cpp', CTRL/'src/fsm/FSMState.cpp',
                          CTRL/'src/inferrer/onnx_inferrer.cpp']
    provenance_paths = production_sources + [yaml_path, model,
        CTRL/'include/rl_controller/common/RobotParameters.h',
        CTRL/'include/rl_controller/common/Math/orientation_tools.h',
        CTRL/'src/rl_controller_node.cpp',
        SRC/'simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_system.cpp',
        SRC/'simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_ros2_control.cpp',
        SRC/'urdfs/d1_description/mujoco/robot.xml',
        WS/'evidence/d1_expanded.urdf', WS/'tools/interface_harness03.cpp', Path(__file__)]
    before_hashes = {str(p): sha(p) for p in provenance_paths}
    cpp = ['#pragma once',
           'inline void configure_interface03(RLParameters& p, ControlFSMData& d) {']
    loaded_keys = {'output_name', 'num_obs', 'num_actions', 'history_len', 'observations_name',
                   'commands_name', 'max_commands', 'min_commands', 'commands_comp', 'episode_length',
                   'time_interval', 'control_type', 'default_joint_angles', 'joint_kp', 'joint_kd',
                   'action_scales', 'lin_vel_scale', 'ang_vel_scale', 'dof_pos_scale', 'dof_vel_scale',
                   'output_torque_scale', 'reindex', 're_sign'}
    for key in sorted(loaded_keys & cfg.keys()):
        value = cfg[key]
        literal = ('{' + ','.join(json.dumps(x) for x in value) + '}'
                   if isinstance(value, list) else json.dumps(value))
        cpp.append(f'  p.{key} = {literal};')
    cpp += [f'  p.policy_path = {json.dumps(str(model))};',
            '  d.params->wheel_indices = {3,7,11,15};',
            '  d.params->hip_indices = {0,4,8,12};',
            '  d.params->rl_policy_names = {"rl_flat","rl_flat_lab"};', '}']
    header = out/'interface_config03.hpp'
    header.write_text('\n'.join(cpp)+'\n')
    exe = out/'interface_harness03'
    command = ['g++', '-std=c++17', '-O2', '-pthread', '-I'+str(CTRL/'include'),
               '-I'+str(out), *production_sources, WS/'tools/interface_harness03.cpp',
               '-lonnxruntime', '-o', exe]
    compiled = run(command)
    (out/'compile.stdout').write_text(compiled.stdout)
    (out/'compile.stderr').write_text(compiled.stderr)
    dump(out/'compile_argv.json', ['rtk', 'proxy', *map(str, command)])
    if compiled.returncode:
        raise SystemExit('Native harness compilation failed; see compile.stderr')
    cases = fixtures(np.array(cfg['default_joint_angles'], np.float64))
    dump(out/'fixtures.json', cases)
    fixture_input = input_text(cases)
    (out/'fixtures.stdin').write_text(fixture_input)
    native = run([exe], stdin=fixture_input)
    (out/'native.stdout').write_text(native.stdout)
    (out/'native.stderr').write_text(native.stderr)
    if native.returncode:
        raise SystemExit('Native harness failed; see native.stderr')
    records = [json.loads(line) for line in native.stdout.splitlines() if line.startswith('{')]
    metadata = next(x for x in records if x['kind'] == 'metadata')
    native_cases = [x for x in records if x['kind'] == 'case']
    assert len(cases) == len(native_cases)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    shape = {'inputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in session.get_inputs()],
             'outputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in session.get_outputs()]}
    ref = Reference(cfg, metadata, session)
    comparison = []
    bridge_reference = []
    urdf = ET.parse(WS/'evidence/d1_expanded.urdf').getroot()
    urdf_effort = {x.attrib['name']: float(x.find('limit').attrib['effort'])
                   for x in urdf.findall('joint') if x.find('limit') is not None}
    effort = np.array([urdf_effort[name] for name in config['joints']])
    for case, actual in zip(cases, native_cases):
        expected = ref.step(case)
        fields = {}
        for field in expected:
            a = np.array(actual[field], dtype=np.float64) if field != 'inputs' else np.concatenate(
                [np.array(x, dtype=np.float64) for x in actual[field]]) if actual[field] else np.array([])
            e = np.array(expected[field], dtype=np.float64) if field != 'inputs' else np.concatenate(
                expected[field]) if expected[field] else np.array([])
            fields[field] = {'max_absolute_error': float(np.max(np.abs(a-e))) if a.size else 0.,
                             'passed': bool(a.shape == e.shape and np.allclose(a, e, rtol=3e-5, atol=3e-5))}
        comparison.append({'label': case['label'], 'fields': fields,
                           'passed': all(x['passed'] for x in fields.values())})
        raw_tau = (np.array(actual['tau_ff']) + np.array(actual['kp'])*(
            np.array(actual['qd'])-case['q']) + np.array(actual['kd'])*(
            np.array(actual['qd_dot'])-case['dq']))
        bridge_reference.append({'label': case['label'], 'raw_torque': raw_tau.tolist(),
                                 'urdf_clipped_qfrc_applied_reference': np.clip(raw_tau, -effort, effort).tolist(),
                                 'evidence_level': 'derived from real FSM LowlevelCmd plus statically audited bridge formula; bridge write method was not invoked'})
    warm = next(x for x in native_cases if x['label'] == 'warm_reentry_stale_action_probe')
    prewarm = next(x for x in native_cases if x['label'] == 'unclipped_action_command_mapping')
    schema = {'version': 'd1-flat-lab-deployed-interface-03', 'policy': str(model), 'policy_sha256': sha(model),
              'onnx': shape, 'joint_order': config['joints'], 'wheel_indices': config['wheel_indices'],
              'hip_indices': config['hip_indices'], 'resolved_defaults_from_native_struct': metadata,
              'yaml_rl_flat_lab': cfg, 'obs_slices': {
                  'body_gyro_filtered_scaled': [0, 3], 'projected_gravity_body': [3, 6],
                  'command_scaled': [6, 9], 'joint_position_offset_wheels_zero': [9, 25],
                  'joint_velocity_scaled': [25, 41], 'last_raw_policy_action': [41, 57]},
              'gyro_filter': {'new': .97, 'previous_filtered': .03, 'update_scope': 'each update_observations call; enter adds one call before first inference'},
              'quaternion_order': 'wxyz, body-to-world orientation; gravity uses inverse rotation',
              'history_order': 'oldest_to_newest; input contains ten previous observation calls, excludes current inference obs; enter repeats first obs ten times',
              'action_clip': None, 'observation_clip': None, 'command_clip': [-1., 1.],
              'effort_limits_from_expanded_urdf': effort.tolist(),
              'yaml_torque_limits_not_used_by_bridge': config['torque_limit'],
              'force_application': 'qfrc_applied; actuatorfrcrange is not the software bridge clipping mechanism',
              'simulation_dt_seconds': .002, 'requested_control_hz': 400,
              'effective_control_hz_at_original_dt': 250, 'write_hz_at_original_dt': 500,
              'inference_target_wall_hz': 50, 'synchronous_training_decimation_proposal': 10,
              'timing_note': '50 Hz wall-clock inference is asynchronous; ten .002-second simulation steps is a training convention, exact only near real-time execution. Bridge write uses state held from most recent 250 Hz read.'}
    dump(out/'schema.json', schema)
    dump(out/'bridge_torque_reference.json', bridge_reference)
    report = {'schema': 'd1-offline-interface-golden-audit-v1',
              'scope': '22 fixed fixtures through unmodified C++ FSMState_RL enter/update_observations/update_forward/run and original ONNXInferrer; independent NumPy/ONNX Runtime comparison',
              'physics_steps': 0, 'training_steps': 0, 'ros_nodes_started': 0,
              'source_hashes_before': before_hashes,
              'source_hashes_unchanged': before_hashes == {str(p): sha(p) for p in provenance_paths},
              'cpp_onnxruntime': metadata['onnxruntime'], 'python_onnxruntime': ort.__version__,
              'fixtures': len(cases), 'actual_native_onnx_calls': sum(x['op']=='forward' for x in cases),
              'comparisons': comparison,
              'warm_reentry_preserves_stale_action': warm['action_after'] == prewarm['action_after'],
              'warm_reentry_history_last_actions_zero': bool(np.all(np.array(warm['history_after']).reshape(10,57)[:,41:57] == 0)),
              'all_parity_checks_passed': all(x['passed'] for x in comparison),
              'limits': ['No ROS transport, asynchronous scheduling, or physical dynamics validation.',
                         'Bridge force clipping is a static source/URDF audit and derived reference, not a runtime bridge test.',
                         'The ZIP G2-arm training configuration with 78 observations/23 actions is not established provenance for this 57+570/16-action policy.'],
              'artifacts': {'schema': str(out/'schema.json'), 'native_stdout': str(out/'native.stdout'),
                            'fixed_fixtures': str(out/'fixtures.json')}}
    dump(out/'report.json', report)
    print(json.dumps({k: report[k] for k in ['physics_steps','fixtures','actual_native_onnx_calls',
                                            'all_parity_checks_passed','source_hashes_unchanged',
                                            'warm_reentry_preserves_stale_action']}, indent=2))
    print('Report:', out/'report.json')
    if not report['all_parity_checks_passed'] or not report['source_hashes_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
