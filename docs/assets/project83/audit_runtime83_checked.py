#!/usr/bin/env python3
"""Independent archived checked-runtime83 audit; stdlib only, no subprocess or inference.

Missing, failed, or truncated formal evidence produces INCOMPLETE, including a
retained successfully read prefix. Expected fault scenarios may pass assertions
while their engine and consumed-output status must remain INCOMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
from pathlib import Path
import signal
import struct
import sys
import time

ROOT = Path('/home/lyh/ddt_ros2_ws')
WORK = Path('/home/lyh/project83_work_20261104')
PARENT_PLAN = ROOT/'docs/experiments83/runtime83_preregistered.json'
PLAN = ROOT/'docs/experiments83/runtime83_checked_preregistered.json'
EXECUTION = ROOT/'evidence/runtime83/execution_01'
CHECKED_EXECUTION = ROOT/'evidence/runtime83/execution_checked_01'
CHECKED_BUILD = WORK/'build_runtime83_checked_01'
DEADLINE = 20_000_000
IDENTITIES = ('binary_sha256', 'model_sha256', 'source_sha256', 'ort_library_sha256')
MODES = ('nominal', 'late_receive', 'late_response', 'stale_command')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite JSON literal: ' + value)


def parse(text):
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def pack32(values):
    return struct.pack('<' + 'f'*len(values), *values)


def float32(value):
    return struct.unpack('<f', pack32([value]))[0]


class Audit:
    def __init__(self):
        self.inputs = {}
        self.checks = 0
        self.failures = []
        self.stages = {}

    def check(self, condition, label, **detail):
        self.checks += 1
        if not condition:
            self.failures.append({'check': label, **detail})
        return bool(condition)

    def read(self, path):
        path = Path(path)
        if path.stat().st_size > 32 << 20:
            raise ValueError('input exceeds read cap: ' + str(path))
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if str(path) in self.inputs:
            self.check(self.inputs[str(path)] == actual, 'input_changed_between_reads', path=str(path))
        else:
            self.inputs[str(path)] = actual
        return data

    def json(self, path):
        return parse(self.read(path))

    def jsonl(self, path):
        data = self.read(path)
        rows = []
        for index, line in enumerate(data.splitlines(keepends=True)):
            try:
                if not line.endswith(b'\n'):
                    raise ValueError('unflushed final JSONL row')
                row = parse(line)
                if not isinstance(row, dict):
                    raise ValueError('non-object JSONL row')
                rows.append(row)
            except (ValueError, UnicodeError) as exc:
                self.check(False, 'jsonl_complete', path=str(path), row=index, error=str(exc))
                break
        return rows

    def verify_map(self, mapping, label):
        for name, expected in mapping.items():
            try:
                actual = self.inputs.get(name)
                if actual is None:
                    actual = sha(name)
                    self.inputs[name] = actual
                self.check(actual == expected, label, path=name, expected=expected, actual=actual)
            except OSError as exc:
                self.check(False, label, path=name, error=str(exc))

    def vector(self, values, size, label, bound=None):
        valid = (isinstance(values, list) and len(values) == size and
                 all(type(v) in (int, float) and math.isfinite(v) for v in values))
        if valid and bound is not None:
            valid = all(abs(v) <= bound for v in values)
        self.check(valid, label)
        if not valid:
            raise ValueError('invalid vector: ' + label)

    def zero(self, row, label):
        self.vector(row['effort16'], 16, label + '.effort16')
        self.check(row['authorized'] is False and row['has_targets'] is False and
                   row['status'] == 'INCOMPLETE' and all(v == 0 for v in row['effort16']), label)

    def finite_tree(self, value, label):
        if isinstance(value, float):
            self.check(math.isfinite(value), label)
        elif isinstance(value, dict):
            for key, child in value.items():
                self.finite_tree(child, label + '.' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.finite_tree(child, label + '[' + str(index) + ']')

    def response(self, row, sequence, label):
        response = row['response']
        self.check(response['format'] == 'd1-cpp-actor19-response-v1' and
                   response['status'] == 'OK' and response['seq'] == sequence and
                   response['inference_attempts'] == sequence + 1 and
                   response['inference_returns'] == sequence + 1, label + '.response_counters')
        self.vector(row['raw_actor627'], 627, label + '.raw_actor627')
        self.vector(response['actions'], 16, label + '.actions', 1.)
        self.vector(response['effective_yaw_commands'], 11, label + '.effective_yaw', .25)
        effective = list(row['raw_actor627'])
        for frame, yaw in enumerate(response['effective_yaw_commands']):
            effective[frame*57 + 8] = yaw
        actual_hash = hashlib.sha256(pack32(effective)).hexdigest()
        self.check(actual_hash == response['effective_input_sha256'], label + '.effective_input_hash',
                   recomputed=actual_hash, recorded=response['effective_input_sha256'])

    def shaper(self, row, states, label):
        # Replay only frozen34's explicit scalar shaper, never the ONNX actor.
        raw = [float32(row['raw_actor627'][frame*57 + 8]) for frame in range(11)]
        targets = [max(-.25, min(.25, float32(value*1.5))) for value in raw]
        generation = row['generation']
        if generation not in states:
            self.check(all(value == raw[0] for value in raw), label + '.shaper_constant_initial_history')
            states[generation] = {'state':targets[0], 'history':targets[1:], 'raw_history':raw[1:]}
        saved = states[generation]
        self.check(raw[1:] == saved['raw_history'], label + '.shaper_raw_history_matches_prior_inputs')
        alpha = math.exp(-.2)
        saved['state'] = alpha*saved['state'] + (1. - alpha)*targets[0]
        current = float32(saved['state'])
        expected_yaw = [current, *saved['history']]
        self.check(pack32(expected_yaw) == pack32(row['response']['effective_yaw_commands']),
                   label + '.shaper_f32_replay_exact')
        effective = list(row['raw_actor627'])
        for frame, value in enumerate(expected_yaw): effective[frame*57 + 8] = value
        self.check(hashlib.sha256(pack32(effective)).hexdigest() == row['response']['effective_input_sha256'],
                   label + '.independently_shaped_f32_input_sha256')
        saved['history'] = saved['history'][1:] + [current]
        saved['raw_history'] = saved['raw_history'][1:] + [raw[0]]

    def stage(self, name, manifest, *, native=False):
        path = EXECUTION/'native' if native else CHECKED_EXECUTION/('ros_' + name)
        summary = {'status': 'INCOMPLETE', 'path': str(path), 'observed_rows': None,
                   'requests_attempted': None, 'requests_completed': None,
                   'session_inference_attempts': None, 'session_inference_returns': None}
        self.stages['native' if native else name] = summary
        report = self.json(path/'report.json')
        summary['source_report_status'] = report.get('status')
        self.check(report['status'] == 'COMPLETE' and report.get('verification_after') == 'COMPLETE' and
                   report.get('error') is None, name + '.formal_complete')
        copied = self.json(path/'manifest.json')
        self.check(copied == manifest, name + '.copied_manifest')
        self.check(report.get('manifest_sha256') == self.inputs[str(path/'manifest.json')],
                   name + '.manifest_sha256')
        engine = self.json(path/'engine/report.json')
        self.check(engine == report['engine'], name + '.engine_report_parity')
        self.check(engine['identities'] == {k: manifest[k] for k in IDENTITIES}, name + '.engine_identities')
        self.check(engine['request_cap'] == manifest['max_requests'], name + '.request_cap')
        self.check(engine['synthetic'] is True and engine['hardware_connected'] is False and
                   engine['physics_steps'] == 0 and engine['parameter_updates'] == 0,
                   name + '.zero_hardware_physics_updates')
        self.check(engine['cleanup_complete'] is True and engine['verification_after'] == 'COMPLETE',
                   name + '.engine_cleanup_verification')
        ticks = self.jsonl(path/'engine/ticks.jsonl')
        attempts = self.jsonl(path/'engine/attempts.jsonl')
        summary.update(observed_rows=len(ticks), requests_attempted=engine['requests_attempted'],
                       requests_completed=engine['requests_completed'], engine_status=engine['status'],
                       fault=engine['fault'], attempt_ledger_rows=len(attempts))
        expected_ticks = 64 if native else (24 if name == 'nominal' else 1)
        expected_attempts = 64 if native else (24 if name == 'nominal' else (0 if name == 'stale_command' else 1))
        expected_completed = expected_attempts - (1 if name == 'late_response' else 0)
        healthy = native or name == 'nominal'
        self.check(len(ticks) == expected_ticks, name + '.tick_count', actual=len(ticks), expected=expected_ticks)
        self.check(len(attempts) == engine['requests_attempted'] == expected_attempts,
                   name + '.attempt_count', ledger=len(attempts), actual=engine['requests_attempted'])
        self.check(engine['requests_completed'] == expected_completed, name + '.completed_count')
        expected_fault = {'late_receive': 'CONSUMER_DEADLINE', 'late_response': 'RPC_DEADLINE',
                          'stale_command': 'COMMAND_STALE'}.get(name, '')
        self.check(engine['fault'] == expected_fault and
                   engine['status'] == ('COMPLETE' if healthy else 'INCOMPLETE'), name + '.runtime_status')
        self.check(engine['sessions_started'] == len(engine['sessions']) == (2 if native else 1),
                   name + '.session_count')
        session_attempts, session_returns = [], []
        for index, session in enumerate(engine['sessions']):
            prefix = name + '.session_' + str(index)
            on_disk = self.json(path/'engine'/('session_' + str(index))/'report.json')
            self.check(on_disk == session, prefix + '.report_parity')
            ready = session['ready']
            self.check(all(ready[k] == manifest[k] for k in IDENTITIES), prefix + '.ready_identities')
            self.check(ready['format'] == 'd1-cpp-actor19-ready-v1' and ready['status'] == 'READY' and
                       ready['onnxruntime_version'] == '1.10.0' and ready['cpu_threads'] == 1 and
                       ready['max_requests'] == manifest['max_requests'] - (32*index if native else 0),
                       prefix + '.ready_contract')
            self.check(ready['signature'] == {'inputs': {'obs': [1,57], 'history': [1,10,57]},
                                              'outputs': {'actions': [1,16]}} and
                       ready['reference_shaping'] == {'dt_s': .02, 'gain': 1.5, 'tau_s': .1, 'reset': 'new_process'},
                       prefix + '.shape_and_shaper')
            self.check(all(ready[k] == 0 for k in ('inference_attempts', 'inference_returns',
                                                  'robot_models_constructed', 'physics_steps_executed')),
                       prefix + '.ready_zero_counts')
            wanted_attempts = 32 if native else expected_attempts
            wanted_completed = 32 if native else expected_completed
            self.check(session['requests_attempted'] == wanted_attempts and
                       session['requests_completed'] == wanted_completed and session['cleanup_complete'] is True,
                       prefix + '.counts_cleanup')
            actual_attempts, actual_returns = session['inference_attempts'], session['inference_returns']
            session_attempts.append(actual_attempts); session_returns.append(actual_returns)
            if name == 'late_response':
                self.check(actual_attempts is None and actual_returns is None,
                           prefix + '.unreturned_inference_unknown_preserved')
            else:
                self.check(actual_attempts == wanted_completed and actual_returns == wanted_completed,
                           prefix + '.known_inference_counts')
            self.check(session['status'] == ('CLOSED' if healthy else 'INCOMPLETE'), prefix + '.terminal_status')
            if healthy:
                end = session['eof_receipt']
                self.check(end['format'] == 'd1-cpp-actor19-closed-v1' and end['status'] == 'CLOSED' and
                           all(end[k] == wanted_completed for k in ('requests', 'inference_attempts', 'inference_returns')) and
                           end['robot_models_constructed'] == end['physics_steps_executed'] == 0 and
                           session['wait_status'] == 0, prefix + '.closed_receipt')
        summary['session_inference_attempts'] = session_attempts
        summary['session_inference_returns'] = session_returns
        attempt_ticks = [r for r in ticks if 'raw_actor627' in r]
        self.check(len(attempt_ticks) == len(attempts), name + '.attempt_tick_alignment')
        for index, (row, intent) in enumerate(zip(attempt_ticks, attempts)):
            self.check(intent['phase'] == 'BEFORE_RPC' and intent['request_attempt'] == index + 1 and
                       intent['actual_inference_attempts'] is None and intent['actual_inference_returns'] is None,
                       name + '.attempt_intent_' + str(index))
            self.check(all(intent[k] == row[k] for k in ('capture_ns','begin_ns','generation','token_sequence','raw_actor627')),
                       name + '.intent_tick_parity_' + str(index))
        ages = {'begin': [], 'accepted': [], 'end': []}
        shaper_states = {}
        for index, row in enumerate(ticks):
            prefix = name + '.tick_' + str(index)
            generation = index//32 + 1 if native else 1
            sequence = index % 32 if native else index
            self.check(row['schema'] == 'd1-adapter83-tick-v1' and row['synthetic'] is True and
                       row['generation'] == generation, prefix + '.identity')
            capture, begin, end = row['capture_ns'], row['begin_ns'], row['end_ns']
            self.check(0 <= capture <= begin <= end and begin - capture <= DEADLINE, prefix + '.capture_age')
            self.check(row['command_age_ns'] == begin - row['command_received_ns'], prefix + '.command_age_arithmetic')
            self.check((row['command_age_ns'] > 500_000_000) if name == 'stale_command' else
                       0 <= row['command_age_ns'] <= 500_000_000, prefix + '.command_age_limit')
            ages['begin'].append(begin - capture); ages['end'].append(end - capture)
            if 'raw_actor627' in row:
                self.check(row['token_sequence'] == sequence, prefix + '.token_sequence')
            if 'response' in row:
                self.response(row, sequence, prefix)
                self.shaper(row, shaper_states, prefix)
                accepted = row['accepted_ns']; ages['accepted'].append(accepted - capture)
                self.check(begin <= accepted <= end and accepted - capture <= DEADLINE, prefix + '.accepted_deadline')
            if healthy or name == 'late_receive':
                self.check(row['status'] == 'COMPLETE' and row['authorized'] is True and
                           row['has_targets'] is True and end - capture <= DEADLINE, prefix + '.authorized_tick_deadline')
            else:
                self.zero(row, prefix + '.fail_closed_tick')
            self.vector(row['effort16'], 16, prefix + '.effort')
            self.check(all(abs(v) <= (12 if i % 4 == 3 else 80) for i,v in enumerate(row['effort16'])),
                       prefix + '.torque_limits')
        summary['maximum_capture_age_ns'] = {k: max(v) if v else None for k,v in ages.items()}
        summary['scalar_shaper_replay_generations'] = sorted(shaper_states)
        if native:
            self.check(report['sessions'] == 2 and report['ticks_per_session'] == 32 and
                       report['fresh_session_value_and_effective_hash_replay'] is True, 'native.declared_replay')
            replay = []
            for index in range(min(32, max(0, len(ticks) - 32))):
                left, right = ticks[index], ticks[index+32]
                fields = {
                    'raw627_f32_bytes': pack32(left['raw_actor627']) == pack32(right['raw_actor627']),
                    'actions16_f32_bytes': pack32(left['response']['actions']) == pack32(right['response']['actions']),
                    'effective_yaw11_f32_bytes': pack32(left['response']['effective_yaw_commands']) == pack32(right['response']['effective_yaw_commands']),
                    'effective_input_sha256': left['response']['effective_input_sha256'] == right['response']['effective_input_sha256'],
                }
                for field, okay in fields.items(): self.check(okay, 'native.replay_' + str(index) + '.' + field)
                replay.append({'tick': index, **fields})
            summary['fresh_session_paired_rows'] = replay
            self.check(len(replay) == 32, 'native.all_32_replay_pairs')
        else:
            self.dds(name, path, report, ticks, summary)
        summary['status'] = 'READ_COMPLETE'

    def dds(self, mode, path, report, ticks, summary):
        packets = self.jsonl(path/'received_packets.jsonl')
        outputs = self.jsonl(path/'observed_efforts.jsonl')
        publications = self.jsonl(path/'publication_events.jsonl')
        size = 24 if mode == 'nominal' else 1
        namespace = '/adapter83/project83_' + mode
        self.check(len(packets) == len(outputs) == len(publications) == len(ticks) == size,
                   mode + '.dds_ledger_counts')
        self.check(all(report[k] == size for k in ('planned_packets','packets_sent','packets_received',
                                                  'efforts_observed','publish_calls_attempted','publish_calls_returned')),
                   mode + '.dds_report_counts')
        self.check(report['domain'] == '183' and report['namespace'] == namespace and report['mode'] == mode and
                   report['scenario_assertions_passed'] is True and
                   report['runtime_accepted'] is (mode == 'nominal') and
                   report['runtime_status'] == ('COMPLETE' if mode == 'nominal' else 'INCOMPLETE') and
                   report['actual_sensor_synchronization_validated'] is False, mode + '.dds_scope_status')
        consumed = wire = expired = revoked = late_published = 0
        ages = {'publish_entry': [], 'publish_return': [], 'receive': [], 'consume': []}
        for index, (packet, output, publication, tick) in enumerate(zip(packets, outputs, publications, ticks)):
            prefix = mode + '.dds_' + str(index)
            self.check(packet['seq'] == output['seq'] == publication['seq'] == index and
                       packet['run_namespace'] == output['run_namespace'] == namespace and
                       packet['generation'] == 1 and packet['synthetic'] is output['synthetic'] is True,
                       prefix + '.sequence_provenance')
            capture = packet['capture_ns']
            self.check(capture == output['capture_ns'] == publication['capture_ns'] == tick['capture_ns'], prefix + '.same_capture')
            self.vector(packet['sensor39'], 39, prefix + '.sensor39')
            self.check(packet['sensor39'] == tick['gyro3'] + tick['quat_wxyz4'] + tick['q16'] + tick['dq16'],
                       prefix + '.sensor_tick_parity')
            expected_command = None if mode == 'stale_command' else [.1,0.,.2 if index < size//2 else -.2]
            self.check(packet['command'] == expected_command, prefix + '.fixed_command')
            entry, returned, received = publication['publish_call_begin_ns'], publication['publish_return_ns'], output['receive_ns']
            prepared = publication['publish_preparation_ns']
            self.check(capture <= prepared <= entry <= returned <= received and
                       prepared == output['publish_preparation_ns'], prefix + '.publication_timing_order')
            self.check(publication['published_message_can_be_recalled'] is False and
                       publication['sent_authorized'] is output['wire_authorized'], prefix + '.wire_authorization')
            is_late = publication['sent_authorized'] and returned - capture > DEADLINE
            self.check(publication['authorized_return_after_deadline'] is is_late and
                       publication['status'] == ('INCOMPLETE' if is_late else 'COMPLETE'), prefix + '.publish_return_receipt')
            late_published += int(is_late)
            for key, stamp in (('publish_entry',entry),('publish_return',returned),('receive',received)):
                ages[key].append(stamp - capture)
            if output['wire_authorized']:
                wire += 1
                self.check(entry - capture <= DEADLINE and returned - capture <= DEADLINE,
                           prefix + '.authorized_publish_deadlines')
                consume_time = output['consume_checked_ns']; ages['consume'].append(consume_time - capture)
                self.check(consume_time >= received, prefix + '.consume_timestamp_order')
            self.check(output['effective_authorized'] is output['authorized'] and
                       output['consumed_authorized_effort'] is output['effective_authorized'], prefix + '.effective_consumption')
            if mode == 'nominal':
                self.check(output['wire_authorized'] is True and output['effective_authorized'] is True and
                           output['status'] == 'COMPLETE' and output['wire_status'] == 'COMPLETE' and
                           output['consume_checked_ns'] - capture <= DEADLINE and received - capture <= DEADLINE and
                           output['effort16'] == output['wire_effort16'], prefix + '.nominal_deadlines_and_consumption')
            else:
                self.zero(output, prefix + '.consumer_zero_effort')
                self.check(output['consumed_authorized_effort'] is False, prefix + '.no_fault_consumption')
                if mode == 'late_receive':
                    injection = output['fault_injection']
                    self.check(output['wire_authorized'] is True and output['wire_status'] == 'COMPLETE' and
                               output['wire_has_targets'] is True and output['reason'] == 'CONSUMER_DEADLINE' and
                               injection['kind'] == 'consumer_delay_after_received_authorized_message' and
                               injection['requested_delay_ns'] == 21_000_000 and
                               received <= injection['started_ns'] <= injection['ended_ns'] <= output['consume_checked_ns'] and
                               injection['started_ns'] - capture <= DEADLINE and
                               injection['ended_ns'] - injection['started_ns'] >= 21_000_000 and
                               output['consume_checked_ns'] - capture > DEADLINE, prefix + '.late_receive_rejected_before_consume')
                else:
                    self.check(output['wire_authorized'] is False and output['wire_has_targets'] is False and
                               all(v == 0 for v in output['wire_effort16']), prefix + '.wire_zero_effort')
            consumed += int(output['consumed_authorized_effort'])
            expired += int(output['wire_authorized'] and output['reason'] == 'CONSUMER_DEADLINE')
            revoked += int(output['wire_authorized'] and output['reason'] == 'CONSUMER_AUTHORITY_REVOKED')
        recomputed = {'authorized_efforts_consumed': consumed, 'wire_authorized_messages_observed': wire,
                      'expired_authorized_messages_rejected': expired,
                      'revoked_authorized_messages_rejected': revoked,
                      'authorized_publish_returns_after_deadline': late_published}
        for key, value in recomputed.items(): self.check(report[key] == value, mode + '.recount_' + key)
        self.check(consumed == (24 if mode == 'nominal' else 0) and
                   expired == (1 if mode == 'late_receive' else 0) and revoked == late_published == 0,
                   mode + '.expected_authorization_counts')
        summary['dds_recount'] = recomputed
        summary['maximum_dds_capture_age_ns'] = {k:max(v) if v else None for k,v in ages.items()}

    def supervisor(self, folder, expected_status='COMPLETE'):
        path = WORK/folder/'report.json'
        report = self.json(path)
        self.check(report['schema'] == 'pressure-supervisor83-v1' and
                   report['status'] == expected_status, folder + '.supervisor_status')
        self.check(report['child_started'] is True and report['child_reaped'] is True and
                   report['owned_cgroup_children_empty'] is True and report['remaining_owned_pids'] == [],
                   folder + '.owned_cleanup_complete')
        self.check(report['retry_permitted'] is False and report['new_physics_steps_in_supervisor'] == 0,
                   folder + '.supervisor_scope')
        wheel = Path('/home/lyh/wheel-legged-control-lab/scripts')
        self.verify_map({str(wheel/'run_pressure83.py'): report['supervisor_sha256'],
                         str(wheel/'resource_pressure83.py'): report['pressure_policy_sha256']},
                        folder + '.supervisor_source_sha')
        if expected_status == 'COMPLETE':
            self.check(report['returncode'] == report['returncode_after_cleanup'] == 0 and
                       report['error'] is None, folder + '.successful_child_exit')
        else:
            self.check(type(report['returncode']) is int and report['returncode'] != 0 and
                       report['returncode_after_cleanup'] == report['returncode'] and report['error'] is not None,
                       folder + '.recorded_failed_child_exit')
        saved = {key:report[key] for key in ('status','command','returncode','returncode_after_cleanup',
                                             'error','child_reaped','owned_cgroup_children_empty',
                                             'remaining_owned_pids','minimum_effective_bytes','peak_group_bytes',
                                             'elapsed_s','maximum_full_avg10','final_memory_events')}
        self.stages.setdefault('supervisors', {})[folder] = saved
        return saved

    def history(self, plan, checked_build):
        facts = {'scope':'Preserved initial failure and cold engineering tests; never recoded as a healthy runtime.'}
        self.stages['engineering_history'] = facts
        old_supervisor = self.supervisor('ros_nominal01', 'INCOMPLETE')
        old_stdout = self.jsonl(WORK/'ros_nominal01/stdout.log')
        old_stderr = self.read(WORK/'ros_nominal01/stderr.log').decode()
        self.check(len(old_stdout) == 1, 'history.old_entry_single_failure_receipt')
        failed = old_stdout[0]
        self.check(failed['status'] == 'INCOMPLETE' and failed['runtime_status'] == 'NOT_STARTED' and
                   failed['runtime_accepted'] is False and failed['scenario_assertions_passed'] is False and
                   failed['error'] == 'explicit synthetic-only opt-in required' and 'engine' not in failed,
                   'history.original_arity_failure_preserved')
        self.check(all(failed[key] == 0 for key in ('packets_received','packets_sent','efforts_observed',
                                                    'publish_calls_attempted','publish_calls_returned','physics_steps')),
                   'history.original_pre_runtime_counters')
        self.check('FileNotFoundError' in old_stderr and 'ros_nominal/report.json' in old_stderr,
                   'history.wrapper_missing_report_failure_preserved')
        source = self.read(ROOT/'runtime83/ros_probe83.cpp').decode()
        guards = list(re.finditer(r'args\.size\(\)\s*==\s*([0-9]+)', source))
        self.check(len(guards) == 1 and guards[0].group(1) == '8', 'history.frozen_guard_still_eight')
        # Source control flow, failure receipt, and the absence of any engine
        # together establish pre-initialization; missing files alone would not.
        self.check(guards[0].start() < source.index('fs::create_directory(output)') <
                   source.index('std::make_unique<SessionEngine>') < source.index('context->init'),
                   'history.failure_precedes_actor_and_DDS_initialization')
        self.check(old_supervisor['command'][-2:] == [str(WORK/'run_ros83.sh'),'nominal'],
                   'history.original_supervised_command')
        original_paths = {mode:str(EXECUTION/('ros_' + mode)) for mode in MODES}
        original_exists = {mode:Path(path).exists() for mode,path in original_paths.items()}
        self.check(not any(original_exists.values()), 'history.original_four_runtime_outputs_not_created')
        self.check(plan['first_ros_attempt']['status'] == 'INCOMPLETE' and
                   plan['first_ros_attempt']['rpc_attempts'] == plan['first_ros_attempt']['ros_contexts'] == 0 and
                   plan['first_ros_attempt']['remaining_original_modes'] == 'NOT_STARTED',
                   'history.plan_retains_original_failed_attempt')
        facts['original_attempt'] = {'failure_stdout':failed,'failure_stderr':old_stderr,
                                    'runtime_output_exists':original_exists,
                                    'rpc_attempts':0,'ros_contexts':0,
                                    'zero_count_basis':'Bound old entry failed its arity guard before output/Config/SessionEngine/DDS; not inferred merely from an absent report.'}
        red_supervisor = self.supervisor('cli_red01', 'INCOMPLETE')
        red_stdout = self.read(WORK/'cli_red01/stdout.log').decode()
        red_stderr = self.read(WORK/'cli_red01/stderr.log').decode()
        self.check('required=8, supplied=7' in red_stderr and 'COLD_ARGV_CONTRACT_FAILURE' in red_stderr,
                   'history.red_contract_failure_recorded')
        self.check(red_supervisor['returncode'] == 2, 'history.red_test_exit_two')
        facts['red_contract'] = {'status':red_supervisor['status'],'returncode':red_supervisor['returncode'],
                                 'stdout':red_stdout,'stderr':red_stderr,'expected_failure':True}
        cold_path = WORK/'cold_runtime_checked83.json'
        cold = self.json(cold_path)
        self.check(self.inputs[str(cold_path)] == plan['cold_validation_sha256'], 'history.cold_receipt_sealed')
        expected_binary = str(CHECKED_BUILD/'ros_checked83')
        self.check(cold['status'] == 'PASS' and cold['actual_binary_sha256'] ==
                   checked_build['artifact_sha256'][expected_binary], 'history.cold_actual_binary_identity')
        self.check(cold['unknown_eighth_pair_rejected'] is True and
                   'exactly seven option/value pairs required' in cold['bad_stderr'], 'history.eighth_pair_rejected')
        self.check(all(cold[key] == 0 for key in ('new_model_calls','new_actor_processes','new_physics_steps','new_ros_contexts')),
                   'history.cold_validation_zero_calls')
        self.check([row['mode'] for row in cold['valid_cases']] == list(MODES), 'history.cold_all_four_modes')
        for row in cold['valid_cases']:
            mode, receipt, command = row['mode'], row['receipt'], row['command']
            expected = plan['ros']['cases'][mode]
            self.check(row['returncode'] == 0 and command[:2] == [expected_binary,'--validate-only'] and
                       len(command[2:]) == 14 and len(dict(zip(command[2::2], command[3::2]))) == 7,
                       'history.cold_actual_seven_pairs_' + mode)
            self.check(receipt['status'] == 'COMPLETE' and receipt['validation_only'] is True and
                       receipt['option_pairs'] == 7 and receipt['mode'] == mode and receipt['ticks'] == expected['ticks'] and
                       receipt['output'] == expected['output'] and receipt['manifest'] == expected['manifest'] and
                       receipt['namespace'] == expected['namespace'] and receipt['domain'] == '183' and
                       receipt['output_created'] is False and receipt['ros_initialized'] is False and
                       receipt['actor_calls'] == receipt['model_calls'] == receipt['physics_steps'] == 0,
                       'history.cold_actual_receipt_' + mode)
        checked_source = self.read(ROOT/'runtime83/ros_checked83.cpp').decode()
        marker = '    fs::copy_file(manifest,output/"manifest.json");'
        self.check(source[source.index(marker):] == checked_source[checked_source.index(marker):],
                   'history.checked_runtime_body_unchanged_after_manifest_copy')
        facts['cold_validation'] = cold
        facts['plan_build_metadata_scope'] = 'Inherited build field describes native build; actual checked build receipt and binary verified independently.'

    def execute(self):
        plan = self.json(PLAN)
        self.check(plan['schema'] == 'd1-runtime83-checked-preregistered-v1', 'checked_plan_schema')
        parent = self.json(PARENT_PLAN)
        self.check(plan['parent_plan_sha256'] == self.inputs[str(PARENT_PLAN)], 'frozen_parent_plan_hash')
        self.check(plan['already_completed_native_requests'] == 64 and plan['native_repeat_authorized'] is False,
                   'native_reused_without_repeat')
        self.check(len(parent['source_sha256']) == 43 and len(plan['source_sha256']) == 56,
                   'registered_binding_counts')
        for path, expected in parent['source_sha256'].items():
            self.check(plan['source_sha256'].get(path) == expected, 'all43_parent_bindings_inherited', path=path)
        self.verify_map(plan['source_sha256'], 'preregistered_input_sha256')
        manifest = self.json(plan['native']['manifest'])
        self.check(manifest['max_requests'] == 64, 'native.manifest_cap64')
        for name in ('binary','model','source','ort_library'):
            self.verify_map({manifest[name]: manifest[name + '_sha256']}, 'actor34.' + name)
        reference = self.json(plan['qualified_reference']['report'])
        self.check(reference['status'] == 'COMPLETE' and reference['error'] is None and
                   reference['cpp_original_and_holdout_qualified'] is True and
                   reference['automatic_model_replacement'] is False, 'qualified34.status')
        cases = [case for stage in ('formal','holdout') for case in reference['stages'][stage]['cases']]
        self.check(len(cases) == 72, 'qualified34.all72_cases')
        for index, case in enumerate(cases):
            self.check(all(case['cpp_reference34']['transport']['identities'][key] == manifest[key]
                           for key in IDENTITIES), 'qualified34.identity_case_' + str(index))
        build = self.json(Path(plan['build']['output'])/'build_receipt.json')
        self.check(build['status'] == build['verification_after'] == 'COMPLETE' and
                   build['plan_sha256'] == self.inputs[str(PARENT_PLAN)] and
                   build['source_sha256'] == parent['source_sha256'], 'native_build.sealed_parent_and_completion')
        self.verify_map(build['artifact_sha256'], 'build.artifact_sha256')
        self.verify_map(build['observed_compiler_dependency_sha256'], 'build.observed_dependency_sha256')
        self.check(bool(build['observed_compiler_dependency_sha256']), 'build.nonempty_observed_dependency_closure')
        checked_build = self.json(CHECKED_BUILD/'build_receipt.json')
        self.check(checked_build['status'] == checked_build['verification_after'] == 'COMPLETE' and
                   checked_build['parent_plan_sha256'] == self.inputs[str(PARENT_PLAN)] and
                   checked_build['optimization'] == '-O1 -DNDEBUG' and checked_build['parallel_jobs'] == 1,
                   'checked_build.completion_and_flags')
        for path, expected in checked_build['source_sha256'].items():
            self.check(plan['source_sha256'].get(path) == expected, 'checked_build_source_sealed', path=path)
        self.verify_map(checked_build['artifact_sha256'], 'checked_build.artifact_sha256')
        self.verify_map(checked_build['observed_compiler_dependency_sha256'], 'checked_build.observed_dependency_sha256')
        self.check(bool(checked_build['observed_compiler_dependency_sha256']), 'checked_build.nonempty_dependency_closure')
        self.stages['binding_summary'] = {'preregistered_bindings': len(plan['source_sha256']),
                                         'parent_bindings': len(parent['source_sha256']),
                                         'checked_build_source_bindings': len(checked_build['source_sha256']),
                                         'checked_build_artifacts': len(checked_build['artifact_sha256']),
                                         'checked_build_observed_compiler_dependencies': len(checked_build['observed_compiler_dependency_sha256']),
                                         'build_artifacts': len(build['artifact_sha256']),
                                         'observed_compiler_dependencies': len(build['observed_compiler_dependency_sha256'])}
        ros_manifest = dict(manifest, max_requests=30)
        ros_bytes = (json.dumps(ros_manifest, indent=2, sort_keys=True)+'\n').encode()
        self.check(hashlib.sha256(ros_bytes).hexdigest() == plan['ros']['derived_manifest_sha256'], 'ros.derived_manifest_sha256')
        try:
            self.history(plan, checked_build)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self.check(False, 'history.read_or_schema_failure', error=type(exc).__name__ + ': ' + str(exc))
        for name in ('native', *MODES):
            before = len(self.failures)
            try:
                self.stage(name, manifest if name == 'native' else ros_manifest, native=name == 'native')
                self.supervisor('native_actual01' if name == 'native' else 'ros_checked_' + name + '01')
            except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as exc:
                self.check(False, name + '.read_or_schema_failure', error=type(exc).__name__ + ': ' + str(exc))
            if name in self.stages:
                self.stages[name]['status'] = 'COMPLETE' if len(self.failures) == before else 'INCOMPLETE'
        stages = [self.stages.get(name,{}) for name in ('native', *MODES)]
        attempts = [s.get('requests_attempted') for s in stages]
        completed = [s.get('requests_completed') for s in stages]
        inference = [v for s in stages for v in (s.get('session_inference_returns') or [None])]
        self.stages['totals'] = {'request_attempts': sum(attempts) if all(type(v) is int for v in attempts) else None,
                                 'request_completions': sum(completed) if all(type(v) is int for v in completed) else None,
                                 'known_session_inference_returns_lower_bound': sum(v for v in inference if type(v) is int),
                                 'unknown_or_missing_session_inference_count': sum(v is None for v in inference),
                                 'exact_inference_returns': sum(inference) if all(type(v) is int for v in inference) else None}
        self.check(attempts == [64,24,1,1,0] and completed == [64,24,1,0,0], 'all_fixed_request_counts',
                   attempts=attempts, completed=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Claim the requested new output before reading; never overwrite another audit.
    with args.output.open('x') as stream:
        json.dump({'schema':'d1-runtime83-checked-independent-audit-v1','status':'INCOMPLETE','phase':'STARTED'}, stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # These apply only when root executes this script, never during preparation.
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_CPU)
    cpu_cap = 10 if inherited_hard == resource.RLIM_INFINITY else min(10, inherited_hard)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap))
    began, cpu = time.monotonic(), time.process_time()
    audit = Audit()
    def interrupted(signum, _frame):
        raise InterruptedError('audit signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(signum, interrupted)
    signal.alarm(30)
    try:
        audit.execute()
    except BaseException as exc:
        audit.check(False, 'audit_exception', error=type(exc).__name__ + ': ' + str(exc))
    changed = []
    for path, initial in tuple(audit.inputs.items()):
        try:
            if sha(path) != initial: changed.append(path)
        except OSError:
            changed.append(path)
    audit.check(not changed, 'all_read_inputs_unchanged_after_audit', changed=changed)
    signal.alarm(0)
    report = {'schema':'d1-runtime83-checked-independent-audit-v1',
              'status':'COMPLETE' if not audit.failures else 'INCOMPLETE',
              'verification_after':'COMPLETE' if not changed else 'INCOMPLETE',
              'predicate_status':'PASS' if not audit.failures else 'FAIL',
              'checks':audit.checks, 'failed_checks':len(audit.failures), 'failures':audit.failures,
              'results':audit.stages, 'input_sha256':audit.inputs, 'audit_script_sha256':sha(__file__),
              'cpu_s':time.process_time()-cpu, 'wall_s':time.monotonic()-began,
              'new_actor_calls':0,'new_model_calls':0,'new_physics_steps':0,'parameter_updates':0,
              'latency_definitions':{
                  'units':'All recorded steady-clock timestamps are nanoseconds; divide differences by 1e6 for milliseconds.',
                  'native_capture_to_accept':'max(engine/ticks.jsonl: accepted_ns - capture_ns), all 64 native ticks.',
                  'native_capture_to_tick_end':'max(engine/ticks.jsonl: end_ns - capture_ns), all 64 native ticks; timestamp precedes trace serialization.',
                  'DDS_nominal_capture_to_publish_entry':'max(publication_events.jsonl: publish_call_begin_ns - capture_ns), 24 nominal messages.',
                  'DDS_nominal_capture_to_publish_return':'max(publication_events.jsonl: publish_return_ns - capture_ns), 24 nominal messages.',
                  'DDS_nominal_capture_to_receive':'max(observed_efforts.jsonl: receive_ns - capture_ns), 24 nominal messages; observer callback entry.',
                  'DDS_nominal_capture_to_consume_check':'max(observed_efforts.jsonl: consume_checked_ns - capture_ns), 24 nominal consumed messages.',
                  'qualification':'Each authorized sample must satisfy <=20,000,000 ns at its relevant boundaries; a maximum over this finite archive is not a hardware or worst-case execution guarantee.',
                  'faults':'Report injected late_receive and late_response separately; never mix their deliberate >20 ms delays into nominal latency. stale_command is a >500 ms command-age fixture.',
                  'interpretation':'Capture-to-accept includes adapter, serialization and transport; it is not isolated ONNX inference duration.'},
              'limitations':['Synthetic archived packets only; no physical sensor or hardware realtime qualification.',
                             'Independent byte/hash and ledger audit, not an ONNX inference rerun.',
                             'Missing/unreturned actual ORT counts remain unknown; request attempts are separate.',
                             'Expected fault scenario assertions may pass while runtime remains INCOMPLETE.',
                             'Supervisor cleanup is checked from saved receipts, not by signalling or rescanning live processes.',
                             'Checked plan build metadata names the original native build; checked build is separately bound and audited.']}
    blob = (json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temporary = args.output.with_name(args.output.name + '.final.tmp')
    with temporary.open('xb') as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,args.output)
    print(json.dumps({'status':report['status'],'checks':audit.checks,'failed_checks':len(audit.failures),
                      'output':str(args.output),'sha256':sha(args.output)}))
    return int(bool(audit.failures))


if __name__ == '__main__':
    sys.exit(main())
