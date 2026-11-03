"""Posthoc81 full-H temporal errors; stdlib only, no project/numeric imports.

Prepare only until separately authorized to run. Never modifies experiment gates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import resource
import signal
import struct
import time
import zipfile
from pathlib import Path

MODELS = (64001, 64002, 64003)
ARMS = ('C6', 'Q6')
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
RESETS = (81401, 81402, 81403)
WHEELS = (3, 7, 11, 15)
NONWHEELS = tuple(i for i in range(16) if i not in WHEELS)
WINDOWS = (('0_1', 0, 50), ('1_3', 50, 150), ('3_5', 150, 250),
           ('5_6', 250, 300), ('6_8', 300, 400), ('8_12', 400, 600))
EXTRA = (('full_0_12', 0, 600), ('translation_stable_4_8', 200, 400),
         ('stop_last_11_12', 550, 600), ('H_time_aligned_V_positive32', 175, 207),
         ('H_time_aligned_V_negative32', 325, 357))
J_KEYS = ('full_horizon_vx_mae_over_0_2', 'full_horizon_wz_mae_over_0_2',
          'stop_8_to_12_horizontal_norm_mean_over_0_2', 'fixed_early_failure_penalty')
SCALARS = ('body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'base_x_m', 'base_y_m',
           'base_height_m', 'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad', 'tilt_rad')
DRIFT_FIELDS = ('body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'base_x_m', 'base_y_m',
                'base_height_m', 'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad')
METRICS = ('yaw_mae_rad_s', 'yaw_signed_bias_rad_s', 'vx_mae_m_s', 'vx_signed_bias_m_s',
           'horizontal_speed_mean_m_s', 'horizontal_speed_mean_over_0_2',
           'yaw_J_contribution', 'vx_J_contribution', 'stop_J_contribution', 'J_contribution')
NPY_TYPES = {'<f4': ('f', 4), '<f8': ('d', 8), '<i8': ('q', 8), '|b1': ('?', 1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values) if values else None


def scalar_stats(values):
    if not values:
        return {'count': 0, 'signed_mean': None, 'mean_absolute': None, 'rms': None, 'maximum_absolute': None}
    return {'count': len(values), 'signed_mean': mean(values), 'mean_absolute': mean([abs(x) for x in values]),
            'rms': math.sqrt(mean([x*x for x in values])), 'maximum_absolute': max(map(abs, values))}


def ratio(a, b):
    return a / b if b != 0 else None


class Readback:
    def __init__(self, repo, run, cpu_budget):
        self.repo, self.run = repo, run
        self.started = time.process_time()
        self.deadline = self.started + cpu_budget
        self.inputs, self.arrays = {}, {}
        self.checks, self.max_gaps = 0, {}
        self.episodes, self.pairs = [], []

    def limit(self):
        if time.process_time() >= self.deadline:
            raise TimeoutError('stdlib readback CPU budget reached; no numeric task was started')

    def check(self, okay, message):
        self.checks += 1
        if not okay:
            raise ValueError(message)

    def close(self, key, value, expected):
        gap = abs(value - expected)
        self.max_gaps[key] = max(self.max_gaps.get(key, 0.), gap)
        self.check(math.isfinite(value) and math.isfinite(expected)
                   and gap <= 1e-12 + 1e-10 * abs(expected), key + ' differs from archived metric')

    def bind(self, path):
        path = Path(path).resolve()
        value = sha(path)
        self.check(str(path) not in self.inputs or self.inputs[str(path)] == value, 'input changed between reads')
        self.inputs[str(path)] = value
        return path

    def json(self, path):
        return json.loads(self.bind(path).read_text())

    def npz(self, path):
        path = self.bind(path)
        result = {}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            self.check(len(names) == len(set(names)), 'duplicate NPZ members')
            self.check(sum(i.file_size for i in archive.infolist()) <= 1 << 20, 'unexpected H archive allocation')
            for name in names:
                self.check(name.endswith('.npy') and '/' not in name, 'unexpected NPZ member')
                raw = archive.read(name)
                self.check(raw[:6] == b'\x93NUMPY' and raw[6] in (1, 2), 'unsupported NPY format')
                start = 10 if raw[6] == 1 else 12
                size = struct.unpack('<H' if raw[6] == 1 else '<I', raw[8:start])[0]
                h = ast.literal_eval(raw[start:start + size].decode('latin1').strip())
                self.check(h['fortran_order'] is False and h['descr'] in NPY_TYPES, 'unsupported dtype/order')
                code, width = NPY_TYPES[h['descr']]
                payload = raw[start + size:]
                self.check(len(payload) == math.prod(h['shape']) * width, 'NPY payload length')
                values = tuple(x[0] for x in struct.iter_unpack('<' + code, payload))
                self.check(all(math.isfinite(v) for v in values), 'nonfinite H data')
                result[name[:-4]] = {'shape': h['shape'], 'dtype': h['descr'], 'values': values}
        return result

    def window(self, a, lo, hi):
        vx, vy, wz, commands = (a[k]['values'] for k in ('body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'command_during_interval'))
        ey = [wz[i] - commands[3*i + 2] for i in range(lo, hi)]
        ex = [vx[i] - commands[3*i] for i in range(lo, hi)]
        speed = [math.hypot(vx[i], vy[i]) for i in range(lo, hi)]
        jc = {'yaw_J_contribution': math.fsum(map(abs, ey)) / (600*.2),
              'vx_J_contribution': math.fsum(map(abs, ex)) / (600*.2),
              'stop_J_contribution': math.fsum(math.hypot(vx[i], vy[i]) for i in range(max(lo, 400), hi)) / (200*.2)}
        result = {'start_s': lo*.02, 'end_s_exclusive': hi*.02, 'start_tick': lo, 'end_tick_exclusive': hi,
            'count': hi-lo, 'yaw_mae_rad_s': mean([abs(x) for x in ey]), 'yaw_signed_bias_rad_s': mean(ey),
            'vx_mae_m_s': mean([abs(x) for x in ex]), 'vx_signed_bias_m_s': mean(ex),
            'horizontal_speed_mean_m_s': mean(speed), 'horizontal_speed_mean_over_0_2': mean(speed)/.2,
            **jc, 'J_contribution': math.fsum(jc.values()), 'command_sign_strata': {}}
        for channel, offset, errors in (('yaw', 2, ey), ('vx', 0, ex)):
            rows = []
            for sign in (-1, 0, 1):
                indexes = [i for i in range(lo, hi) if (1 if commands[3*i+offset] > 0 else -1 if commands[3*i+offset] < 0 else 0) == sign]
                selected = [errors[i-lo] for i in indexes]
                rows.append({'sign': sign, 'count': len(indexes), 'error_mae': mean([abs(e) for e in selected]),
                    'signed_error_bias': mean(selected),
                    'command_aligned_error_bias': mean([sign*e for e in selected]) if sign else None,
                    'command_mean': mean([commands[3*i+offset] for i in indexes]),
                    'interpretation': 'aligned negative means undertracking along commanded sign; zero-command has no aligned bias'})
            result['command_sign_strata'][channel] = rows
        return result

    def episode(self, item):
        self.limit()
        path = self.run / item['trace_path']
        self.check(item['status'] == item['metrics']['status'] == 'COMPLETE' and item['metrics']['failure'] is False,
                   'only complete H episodes supported')
        self.check(sha(path) == item['trace_sha256'] == item['metrics']['trace_sha256'], 'H trace source SHA')
        self.check(self.report['raw_artifacts_sha256'].get(item['trace_path']) == sha(path), 'trace formal manifest')
        a = self.npz(path)
        required = {'reward','command_during_interval',*SCALARS,'fallen','illegal_body_contact','tau_saturation_count',
                    'issued_action','applied_action','boundary_xy_m','policy_interval_start_s'}
        self.check(set(a) == required, 'H member schema differs')
        for name in required:
            expected = (600,3) if name=='command_during_interval' else (600,16) if name in ('issued_action','applied_action') else (601,2) if name=='boundary_xy_m' else (600,)
            self.check(a[name]['shape']==expected, 'H field shape: '+name)
        self.check(a['issued_action']['dtype']=='<f4' and a['applied_action']['dtype']=='<f8', 'action precision')
        for i, t in enumerate(a['policy_interval_start_s']['values']):
            self.close('policy_clock', t, i*.02)
        self.check(not any(a['fallen']['values']) and not any(a['illegal_body_contact']['values']), 'unsafe archived H episode')
        identity = (item.get('model_seed'), item.get('arm','baseline'), item['task'], item['seed'])
        self.check(item['policy_mapping_sha256'] == self.policy_mappings[identity[:2]], 'H policy differs from sealed main candidate')
        self.check(identity not in self.arrays, 'duplicate H identity')
        self.arrays[identity] = a
        windows = {name:self.window(a,lo,hi) for name,lo,hi in WINDOWS}
        extra = {name:self.window(a,lo,hi) for name,lo,hi in EXTRA}
        full = extra['full_0_12']
        components = {J_KEYS[0]:full['vx_J_contribution'],J_KEYS[1]:full['yaw_J_contribution'],
                      J_KEYS[2]:full['stop_J_contribution'],J_KEYS[3]:0.}
        self.close('J_from_H_arrays', math.fsum(components.values()), item['metrics']['J'])
        for key,value in components.items(): self.close('J_component_'+key,value,item['metrics']['J_components'][key])
        self.close('window_J_partition', math.fsum(w['J_contribution'] for w in windows.values()),item['metrics']['J'])
        stable_yaw=max(windows['3_5']['yaw_mae_rad_s'],windows['6_8']['yaw_mae_rad_s'])
        if item['task']=='turn_stop': self.close('stable_yaw_worst_sign',stable_yaw,item['metrics']['stable_wz_mae_rad_s'])
        else: self.close('stable_translation_4_8',extra['translation_stable_4_8']['vx_mae_m_s'],item['metrics']['stable_vx_mae_m_s'])
        xy=a['boundary_xy_m']['values']
        stop_displacement=math.hypot(xy[1200]-xy[800],xy[1201]-xy[801])
        self.close('stop_displacement',stop_displacement,item['metrics']['stop_displacement_m'])
        self.close('stop_last_second_speed',extra['stop_last_11_12']['horizontal_speed_mean_m_s'],item['metrics']['stop_last_second_horizontal_speed_m_s'])
        self.close('stop_last_second_abs_yaw',mean([abs(x) for x in a['body_wz_rad_s']['values'][550:600]]),item['metrics']['stop_last_second_abs_wz_rad_s'])
        issued,applied=a['issued_action']['values'],a['applied_action']['values']
        row={'model_seed':identity[0],'arm':identity[1],'task':identity[2],'reset_seed':identity[3],
            'trace_path':item['trace_path'],'trace_sha256':item['trace_sha256'],'windows':windows,'extra_windows':extra,
            'J':math.fsum(components.values()),'J_components':components,
            'stable_yaw_worst_sign_mae':stable_yaw if item['task']=='turn_stop' else None,
            'stable_translation_mae':extra['translation_stable_4_8']['vx_mae_m_s'] if item['task']!='turn_stop' else None,
            'stop_displacement_m':stop_displacement,
            'issued_applied_exact_all_coordinates':issued==applied,
            'issued_applied_max_absolute_difference':max(abs(x-y) for x,y in zip(issued,applied,strict=True)),
            'issued_wheel_at_bound_count':sum(abs(issued[16*i+w])>=1. for i in range(600) for w in WHEELS),
            'issued_wheel_coordinate_count':2400,'tau_saturation_count_sum':sum(a['tau_saturation_count']['values']),
            'original_reference_thresholds_met':item['metrics']['reference_thresholds_met']}
        self.episodes.append(row)
        return row

    def pair_window(self, cand, base, name, lo, hi):
        a,b=self.arrays[cand],self.arrays[base]
        action={}
        for field in ('issued_action','applied_action'):
            av,bv=a[field]['values'],b[field]['values']
            cols=[[av[16*i+w]-bv[16*i+w] for i in range(lo,hi)] for w in WHEELS]
            nonwheel=[av[16*i+w]-bv[16*i+w] for i in range(lo,hi) for w in NONWHEELS]
            action[field]={'wheel_rows':list(WHEELS),'per_wheel_delta':[scalar_stats(v) for v in cols],
                'pooled_wheel_delta':scalar_stats([x for col in cols for x in col]),
                'nonwheel_delta':scalar_stats(nonwheel),
                'wheel_changed_coordinates':sum(x!=0 for col in cols for x in col),
                'nonwheel_changed_coordinates':sum(x!=0 for x in nonwheel)}
        drift={}
        for field in DRIFT_FIELDS:
            diffs=[a[field]['values'][i]-b[field]['values'][i] for i in range(lo,hi)]
            if field=='base_yaw_rad': diffs=[math.atan2(math.sin(x),math.cos(x)) for x in diffs]
            drift[field]=scalar_stats(diffs)
        return {'window':name,'action_delta_candidate_minus_reference':action,
                'post_interval_observed_component_delta':drift,
                'same_actor_input_verified':None,'action_delta_semantics':'same clock,closed-loop trajectories; not a same-input head response'}

    def pair(self,candidate,reference,kind):
        self.limit()
        ci=(candidate['model_seed'],candidate['arm'],candidate['task'],candidate['reset_seed'])
        bi=(reference['model_seed'],reference['arm'],reference['task'],reference['reset_seed'])
        a,b=self.arrays[ci],self.arrays[bi]
        self.check(a['command_during_interval']['values']==b['command_during_interval']['values'],'paired command clock differs')
        rows=[]
        for name,lo,hi in WINDOWS:
            cw,bw=candidate['windows'][name],reference['windows'][name]
            rows.append(self.pair_window(ci,bi,name,lo,hi)|{'metric_delta':{key:cw[key]-bw[key] for key in METRICS}})
        deltas={key:candidate['J_components'][key]-reference['J_components'][key] for key in J_KEYS}
        def first_change(fields, action_rows=None):
            for i in range(600):
                if action_rows is not None:
                    if any(a['issued_action']['values'][16*i+w]!=b['issued_action']['values'][16*i+w] for w in action_rows): return i
                elif any(a[field]['values'][i]!=b[field]['values'][i] for field in fields): return i
            return None
        stable_delta=(candidate['stable_yaw_worst_sign_mae']-reference['stable_yaw_worst_sign_mae']) if candidate['task']=='turn_stop' else None
        first_nonwheel=first_change((),NONWHEELS)
        row={'model_seed':candidate['model_seed'],'candidate_arm':candidate['arm'],'reference_arm':reference['arm'],
            'comparison':kind,'task':candidate['task'],'reset_seed':candidate['reset_seed'],
            'candidate_trace':candidate['trace_path'],'reference_trace':reference['trace_path'],
            'J_delta':candidate['J']-reference['J'],'J_component_delta':deltas,
            'stable_yaw_delta':stable_delta,'windows':rows,
            'first_wheel_issued_action_change_tick':first_change((),WHEELS),
            'first_nonwheel_issued_action_change_tick':first_nonwheel,
            'first_saved_post_observable_change_tick':first_change(DRIFT_FIELDS),
            'initial_world_xy_boundary_equal':a['boundary_xy_m']['values'][:2]==b['boundary_xy_m']['values'][:2],
            'same_pre_action_actor_input_verified':None,
            'nonwheel_change_interpretation':'unchanged nonwheel parameters with changed outputs is consistent with different closed-loop input/latent; H lacks627-input logs and does not identify causal model error'}
        self.close('paired_window_J_partition',math.fsum(r['metric_delta']['J_contribution'] for r in rows),row['J_delta'])
        self.close('paired_component_J_partition',math.fsum(deltas.values()),row['J_delta'])
        self.pairs.append(row)
        return row

    def aggregate(self):
        raw=[]
        for model,arm in [(None,'baseline')]+[(s,a) for s in MODELS for a in ARMS]:
            for task in TASKS:
                selected=[r for r in self.episodes if (r['model_seed'],r['arm'],r['task'])==(model,arm,task)]
                self.check(len(selected)==3,'three resets per raw aggregate')
                windows={name:{key:mean([r['windows'][name][key] for r in selected]) for key in METRICS} for name,_,_ in WINDOWS}
                raw.append({'model_seed':model,'arm':arm,'task':task,'reset_count':3,'reset_ids':list(RESETS),
                    'windows_equal_reset_mean':windows,'J_mean':mean([r['J'] for r in selected]),
                    'J_components_mean':{k:mean([r['J_components'][k] for r in selected]) for k in J_KEYS},
                    'stable_yaw_worst_sign_mean':mean([r['stable_yaw_worst_sign_mae'] for r in selected]) if task=='turn_stop' else None,
                    'stable_translation_mean':mean([r['stable_translation_mae'] for r in selected]) if task!='turn_stop' else None})
        paired=[]
        for s in MODELS:
            for kind in ('C6_minus_baseline','Q6_minus_baseline','Q6_minus_C6'):
                for task in TASKS:
                    selected=[r for r in self.pairs if (r['model_seed'],r['comparison'],r['task'])==(s,kind,task)]
                    self.check(len(selected)==3,'three paired resets per delta aggregate')
                    windows=[]
                    for index,(name,_,_) in enumerate(WINDOWS):
                        rows=[r['windows'][index] for r in selected]
                        windows.append({'window':name,'metric_delta_equal_reset_mean':{key:mean([r['metric_delta'][key] for r in rows]) for key in METRICS},
                            'each_reset_J_contribution_delta':[r['metric_delta']['J_contribution'] for r in rows],
                            'each_reset_yaw_mae_delta':[r['metric_delta']['yaw_mae_rad_s'] for r in rows],
                            'issued_wheel_delta_equal_reset_mean_absolute':mean([r['action_delta_candidate_minus_reference']['issued_action']['pooled_wheel_delta']['mean_absolute'] for r in rows]),
                            'applied_wheel_delta_equal_reset_mean_absolute':mean([r['action_delta_candidate_minus_reference']['applied_action']['pooled_wheel_delta']['mean_absolute'] for r in rows]),
                            'post_observable_rms_equal_reset_mean':{f:mean([r['post_interval_observed_component_delta'][f]['rms'] for r in rows]) for f in DRIFT_FIELDS}})
                    paired.append({'model_seed':s,'comparison':kind,'task':task,'reset_count':3,'reset_ids':list(RESETS),
                        'J_delta_mean':mean([r['J_delta'] for r in selected]),'J_component_delta_mean':{k:mean([r['J_component_delta'][k] for r in selected]) for k in J_KEYS},
                        'stable_yaw_delta_mean':mean([r['stable_yaw_delta'] for r in selected]) if task=='turn_stop' else None,
                        'windows':windows})
        return raw,paired

    def transfer_table(self, index):
        rows=[]
        branches=self.report['short_branches']
        for model in MODELS:
            for arm in ARMS:
                def vq(which):
                    selected_model=None if which=='anchor' else model
                    per=[]
                    for reset in (81207,81208,81209):
                        local=[r for r in branches if r['model_seed']==selected_model and r['arm']==which and r['seed']==reset
                               and r['tick'] in (175,325) and r['repeat']==0
                               and (which=='anchor' or (r['scale_index']==0 and r['sign']==1))]
                        self.check(len(local)==2,'V primary paired starts')
                        per.append(max(r['yaw_mae'] for r in local))
                    return per
                vbase,vactual=vq('anchor'),vq(arm)
                hbase=[index[(None,'baseline','turn_stop',r)] for r in RESETS]
                hactual=[index[(model,arm,'turn_stop',r)] for r in RESETS]
                def short(rows):
                    return [max(r['extra_windows']['H_time_aligned_V_positive32']['yaw_mae_rad_s'],
                                r['extra_windows']['H_time_aligned_V_negative32']['yaw_mae_rad_s']) for r in rows]
                hb,ha=short(hbase),short(hactual)
                bfull=[r['stable_yaw_worst_sign_mae'] for r in hbase]
                afull=[r['stable_yaw_worst_sign_mae'] for r in hactual]
                rows.append({'model_seed':model,'arm':arm,'V_reset_ids':[81207,81208,81209],'H_reset_ids':list(RESETS),
                    'V_snapshot_reset32_baseline_Q':vbase,'V_snapshot_reset32_candidate_Q':vactual,
                    'V_ratio_of_equal_reset_means':ratio(mean(vactual),mean(vbase)),
                    'H_time_aligned32_baseline_Q':hb,'H_time_aligned32_candidate_Q':ha,
                    'H_time_aligned32_ratio_of_means':ratio(mean(ha),mean(hb)),
                    'H_full_stable_baseline_Q':bfull,'H_full_stable_candidate_Q':afull,
                    'H_full_stable_ratio_of_means':ratio(mean(afull),mean(bfull)),
                    'interpretation':'V/H reset groups differ; H candidate has acted since t0 whereas V restores baseline snapshots. Both state history and reset can differ; this contrast is not a causal isolation.'})
        return rows

    def run_readback(self):
        self.report=self.json(self.run/'report.json')
        self.check(self.report['status']==self.report['verification_after']=='COMPLETE'
                   and self.report['evaluation_status']=='COMPLETE' and self.report['outcome']=='FULL_EVALUATION_COMPLETE','complete81 H required')
        plan_path=self.repo/'docs/experiments81/objective81_preregistered.json'
        plan=self.json(plan_path)
        self.check(sha(plan_path)==self.report['plan_sha256'],'81 plan binding')
        original53_path=self.repo/'docs/experiments53/model_gradient53_preregistered.json'
        original53=self.json(original53_path)
        self.check(sha(original53_path)==plan['source_sha256'][str(original53_path.relative_to(self.repo))],'nested53 plan binding')
        for name in ('tools81/run81.py','tools81/compare81.py','tools53/collection53.py','tools16/evaluate_comparison16.py','tools/evaluate_d1_ppo.py'):
            path=self.bind(self.repo/name)
            expected=plan['source_sha256'].get(name,original53['identities'].get(name,{}).get('sha256'))
            self.check(expected is not None and sha(path)==expected,'frozen evaluation source changed: '+name)
        audit_path=self.repo/'reports/objective81_policy_independent_audit_20261103.json'
        audit=self.json(audit_path)
        self.check(audit['status']=='PASS' and audit['audit_complete'] is True
                   and audit['source_report_sha256']==sha(self.run/'report.json'),'completed policy audit provenance')
        bounds=[]
        self.policy_mappings={(None,'baseline'):self.report['anchor_mapping_sha256']}
        for stage in audit['stages']:
            self.policy_mappings[(stage['seed'],stage['arm'])]=stage['main_mapping_sha256']
            p=next(p for p in stage['probes'] if p['scale_index']==0 and p['sign']==1)
            bounds.append({'model_seed':stage['seed'],'arm':stage['arm'],'parameter_row_L1_from_anchor':p['row_l1'],
                'maximum_parameter_row_L1':max(p['row_l1']),'original_0p01_bound_fraction':max(p['row_l1'])/.01,
                'same_input_theoretical_action_bound':max(p['row_l1']),
                'scope':'same bounded final latent; H closed-loop action differences need not satisfy this bound'})
        baseline=self.report['baseline'];candidate=self.report['candidates']
        self.check(len(baseline)==9 and len(candidate)==54,'all63 H episodes required')
        expected={(None,'baseline',t,r) for t in TASKS for r in RESETS}|{(s,a,t,r) for s in MODELS for a in ARMS for t in TASKS for r in RESETS}
        index={}
        for item in [*baseline,*candidate]:
            row=self.episode(item)
            index[(row['model_seed'],row['arm'],row['task'],row['reset_seed'])]=row
        self.check(set(index)==expected and len(index)==63,'exact H identity grid')
        for s in MODELS:
            for t in TASKS:
                for r in RESETS:
                    b=index[(None,'baseline',t,r)]
                    for a in ARMS:self.pair(index[(s,a,t,r)],b,a+'_minus_baseline')
                    self.pair(index[(s,'Q6',t,r)],index[(s,'C6',t,r)],'Q6_minus_C6')
        self.limit()
        raw,paired=self.aggregate()
        self.limit()
        transfer=self.transfer_table(index)
        self.limit()
        for path,digest in self.inputs.items():
            self.check(sha(path)==digest,'input changed during readback: '+path)
        self.limit()
        return {'schema':'d1-time-error82-posthoc-readback-v1','status':'COMPLETE','analysis_role':'derived_posthoc_descriptive',
            'source_report_sha256':sha(self.run/'report.json'),'source_plan_sha256':sha(plan_path),'script_sha256':sha(__file__),
            'source_policy_audit_sha256':sha(audit_path),'inputs_sha256':self.inputs,
            'checks':self.checks,'maximum_metric_reconstruction_gaps':self.max_gaps,'episode_count':63,'unique_baseline_episodes':9,
            'candidate_episodes':54,'paired_candidate_vs_baseline':54,'paired_Q6_vs_C6':27,'raw_policy_rows':37800,
            'window_partition':[{'name':n,'start_tick':lo,'end_tick_exclusive':hi,'seconds':[lo*.02,hi*.02]} for n,lo,hi in WINDOWS],
            'clock_semantics':'Interval index t carries post-transition response and command_during_interval; windows use interval start=t*0.02 exactly as original evaluator.',
            'sign_semantics':'signed bias=response-command; command-aligned negative bias means undertracking. Strata use actual per-interval command sign,including mixed1–3s.',
            'J_formula':{'vx':'sum_all600 abs(vx-command_vx)/(600*0.2)','yaw':'sum_all600 abs(wz-command_wz)/(600*0.2)',
                         'stop':'sum_ticks400..599 hypot(vx,vy)/(200*0.2)','failure':'0 for all63 complete,safe archived episodes',
                         'scope':'Original full-evaluation J; not frozen54 six-part imaginary cost. Those six terms cannot be reconstructed exactly from compact H state fields.'},
            'aggregation':'all three registered reset seeds equal-weight; baseline9episodes shared,not counted as independent per-model replicas; no scale/seed/window selection gate',
            'candidate_parameter_amplitude':bounds,'episodes':self.episodes,'episode_aggregates':raw,
            'paired_episodes':self.pairs,'paired_aggregates':paired,'V_H_temporal_scope_comparison':transfer,
            'original_decision':{'objective_passed':self.report['objective_passed'],'global_candidate_accepted':self.report['global_candidate_accepted'],
                'complete_model_eligibility':self.report['complete_model_eligibility']},
            'new_model_calls':0,'new_actor_calls':0,'new_physics_steps':0,'new_updates':0,'original_gates_changed':False,
            'cpu_seconds':time.process_time()-self.started,
            'interpretation_limits':['A single fixed+0.009 candidate per objective is available in H; no H plus/minus pair or multi-scale dose response. Finite outcome differences are not full-policy gradients.',
                'A large parameter-bound utilization or small control gain does not prove amplitude is the bottleneck; increasing the radius is neither tested nor selected here.',
                'H lacks full45-state,627 pre-action policy input,and frozen-model predictions. Observed-component drift cannot diagnose latent coverage or model distribution mismatch causally.',
                'At the same clock,candidate and baseline generally have different closed-loop observations. Their action difference includes parameter and state/history response; do not compare it to a same-input0.009 bound as a violation.',
                'Nonwheel outputs can change through changed inputs despite frozen nonwheel weights; this is not evidence of a parameter-freezing failure.',
                'V/H contrast changes reset group and policy history simultaneously; it identifies transfer failure location,not its cause.',
                'All new windows,sign strata,action andstate summaries are descriptive posthoc diagnostics. No threshold,seed,scale,policy,or prior qualification is changed.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/lyh/ddt_ros2_ws'))
    parser.add_argument('--run',type=Path,default=Path('/home/lyh/ddt_ros2_ws/evidence/objective81/execution_01'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-seconds',type=float,default=8.)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not math.isfinite(args.cpu_seconds) or not 0<args.cpu_seconds<=8.:
        raise ValueError('cooperative CPU budget must be within(0,8]; hard limit remains10s')
    resource.setrlimit(resource.RLIMIT_CPU,(9,10))
    def interrupted(signum,_frame):
        raise TimeoutError('readback hard supervision signal '+str(signum))
    for sig in (signal.SIGXCPU,signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,interrupted)
    signal.alarm(30)
    reader=Readback(args.repo.resolve(),args.run.resolve(),args.cpu_seconds)
    def incomplete(reason,stage):
        return {'schema':'d1-time-error82-posthoc-readback-v1','status':'INCOMPLETE',
            'failure':reason,'failure_stage':stage,'checks':reader.checks,
            'completed_episode_count':len(reader.episodes),'completed_pair_count':len(reader.pairs),
            'completed_episode_ids':[{k:r[k] for k in ('model_seed','arm','task','reset_seed')} for r in reader.episodes],
            'maximum_metric_reconstruction_gaps':reader.max_gaps,'inputs_sha256':reader.inputs,
            'new_model_calls':0,'new_actor_calls':0,'new_physics_steps':0,'new_updates':0,
            'cpu_seconds':time.process_time()-reader.started,'hard_cpu_limit_s':10,'wall_alarm_s':30,
            'new_output_cap_bytes':4<<20,'partial_diagnostic_semantics':'metadata only; no incomplete diagnostic is promoted to complete'}
    stage='readback'
    try:
        result=reader.run_readback()
        reader.limit()
        stage='serialization'
        result['limits_enforced']={'cpu_soft_s':9,'cpu_hard_s':10,'cooperative_cpu_s':args.cpu_seconds,
                                   'wall_alarm_s':30,'new_output_cap_bytes':4<<20}
        reader.limit()
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        reader.limit()
        if len(text.encode())>4<<20:
            text=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
            reader.limit()
        if len(text.encode())>4<<20:
            result=incomplete('compact JSON exceeds4MiB; full result was not written','output_size')
            text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        reader.limit()
    except BaseException as exc:
        result=incomplete(f'{type(exc).__name__}: {exc}',stage)
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    # The short failure summary also obeys the same exclusive-output size bound.
    if len(text.encode())>4<<20:
        result={'schema':'d1-time-error82-posthoc-readback-v1','status':'INCOMPLETE',
                'failure':'failure metadata exceeded4MiB','checks':reader.checks,
                'completed_episode_count':len(reader.episodes),'completed_pair_count':len(reader.pairs),
                'new_model_calls':0,'new_actor_calls':0,'new_physics_steps':0,'new_updates':0}
        text=json.dumps(result,allow_nan=False)+'\n'
    with args.output.open('x') as stream:stream.write(text)
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('status','checks','episode_count','paired_candidate_vs_baseline',
        'paired_Q6_vs_C6','cpu_seconds','failure')},allow_nan=False))
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
