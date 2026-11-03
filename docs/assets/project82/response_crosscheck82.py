"""Read saved81 direction responses using stdlib only; no model/pickle execution.

Posthoc diagnostics, not a new experiment or a requalification of any policy.
All fixed scales, both objectives and all three model/reset identities remain.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pickletools
import resource
import signal
import struct
import time
import zipfile
from fractions import Fraction
from pathlib import Path

MODELS = (64001, 64002, 64003)
ARMS = ('C6', 'Q6')
RESETS = (81207, 81208, 81209)
SCALES = (.009, .0045, .00225, .001125)
PREFIXES = (1, 8, 16, 32)
TICKS = (175, 325)
WHEELS = (3, 7, 11, 15)
HEAD_INDEX = tuple(i for row in WHEELS for i in range(row * 128, (row + 1) * 128))
CAP = 4 << 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values) if values else None


def stats(values):
    values = list(values)
    return {'count': len(values), 'mean': mean(values), 'min': min(values) if values else None,
            'max': max(values) if values else None,
            'rms': math.sqrt(mean([x*x for x in values])) if values else None}


def ratio(a, b):
    return a / b if b != 0 else None


def sign(x):
    return (x > 0) - (x < 0)


def response(zero, plus, minus, h):
    """Exact rational subtraction of stored floats, rounded only at output."""
    z, p, m, step = map(Fraction.from_float, (zero, plus, minus, h))
    odd, even = (p-m)/2, (p+m)/2-z
    return {'zero': zero, 'plus': plus, 'minus': minus,
            'plus_delta': float(p-z), 'minus_delta': float(m-z),
            'odd_change': float(odd), 'even_midpoint_offset': float(even),
            'central': float(odd/step), 'forward': float((p-z)/step),
            'backward': float((z-m)/step),
            'even_over_abs_odd': ratio(float(even), abs(float(odd))),
            'plus_down': p < z, 'minus_up': m > z, 'strong_descent': p < z < m,
            'both_endpoints_worse': p > z and m > z,
            'plus_equals_zero': p == z, 'minus_equals_zero': m == z,
            'plus_closure_gap': float((p-z)-(odd+even)),
            'minus_closure_gap': float((m-z)-(-odd+even)),
            'interpretation': 'odd/even finite endpoint decomposition; even is not a curvature estimate'}


class Readback:
    def __init__(self, repo, run, cpu_budget):
        self.repo, self.run = repo, run
        self.started = time.process_time()
        self.deadline = self.started + cpu_budget
        self.inputs, self.cache, self.max_gaps = {}, {}, {}
        self.checks = 0
        self.rows, self.parameters, self.action_rows = [], [], []
        self.manifest = {}

    def limit(self):
        if time.process_time() >= self.deadline:
            raise TimeoutError('cooperative stdlib CPU budget exhausted')

    def check(self, yes, message):
        self.checks += 1
        if not yes:
            raise ValueError(message)

    def close(self, name, actual, expected):
        self.check(math.isfinite(actual) and math.isfinite(expected), name+' nonfinite')
        gap = abs(actual-expected)
        self.max_gaps[name] = max(self.max_gaps.get(name, 0.), gap)
        self.check(gap <= 1e-12 + 1e-10*abs(expected), name+' source reconstruction differs')

    def bind(self, path, expected=None):
        self.limit()
        path = Path(path).resolve()
        digest = sha(path)
        self.check(str(path) not in self.inputs or self.inputs[str(path)] == digest, 'input changed')
        self.inputs[str(path)] = digest
        if expected is not None:
            self.check(digest == expected, 'explicit source SHA mismatch: '+str(path))
        if path.is_relative_to(self.run) and path.name != 'report.json':
            rel = str(path.relative_to(self.run))
            self.check(self.manifest.get(rel) == digest, 'formal81 manifest mismatch: '+rel)
        return path

    def json(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def npy(self, raw):
        self.check(raw[:6] == b'\x93NUMPY' and raw[6] in (1, 2), 'NPY format')
        start = 10 if raw[6] == 1 else 12
        size = struct.unpack('<H' if raw[6] == 1 else '<I', raw[8:start])[0]
        h = ast.literal_eval(raw[start:start+size].decode('latin1').strip())
        self.check(not h['fortran_order'] and h['descr'] in ('<f4', '<f8', '|b1'), 'NPY dtype/order')
        code, width = {'<f4': ('f', 4), '<f8': ('d', 8), '|b1': ('?', 1)}[h['descr']]
        data = raw[start+size:]
        self.check(len(data) == math.prod(h['shape'])*width, 'NPY payload length')
        values = tuple(x[0] for x in struct.iter_unpack('<'+code, data))
        self.check(all(math.isfinite(x) for x in values), 'nonfinite saved trace')
        return h['shape'], values

    def trace(self, row, physical):
        key = row['trace_path']
        if key in self.cache:
            return self.cache[key]
        path = self.bind(self.run/key, row['trace_sha256'])
        if physical:
            self.check(row['status'] == 'COMPLETE' and row['steps'] == 32, 'physical branch incomplete')
        with zipfile.ZipFile(path) as z:
            self.check(len(z.namelist()) == len(set(z.namelist())), 'duplicate NPZ members')
            self.check(sum(x.file_size for x in z.infolist()) <= 1 << 20, 'unexpected trace size')
            names = ('body_wz', 'commands', 'costs', 'issued_actions', 'first_actor_input') if physical else (
                'states', 'commands', 'costs', 'actions', 'actor_inputs')
            a = {n: self.npy(z.read(n+'.npy')) for n in names}
        self.check(a['commands'][0] == (32, 3) and a['costs'][0] == (32,), 'trace command/cost shape')
        if physical:
            self.check(a['body_wz'][0] == (32,) and a['issued_actions'][0] == (32, 16)
                       and a['first_actor_input'][0] == (627,), 'physical response shape')
            wz = a['body_wz'][1]
            action, actor_input = a['issued_actions'][1][:16], a['first_actor_input'][1]
        else:
            self.check(a['states'][0] == (32, 45) and a['actions'][0] == (32, 16)
                       and a['actor_inputs'][0] == (32, 627), 'model response shape')
            wz = a['states'][1][28::45]
            action, actor_input = a['actions'][1][:16], a['actor_inputs'][1][:627]
        errors = [w-a['commands'][1][3*i+2] for i, w in enumerate(wz)]
        prefixes = {h: {'C': mean(a['costs'][1][:h]), 'MAE': mean([abs(v) for v in errors[:h]])}
                    for h in PREFIXES}
        for h, v in prefixes.items():
            old = row['prefix_metrics'][str(h)]
            self.close('prefix_cost', v['C'], old['cost'])
            self.close('prefix_yaw_mae', v['MAE'], old['yaw_mae'])
        result = {'prefixes': prefixes, 'first_action': action, 'first_actor_input': actor_input,
                  'error_signs': tuple(sign(x) for x in errors)}
        self.cache[key] = result
        return result

    def pair(self, rows, physical):
        self.check(len(rows) == 2 and {r['tick'] for r in rows} == set(TICKS), '175/325 pair missing')
        traces = [(r['tick'], self.trace(r, physical)) for r in rows]
        result = {}
        for h in PREFIXES:
            q = max(t['prefixes'][h]['MAE'] for _, t in traces)
            result[h] = {'C': mean([t['prefixes'][h]['C'] for _, t in traces]), 'Q': q,
                         'max_ticks': sorted(tick for tick, t in traces if t['prefixes'][h]['MAE'] == q)}
        return result

    def tensor_descriptors(self, ops):
        """Parse this fixed protocol2 data grammar without invoking pickle globals.

        GLOBAL/REDUCE/PERSID become symbolic tuples only. No callable is imported,
        instantiated or executed; unsupported construction forms fail closed.
        """
        allowed = {'torch._utils _rebuild_tensor_v2', 'torch FloatStorage',
                   'torch DoubleStorage', 'collections OrderedDict'}
        stack, memo, marker = [], {}, object()

        def marked_items():
            self.check(any(x is marker for x in stack), 'descriptor MARK missing')
            at = next(i for i in range(len(stack)-1, -1, -1) if stack[i] is marker)
            items = stack[at+1:]
            del stack[at:]
            return items

        for index, (op, arg, _position) in enumerate(ops):
            name = op.name
            if name == 'PROTO':
                self.check(index == 0 and arg == 2, 'only fixed protocol2 descriptors supported')
            elif name == 'GLOBAL':
                self.check(arg in allowed, 'descriptor global unsupported')
                stack.append(('global', arg))
            elif name in ('BINUNICODE', 'BININT', 'BININT1', 'BININT2'):
                stack.append(arg)
            elif name == 'NEWFALSE':
                stack.append(False)
            elif name == 'EMPTY_DICT':
                stack.append({})
            elif name == 'EMPTY_TUPLE':
                stack.append(())
            elif name == 'MARK':
                stack.append(marker)
            elif name == 'BINPUT':
                self.check(bool(stack) and arg not in memo, 'descriptor memo store invalid')
                memo[arg] = stack[-1]
            elif name == 'BINGET':
                self.check(arg in memo, 'descriptor memo reference missing')
                stack.append(memo[arg])
            elif name == 'TUPLE':
                stack.append(tuple(marked_items()))
            elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                count = int(name[-1])
                self.check(len(stack) >= count and not any(x is marker for x in stack[-count:]),
                           'descriptor tuple operands invalid')
                value = tuple(stack[-count:])
                del stack[-count:]
                stack.append(value)
            elif name == 'BINPERSID':
                self.check(bool(stack), 'persistent storage descriptor missing')
                value = stack.pop()
                self.check(type(value) is tuple and len(value) == 5 and value[0] == 'storage',
                           'persistent storage descriptor malformed')
                stack.append(('storage', value))
            elif name == 'REDUCE':
                self.check(len(stack) >= 2, 'descriptor reduction operands missing')
                args, function = stack.pop(), stack.pop()
                if function == ('global', 'collections OrderedDict'):
                    self.check(args == (), 'nonempty hook constructor unsupported')
                    stack.append({})
                else:
                    self.check(function == ('global', 'torch._utils _rebuild_tensor_v2')
                               and type(args) is tuple and len(args) == 6,
                               'tensor reconstruction descriptor malformed')
                    stack.append(('tensor', args))
            elif name == 'SETITEMS':
                items = marked_items()
                self.check(bool(stack) and type(stack[-1]) is dict and len(items) % 2 == 0,
                           'descriptor dictionary operands invalid')
                for key, value in zip(items[::2], items[1::2], strict=True):
                    self.check(type(key) is str and key not in stack[-1], 'duplicate/nonstring descriptor key')
                    stack[-1][key] = value
            elif name == 'STOP':
                self.check(index == len(ops)-1 and len(stack) == 1 and type(stack[0]) is dict,
                           'descriptor root/trailing opcodes invalid')
                return stack[0]
            else:
                raise ValueError('unsupported fixed descriptor opcode: '+name)
        raise ValueError('descriptor STOP missing')

    def read_storage(self, path, expected, kind):
        """Read fixed documented tensor storages; never execute pickle opcodes."""
        path = self.bind(path, expected)
        layouts = {'actual': [('weight', 2048, 'f'), ('bias', 16, 'f'),
                              ('delta_weight64', 2048, 'd'), ('delta_bias64', 16, 'd')],
                   'ideal': [('weight', 2048, 'd'), ('bias', 16, 'd')],
                   'step': [('head_before_weight64', 2048, 'd'), ('head_before_bias64', 16, 'd'),
                            ('masked_gradient_weight64', 2048, 'd'), ('masked_gradient_bias64', 16, 'd'),
                            ('direction_weight64', 2048, 'd'), ('direction_bias64', 16, 'd')]}
        layout = layouts[kind]
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            self.check(len(names) == len(set(names)) and sum(i.file_size for i in z.infolist()) < 100000,
                       'unexpected parameter archive')
            pkl = [n for n in names if n.endswith('/data.pkl')]
            self.check(len(pkl) == 1, 'parameter pickle descriptor missing')
            prefix = pkl[0].rsplit('/', 1)[0]
            self.check(z.read(prefix+'/byteorder') == b'little', 'parameter storage byte order')
            ops = list(pickletools.genops(z.read(pkl[0])))
            globals_seen = {arg for op, arg, _ in ops if op.name == 'GLOBAL'}
            allowed = {'torch._utils _rebuild_tensor_v2', 'torch FloatStorage',
                       'torch DoubleStorage', 'collections OrderedDict'}
            self.check(globals_seen <= allowed and 'torch._utils _rebuild_tensor_v2' in globals_seen,
                       'unrecognized storage descriptor global')
            self.check(not any(op.name in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4', 'INST', 'OBJ', 'NEWOBJ',
                                          'NEWOBJ_EX', 'BUILD') for op, _, _ in ops), 'unexpected pickle descriptor opcode')
            strings = [arg for op, arg, _ in ops if op.name in ('BINUNICODE', 'SHORT_BINUNICODE')]
            observed = [s for s in strings if s in {x[0] for x in layout}]
            self.check(observed == [x[0] for x in layout], 'tensor descriptor order changed')
            self.check([s for s in strings if s.isdigit()] == [str(i) for i in range(len(layout))],
                       'storage key order changed')
            self.check(sum(op.name == 'BINPERSID' for op, _, _ in ops) == len(layout), 'storage descriptor count')
            self.check({n for n in names if '/data/' in n} ==
                       {prefix+'/data/'+str(i) for i in range(len(layout))}, 'extra tensor storage')
            descriptors = self.tensor_descriptors(ops)
            expected_keys = {entry[0] for entry in layout}
            if kind == 'step':
                expected_keys.add('anchor_mapping_sha256')
                self.check(type(descriptors.get('anchor_mapping_sha256')) is str,
                           'anchor identity descriptor missing')
            self.check(set(descriptors) == expected_keys, 'tensor root keys differ')
            result = {}
            for i, (name, count, code) in enumerate(layout):
                descriptor = descriptors[name]
                self.check(type(descriptor) is tuple and len(descriptor) == 2
                           and descriptor[0] == 'tensor', 'tensor descriptor missing: '+name)
                storage, offset, shape, stride, requires_grad, hooks = descriptor[1]
                storage_type = 'torch FloatStorage' if code == 'f' else 'torch DoubleStorage'
                expected_storage = ('storage', ('storage', ('global', storage_type), str(i), 'cpu', count))
                self.check(storage == expected_storage, 'tensor storage type/key/device/numel differs: '+name)
                expected_shape, expected_stride = ((16, 128), (128, 1)) if count == 2048 else ((16,), (1,))
                self.check(type(offset) is int and offset == 0, 'nonzero/noninteger tensor offset: '+name)
                self.check(type(shape) is tuple and all(type(x) is int for x in shape)
                           and shape == expected_shape, 'tensor shape differs: '+name)
                self.check(type(stride) is tuple and all(type(x) is int for x in stride)
                           and stride == expected_stride, 'tensor stride differs: '+name)
                self.check(requires_grad is False and type(hooks) is dict and not hooks,
                           'tensor gradient/hooks descriptor differs: '+name)
                data = z.read(prefix+'/data/'+str(i))
                self.check(len(data) == count*struct.calcsize(code), 'tensor storage byte length')
                vals = tuple(x[0] for x in struct.iter_unpack('<'+code, data))
                self.check(all(math.isfinite(x) for x in vals), 'nonfinite parameter storage')
                result[name] = vals
        return result

    @staticmethod
    def wheel_vector(values, weight='weight', bias='bias'):
        return tuple(values[weight][i] for i in HEAD_INDEX) + tuple(values[bias][i] for i in WHEELS)

    def parameters_readback(self):
        for model_seed in MODELS:
            for arm in ARMS:
                folder = self.run/f'seed{model_seed}'/f'actor_{arm}'
                receipt = self.json(folder/'receipt.json')
                self.check(receipt['status'] == receipt['verification_after'] == 'COMPLETE'
                           and receipt['updates'] == 1 and receipt['model_updates'] == 0
                           and receipt['scales'] == list(SCALES), 'actor receipt incomplete')
                step = self.read_storage(folder/receipt['head_step_file'], receipt['head_step_file_sha256'], 'step')
                anchor = self.wheel_vector(step, 'head_before_weight64', 'head_before_bias64')
                direction = self.wheel_vector(step, 'direction_weight64', 'direction_bias64')
                grad = self.wheel_vector(step, 'masked_gradient_weight64', 'masked_gradient_bias64')
                self.close('D_analytic_storage', math.fsum(g*d for g, d in zip(grad, direction, strict=True)),
                           receipt['directional_derivative'])
                probes = {(p['scale_index'], p['sign']): p for p in receipt['probes']}
                self.check(len(probes) == 8 and set(probes) == {(k,s) for k in range(4) for s in (-1,1)},
                           'probe parameter grid missing')
                for k, h in enumerate(SCALES):
                    endpoints = {}
                    for s in (-1, 1):
                        p = probes[k,s]
                        self.check(p['status'] == 'COMPLETE' and p['epsilon'] == h, 'probe metadata mismatch')
                        a = self.read_storage(folder/p['actual_file'], p['actual_file_sha256'], 'actual')
                        ideal = self.read_storage(folder/p['ideal_file'], p['ideal_file_sha256'], 'ideal')
                        av, iv = self.wheel_vector(a), self.wheel_vector(ideal)
                        dv = self.wheel_vector(a, 'delta_weight64', 'delta_bias64')
                        self.check(all(d == x-z for d,x,z in zip(dv,av,anchor,strict=True)), 'actual delta storage inconsistent')
                        self.check(all(x == z+s*h*d for x,z,d in zip(iv,anchor,direction,strict=True)),
                                   'ideal endpoint differs from fixed sealed line')
                        self.check(all(x == struct.unpack('<f', struct.pack('<f', y))[0]
                                       for x,y in zip(av,iv,strict=True)), 'actual endpoint is not ideal rounded F32')
                        nonwheel = [i for i in range(2048) if i not in HEAD_INDEX]
                        self.check(all(a['weight'][i] == step['head_before_weight64'][i] for i in nonwheel)
                                   and all(a['bias'][i] == step['head_before_bias64'][i]
                                           for i in range(16) if i not in WHEELS), 'protected head coordinates changed')
                        e = [x-y for x,y in zip(av,iv,strict=True)]
                        delta_norm = math.sqrt(math.fsum(x*x for x in dv))
                        endpoints[s] = {'vector': av, 'delta': dv,
                            'rounding_l2': math.sqrt(math.fsum(x*x for x in e)), 'rounding_maxabs': max(map(abs,e)),
                            'rounding_l2_over_actual_delta_l2': ratio(math.sqrt(math.fsum(x*x for x in e)),delta_norm),
                            'changed_516': sum(x != 0 for x in dv), 'recorded_parameter_ulps': p['parameter_ulps']}
                    odd = [(p-m)/2 for p,m in zip(endpoints[1]['vector'], endpoints[-1]['vector'],strict=True)]
                    even = [(p+m)/2-z for p,m,z in zip(endpoints[1]['vector'],endpoints[-1]['vector'],anchor,strict=True)]
                    ideal_odd = [h*d for d in direction]
                    dref = receipt['selfchecks'][k]
                    self.check(dref['epsilon'] == h, 'ideal D selfcheck scale mismatch')
                    dresp = response(receipt['before'][arm], probes[k,1]['ideal_D_values'][arm],
                                     probes[k,-1]['ideal_D_values'][arm],h)
                    self.close('D_ideal_fd',dresp['central'],dref['ideal_model_fd'])
                    self.parameters.append({'model_seed':model_seed,'arm':arm,'scale_index':k,'h':h,
                        'actual_f32_endpoints':{str(s):{n:v for n,v in e.items() if n not in ('vector','delta')}
                                                for s,e in endpoints.items()},
                        'actual_parameter_even_l2':math.sqrt(math.fsum(x*x for x in even)),
                        'actual_parameter_even_maxabs':max(map(abs,even)),
                        'actual_parameter_odd_minus_ideal_l2':math.sqrt(math.fsum((x-y)**2 for x,y in zip(odd,ideal_odd,strict=True))),
                        'actual_parameter_even_over_odd_l2':ratio(math.sqrt(math.fsum(x*x for x in even)),math.sqrt(math.fsum(x*x for x in odd))),
                        'D_ideal64_objective_response':dresp,'D_ideal64_selfcheck':dref,
                        'same_D_actualF32_rollout_available':False,'same_V_ideal64_rollout_available':False})

    def run_readback(self):
        report = self.json(self.run/'report.json')
        self.report = report
        self.check(report['status'] == report['verification_after'] == 'COMPLETE', 'formal81 incomplete')
        self.manifest = report['raw_artifacts_sha256']
        plan = self.repo/'docs/experiments81/objective81_preregistered.json'
        self.bind(plan,report['plan_sha256'])
        for name,digest in report['seals'].items():
            self.bind(self.run/name,digest)
        cross = self.json(self.run/'cross_readback.json')
        multiscale = self.json(self.run/'multiscale_diagnostics.json')
        ad = self.json(self.run/'predictions/validation_ad.json')
        self.check(cross['status'] == 'COMPLETE' and len(cross['responses']) == 288, 'cross record grid')
        predictions, branches = report['predictions'],report['short_branches']
        self.check(len(predictions) == 387 and len(branches) == 372, 'formal branch/prediction counts')
        def identity(r):
            return (r['model_seed'],r['arm'],r['scale_index'],r['sign'],r['seed'],r['tick'])
        pi = {identity(r):r for r in predictions}
        bi = {identity(r):r for r in branches if r['repeat'] == 0}
        self.check(len(pi) == 387 and len(bi) == 357, 'duplicate prediction/physical identity')
        old = {(r['model_seed'],r['arm'],r['seed'],r['scale_index'],r['horizon']):r for r in cross['responses']}
        ads = {(r['model_seed'],r['arm'],r['seed'],r['metric']):r for r in ad}
        self.check(len(old) == 288 and len(ads) == 36, 'duplicate cross/AD identity')
        ms = {(r['model_seed'],r['arm'],r['seed'],r['scale_index']):r
              for r in multiscale['by_model_arm_reset_scale']}
        self.check(len(ms) == 72, 'multiscale grid size')
        for model_seed in MODELS:
            for arm in ARMS:
                for seed in RESETS:
                    for k,h in enumerate(SCALES):
                        grouped = {}
                        for label,index,physical in (('model',pi,False),('physical',bi,True)):
                            z = [index[(None if physical else model_seed,'anchor',None,0,seed,t)] for t in TICKS]
                            plus = [index[(model_seed,arm,k,1,seed,t)] for t in TICKS]
                            minus = [index[(model_seed,arm,k,-1,seed,t)] for t in TICKS]
                            grouped[label] = [self.pair(rows,physical) for rows in (z,plus,minus)]
                        for horizon in PREFIXES:
                            src = old[model_seed,arm,seed,k,horizon]
                            self.check(src['h'] == h, 'cross fixed h changed')
                            row = {'model_seed':model_seed,'arm':arm,'seed':seed,'scale_index':k,'h':h,'horizon':horizon}
                            for label,(z,p,m) in grouped.items():
                                row[label] = {metric:response(z[horizon][metric],p[horizon][metric],m[horizon][metric],h)
                                              for metric in ('C','Q')}
                                row[label]['max_ticks'] = {n:v[horizon]['max_ticks'] for n,v in zip(('zero','plus','minus'),(z,p,m),strict=True)}
                                row[label]['max_tick_switch'] = z[horizon]['max_ticks'] != p[horizon]['max_ticks'] or z[horizon]['max_ticks'] != m[horizon]['max_ticks']
                                for metric in ('C','Q'):
                                    for key in ('zero','plus','minus','central','forward','backward','plus_delta','minus_delta'):
                                        self.close('cross_'+key,row[label][metric][key],src[label][metric][key])
                            row['comparison'] = {}
                            for metric in ('C','Q'):
                                model,physical = row['model'][metric],row['physical'][metric]
                                row['comparison'][metric] = {f'{key}_same_nonzero_sign':sign(model[key]) == sign(physical[key]) != 0
                                    for key in ('central','forward','backward','plus_delta','odd_change','even_midpoint_offset')}
                                row['comparison'][metric].update({f'model_over_physical_{key}_raw':ratio(model[key],physical[key])
                                    for key in ('central','forward','backward','plus_delta','odd_change','even_midpoint_offset')})
                                if horizon == 32:
                                    ad_value = ads[model_seed,arm,seed,metric]['model_directional_ad']
                                    row['comparison'][metric].update(model_ideal_direction_AD=ad_value,
                                        model_actualF32_parameter_endpoint_FD_minus_ideal_direction_AD=model['central']-ad_value,
                                        physical_FD_minus_model_AD=physical['central']-ad_value)
                            if horizon == 32:
                                self.close('multiscale_Q_plus',row['physical']['Q']['plus'],ms[model_seed,arm,seed,k]['Q_plus'])
                                self.close('multiscale_Q_zero',row['physical']['Q']['zero'],ms[model_seed,arm,seed,k]['Q_zero'])
                                self.close('multiscale_Q_minus',row['physical']['Q']['minus'],ms[model_seed,arm,seed,k]['Q_minus'])
                            self.rows.append(row)
        # First-action parity and finite amplitude, using every steady endpoint.
        action_index = {identity(r):r for r in cross['actual_action_response']}
        self.check(len(action_index) == 342, 'actual action response grid')
        for key,row in bi.items():
            model_seed,arm,k,s,seed,tick = key
            if arm == 'anchor' or tick not in TICKS:
                continue
            p = self.trace(row,True)
            z = self.trace(bi[(None,'anchor',None,0,seed,tick)],True)
            m = self.trace(pi[key],False)
            source = action_index[key]
            self.check(p['first_actor_input'] == z['first_actor_input'], 'physical anchor/endpoint initial actor input values differ')
            delta = [a-b for a,b in zip(p['first_action'],z['first_action'],strict=True)]
            for value,expected in zip(delta,source['first_action']['delta'],strict=True):
                self.close('physical_first_action_delta',value,expected)
            self.action_rows.append({'model_seed':model_seed,'arm':arm,'seed':seed,'tick':tick,'scale_index':k,'sign':s,
                'all_four_wheels_nonzero':all(delta[i] != 0 for i in WHEELS),
                'wheel_min_abs_delta':min(abs(delta[i]) for i in WHEELS),
                'first_actor_input_value_exact_to_anchor':True,
                'model_vs_physical_first_action_maxabs':max(abs(a-b) for a,b in zip(m['first_action'],p['first_action'],strict=True)),
                'model_vs_physical_first_actor_input_maxabs':max(abs(a-b) for a,b in zip(m['first_actor_input'],p['first_actor_input'],strict=True)),
                'recorded_min_nonzero_action_delta_ulps':source['first_action']['min_nonzero_delta_ulps'],
                'physical_yaw_sign_switches_from_anchor':sum(a != b for a,b in zip(p['error_signs'],z['error_signs'],strict=True))})
        self.check(len(self.action_rows) == 288, 'steady action precision grid incomplete')
        self.parameters_readback()
        summaries = []
        for arm in ARMS:
            for horizon in PREFIXES:
                for k,h in enumerate(SCALES):
                    rows = [r for r in self.rows if r['arm'] == arm and r['horizon'] == horizon and r['scale_index'] == k]
                    self.check(len(rows) == 9, 'summary grid incomplete')
                    summary = {'arm':arm,'horizon':horizon,'scale_index':k,'h':h,'paired_rows':9,'independent_environment_resets':3}
                    for metric in ('C','Q'):
                        item = {}
                        for label in ('model','physical'):
                            vals = [r[label][metric] for r in rows]
                            item[label] = {name+'_count':sum(v[name] for v in vals) for name in
                                ('plus_down','minus_up','strong_descent','both_endpoints_worse','plus_equals_zero','minus_equals_zero')}
                            item[label].update(negative_central_count=sum(v['central'] < 0 for v in vals),
                                positive_central_count=sum(v['central'] > 0 for v in vals),
                                central=stats(v['central'] for v in vals),odd_change=stats(v['odd_change'] for v in vals),
                                even_midpoint_offset=stats(v['even_midpoint_offset'] for v in vals))
                        item['same_sign_counts'] = {key:sum(r['comparison'][metric][key+'_same_nonzero_sign'] for r in rows)
                                                   for key in ('central','forward','backward','plus_delta')}
                        item['raw_model_over_physical_central_ratios'] = stats(r['comparison'][metric]['model_over_physical_central_raw']
                            for r in rows if r['comparison'][metric]['model_over_physical_central_raw'] is not None)
                        summary[metric] = item
                    summaries.append(summary)
        model_means = []
        for model_seed in MODELS:
            for arm in ARMS:
                for k,h in enumerate(SCALES):
                    rows = [r for r in self.rows if (r['model_seed'],r['arm'],r['scale_index'],r['horizon']) == (model_seed,arm,k,32)]
                    self.check(len(rows) == 3, 'model/reset mean grid')
                    model_means.append({'model_seed':model_seed,'arm':arm,'scale_index':k,'h':h,
                        **{label:{metric:response(*(mean([r[label][metric][part] for r in rows]) for part in ('zero','plus','minus')),h)
                                   for metric in ('C','Q')} for label in ('model','physical')}})
        for path,digest in self.inputs.items():
            self.limit()
            self.check(sha(path) == digest, 'late input identity mismatch')
        return {'schema':'d1-response-crosscheck82-v1','status':'COMPLETE','analysis_role':'posthoc_descriptive',
            'script_sha256':sha(__file__),'inputs_sha256':self.inputs,'source_report_sha256':sha(self.run/'report.json'),
            'source_plan_sha256':report['plan_sha256'],'checks':self.checks,'maximum_reconstruction_gaps':self.max_gaps,
            'counts':{'response_rows_all_prefixes':len(self.rows),'response_rows_H32':72,
                      'steady_endpoint_actions':len(self.action_rows),'parameter_pairs':len(self.parameters),
                      'model_seeds':3,'independent_V_reset_seeds':3,'scales':4,'arms':2},
            'definitions':{'odd':'(plus-minus)/2','even_midpoint_offset':'(plus+minus)/2-zero',
                'plus_delta':'odd+even','minus_delta':'-odd+even','central':'odd/h',
                'C':'mean of the two175/325 H-prefix mean frozen54 six-term costs; not full evaluation J',
                'Q':'max of175/325 H-prefix yaw MAEs; then equal-weight reset mean when explicitly aggregated',
                'ratios':'raw signed ratios; null for exact zero denominator only; not a qualification or confidence estimate'},
            'summaries_by_arm_prefix_scale':summaries,'H32_model_reset_means':model_means,
            'H32_Q_plus_not_down_rows':[{k:r[k] for k in ('model_seed','arm','seed','scale_index','h')}
                | {'model':r['model']['Q'],'physical':r['physical']['Q'],
                   'physical_central_nonnegative':r['physical']['Q']['central'] >= 0,
                   'negative_odd_overridden_by_positive_even':r['physical']['Q']['odd_change'] < 0
                       and r['physical']['Q']['even_midpoint_offset'] >= -r['physical']['Q']['odd_change']}
                for r in self.rows if r['horizon'] == 32 and not r['physical']['Q']['plus_down']],
            'responses':self.rows,'first_action_precision':self.action_rows,'parameter_precision_and_D_selfchecks':self.parameters,
            'precision_contract':{'D':'ideal-F64 policy endpoints and F64 model, D states only',
                'V_model':'actual-F32 policy parameters promoted toF64; F64 actor/control/model arithmetic',
                'V_physical':'actual-F32 policy execution, MuJoCo physical dynamics',
                'same_input_D_actual_vs_ideal_rollout_pair_available':False,
                'same_input_V_actual_vs_ideal_rollout_pair_available':False,'F64_physical_endpoint_available':False,
                'actor_input_equality':'decoded numeric equality; signed zeros are not distinguished',
                'pt_reading':'raw zip storage with verified fixed type/key/device/numel/offset/shape/stride; symbolic descriptor parsing never imports or invokes pickle globals or reconstruction code'},
            'limits_of_inference':[
                'Finite-scale sign/odd/even response is not an independently qualified derivative at h→0.',
                'The even midpoint offset can include nonlinear dynamics,parameter/action rounding,clamping and nonsmooth objective effects; it is not a causal curvature estimate.',
                'The four fixed scales are all retained. No threshold,step,seed,policy or previous qualification is selected or changed.',
                'AD is an idealF64 line; V modelFD uses quantized parameter endpoints. Their difference includes finite-step and parameter-rounding effects.',
                'D ideal endpoint targets and V actual endpoint targets are from different states; they are not a controlled precision comparison.',
                'Nonzero large actionULPs and small parameter-rounding errors exclude exact initial rounding-to-zero only,not nonlinear amplification or a quantization contribution later.',
                'No physics/model calls are made. The initial point derivative of real physics remains unavailable.',
                'Three model copies sharing three reset seeds do not yield nine independent environment seeds.',
                'V localQ/cost evidence does not requalify fullH performance or identify the cause of V-to-H ordering reversal.'],
            'original_decision':{k:report.get(k) for k in ('objective_passed','global_candidate_accepted','complete_model_eligibility')},
            'new_model_calls':0,'new_actor_calls':0,'new_physics_steps':0,'new_updates':0,'original_gates_changed':False,
            'cpu_seconds':time.process_time()-self.started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/lyh/ddt_ros2_ws'))
    parser.add_argument('--run',type=Path,default=Path('/home/lyh/ddt_ros2_ws/evidence/objective81/execution_01'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-seconds',type=float,default=8.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not math.isfinite(args.cpu_seconds) or not 0 < args.cpu_seconds <= 8:
        raise ValueError('cooperative CPU budget must be in(0,8]')
    resource.setrlimit(resource.RLIMIT_CPU,(9,10))
    def interrupted(signum,_frame):
        raise TimeoutError('readback supervisor signal '+str(signum))
    for sig in (signal.SIGXCPU,signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,interrupted)
    signal.alarm(30)
    started_wall = time.monotonic()
    reader = Readback(args.repo.resolve(),args.run.resolve(),args.cpu_seconds)
    def incomplete(exc):
        return {'schema':'d1-response-crosscheck82-v1','status':'INCOMPLETE','failure':str(exc),
            'completed_response_rows':len(reader.rows),'completed_parameter_pairs':len(reader.parameters),
            'checks':reader.checks,'inputs_sha256':reader.inputs,'maximum_reconstruction_gaps':reader.max_gaps,
            'new_model_calls':0,'new_actor_calls':0,'new_physics_steps':0,'new_updates':0,
            'original_gates_changed':False,'cpu_seconds':time.process_time()-reader.started,
            'partial_semantics':'completed counts and identities retained; no incomplete diagnostic is complete'}
    try:
        result = reader.run_readback()
        reader.limit()
        result['limits_enforced'] = {'cpu_soft_s':9,'cpu_hard_s':10,'cooperative_cpu_s':args.cpu_seconds,
                                     'wall_alarm_s':30,'output_cap_bytes':CAP}
        result['wall_seconds'] = time.monotonic()-started_wall
        encoded = json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
        reader.limit()
        if len(encoded.encode()) > CAP:
            raise ValueError('complete compact result exceeds4MiB')
    except BaseException as exc:  # noqa: BLE001 - preserve interruption/deadline failure metadata.
        result = incomplete(f'{type(exc).__name__}: {exc}')
        encoded = json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode()) > CAP:
        result = {'schema':'d1-response-crosscheck82-v1','status':'INCOMPLETE','failure':'failure metadata exceeds4MiB',
                  'completed_response_rows':len(reader.rows),'completed_parameter_pairs':len(reader.parameters)}
        encoded = json.dumps(result,allow_nan=False)+'\n'
    with args.output.open('x') as stream:
        stream.write(encoded)
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('status','counts','cpu_seconds','failure')},allow_nan=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
