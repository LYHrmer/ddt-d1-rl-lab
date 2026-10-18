"""Pure synthetic objects + installed SB3 AST; no framework/NN/physics imports."""
from __future__ import annotations

import ast
import copy
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools25 import low_memory_runtime25 as r

HYPER = {'aux_coef':0.,'device':'cpu','cpu_threads':1,'learning_rate':1e-4,'n_steps':4096,
    'batch_size':256,'n_epochs':4,'gamma':.99,'gae_lambda':.95,'clip_range':.2,'ent_coef':0.,
    'vf_coef':.5,'max_grad_norm':.5,'log_std_init':-1.,'target_kl':.005}


class Optimizer:
    def __init__(self):
        self.param_groups = [{'lr':3e-4,'params':[0],'eps':1e-5}]
        self.momentum = {'step':1024,'exp_avg':[.1,.2],'exp_avg_sq':[.3,.4]}
    def state_dict(self):
        return {'param_groups':copy.deepcopy(self.param_groups),'state':{0:copy.deepcopy(self.momentum)}}


class Array:
    def __init__(self, shape, dtype='float32'):
        self.shape, self.dtype = shape, dtype


class Buffer:
    def __init__(self):
        self.buffer_size,self.n_envs,self.pos,self.full,self.generator_ready = 4096,1,0,False,False
        self.observations = {'actor':Array((4096,1,627)),'velocity_target':Array((4096,1,3))}
        self.actions = Array((4096,1,16))
        for key in ('rewards','returns','episode_starts','values','log_probs','advantages'):
            setattr(self,key,Array((4096,1)))


@pytest.fixture
def warm(monkeypatch,tmp_path):
    mode = [None]
    calls = []
    class Algorithm:
        def __init__(self, vec):
            self.__dict__.update({k:v for k,v in HYPER.items() if k not in ('device','cpu_threads','log_std_init','clip_range')})
            self.learning_rate,self.target_kl,self.lr_schedule = 3e-4,None,lambda progress:3e-4
            self.n_envs,self.seed,self.device,self.num_timesteps,self._n_updates = 1,181,'cpu',1048576,1024
            self.clip_range,self.normalize_advantage,self.clip_range_vf,self.use_sde = lambda p:.2,True,None,False
            self.policy = SimpleNamespace(optimizer=Optimizer())
            self.policy_marker,self.vec,self._last_obs,self.rollout_buffer = 'a'*64,vec,None,Buffer()
        @classmethod
        def load(cls,path,**kwargs):
            calls.append((path,kwargs))
            assert kwargs == {'env':vec,'device':'cpu','seed':181,'force_reset':True,'n_steps':4096}
            model = cls(kwargs['env'])
            if mode[0] == 'envs':model.n_envs = 4
            elif mode[0] == 'n_steps':model.n_steps = 1024
            elif mode[0] == 'last_obs':model._last_obs = 'old four-env observation'
            elif mode[0] == 'buffer_shape':model.rollout_buffer.observations['actor'] = Array((1024,4,627))
            elif mode[0] == 'buffer_dtype':model.rollout_buffer.observations['velocity_target'].dtype = 'float64'
            elif mode[0] == 'wrong_buffer':model.rollout_buffer = SimpleNamespace(**model.rollout_buffer.__dict__)
            elif mode[0] == 'filled_buffer':model.rollout_buffer.pos = 1
            elif mode[0] == 'updates':model._n_updates = 1023
            elif mode[0] == 'timesteps':model.num_timesteps = 1048577
            elif mode[0] == 'policy':model.policy_marker = 'b'*64
            elif mode[0] == 'adam':model.policy.optimizer.momentum['step'] = 1023
            elif mode[0] == 'baseline_lr':model.policy.optimizer.param_groups[0]['lr'] = 1e-4
            return model
        def get_env(self):return self.vec
        def _setup_lr_schedule(self):
            if mode[0] != 'schedule':self.lr_schedule = lambda progress:self.learning_rate
            if mode[0] == 'momentum_lr':self.policy.optimizer.momentum['exp_avg'] = [.9,.9]
            elif mode[0] == 'policy_lr':self.policy_marker = 'c'*64
        def __setattr__(self,key,value):
            object.__setattr__(self,key,value)
            if key == 'target_kl' and value == .005 and mode[0] == 'momentum_kl':
                self.policy.optimizer.momentum['step'] += 1
        def set_logger(self, logger):self.logger = logger
    contract = SimpleNamespace(INITIAL=1048576,TRANSITIONS=262144,HYPERPARAMETERS=HYPER,
        BASE_INITIAL_STATE={'num_timesteps':1048576,'policy_state_sha256':'a'*64,
            'optimizer_state_sha256':r.optimizer_sha(Optimizer()),'updates':1024,'optimizer_restored':True})
    monkeypatch.setattr(r,'contract25',lambda:contract)
    monkeypatch.setattr(r,'loading_libraries25',lambda:(Algorithm,Buffer,lambda *args:('synthetic logger',args)))
    runtime = r.LowMemoryRuntime25()
    runtime.policy_sha = lambda model:model.policy_marker
    vec = SimpleNamespace(num_envs=1)
    runtime.vec = vec
    plan = {'envs':1,'seed':181,'base_checkpoint':{'path':'original_complete_control.zip','trained_transitions':1048576}}
    return SimpleNamespace(mode=mode,calls=calls,runtime=runtime,vec=vec,plan=plan,contract=contract,output=tmp_path)


@pytest.mark.parametrize('fault',[None,'envs','n_steps','last_obs','buffer_shape','buffer_dtype','wrong_buffer','filled_buffer',
    'updates','timesteps','policy','adam','baseline_lr','schedule','momentum_lr','policy_lr','momentum_kl'])
def test_new_loader_exact_original_policy_adam_fresh_layout_and_lr_kl_proofs(warm,fault):
    warm.mode[0] = fault
    if fault:
        with pytest.raises(ValueError):warm.runtime.create_model(warm.vec,warm.plan,HYPER,warm.output)
    else:
        model = warm.runtime.create_model(warm.vec,warm.plan,HYPER,warm.output)
        proof = warm.runtime.initial_identity
        assert r.initial_identity_valid25(proof)
        assert model.n_steps == 4096 and model.n_envs == 1 and model._last_obs is None
        assert model.learning_rate == 1e-4 and model.target_kl == .005 and model._n_updates == 1024
        assert model.policy.optimizer.momentum == Optimizer().momentum
        assert proof['rollout_layout']['actor_shape'] == [4096,1,627]
        assert [row['phase'] for row in warm.runtime.memory_samples25] == ['before_model_load','after_model_load']
    assert len(warm.calls) == 1


@pytest.mark.parametrize('field',['policy_state_sha256','optimizer_state_sha256','optimizer_momentum_preserved',
    'optimizer_without_lr_sha256_after','target_kl_after','optimizer_state_sha256_after_target_kl','rollout_layout'])
def test_initial_receipt_rejects_claimed_identity_or_transition_without_proof(warm,field):
    warm.runtime.create_model(warm.vec,warm.plan,HYPER,warm.output)
    value = copy.deepcopy(warm.runtime.initial_identity)
    value[field] = None
    assert r.initial_identity_valid25(value) is False


def test_layout_receipt_does_not_accept_bool_as_one_environment(warm):
    warm.runtime.create_model(warm.vec,warm.plan,HYPER,warm.output)
    value = copy.deepcopy(warm.runtime.initial_identity)
    value['rollout_layout']['envs'] = True
    assert not r.initial_identity_valid25(value)


def counters(ticks=0,resets=0):
    return {'policy_transitions':ticks,'native_steps':10*ticks,'pd_updates':5*ticks,'resets':resets,'forward_calls':ticks+2*resets}


def delays(ticks=0,resets=0):
    return {'delay_counts':{'0':resets,'1':0,'2':0,'3':0},'last_selection':None if not resets else {
        'mode':'zero','seed':181,'rank':0,'episode_index':resets-1,'stream':18018,'selected_delay_ticks':0,'selection':'zero'}}


class FakeDummy:
    """A direct env owner; no framework, subprocess, queue, pipe or NN."""
    def __init__(self,env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.ticks,self.resets,self.sent = 0,0,[]
    def step_async(self,actions):self.actions = actions
    def step_wait(self):
        self.ticks += 1
        return 'step result'
    def reset(self):
        self.resets += 1
        return 'reset result'
    def env_method(self,name,indices=None):
        self.sent.append(name)
        return [counters(self.ticks,self.resets) if name=='counters' else delays(self.ticks,self.resets)]
    def close(self):
        for env in self.envs:env.close()


@pytest.fixture
def direct(monkeypatch):
    env = SimpleNamespace(close=lambda:None)
    cls = r.checked_dummy_class25(FakeDummy)
    vec = cls([lambda:env],mode='zero',seed=181,max_transitions=262144)
    runtime = r.LowMemoryRuntime25()
    runtime.vec = vec
    monkeypatch.setattr(r,'bounded_call',lambda function,**kwargs:function())
    return runtime,vec


def test_actual_direct_wrapper_preserves_ordered_steps_counters_delay_close(direct):
    runtime,vec = direct
    assert vec.reset() == 'reset result'
    vec.step_async('action')
    assert vec.step_wait() == 'step result'
    assert runtime.counters(vec) == [counters(1,1)]
    assert runtime.delay_state(vec) == [delays(1,1)]
    snapshot = runtime.snapshot()
    assert snapshot['backend'] == 'DummyVecEnv' and snapshot['interprocess_rpc'] is False
    assert snapshot['environment_subprocesses'] == 0 and snapshot['metadata_validated'] == 2
    assert not any('rpc' in key for key in snapshot if key != 'interprocess_rpc')
    assert runtime.close() == (True,None) and runtime.snapshot()['closed'] is True
    assert runtime.snapshot()['process_memory_samples'][-1]['phase'] == 'after_close'


@pytest.mark.parametrize('fault',['pending','stale_delays','wrong_row','wrong_type','regression','delay_rank','delay_histogram','native','closed'])
def test_metadata_is_not_read_from_pending_dirty_stale_or_invalid_direct_source(direct,monkeypatch,fault):
    runtime,vec = direct
    vec.reset()
    runtime.counters(vec)
    before = len(vec.sent)
    operation = lambda:runtime.counters(vec)
    if fault == 'pending':vec.step_async('action')
    elif fault == 'stale_delays':
        vec.step_async('action');vec.step_wait()
        operation = lambda:runtime.delay_state(vec)
    elif fault == 'closed':runtime.close()
    else:
        original = FakeDummy.env_method
        def altered(self,name,indices=None):
            value = original(self,name,indices)
            if fault == 'wrong_row':return value*4
            if fault == 'wrong_type':return [('observation',.1,False,{})]
            if fault == 'regression':value[0]['resets'] = 0
            if fault == 'native':value[0]['native_steps'] = 11
            if fault in ('delay_rank','delay_histogram'):
                value = [delays(0,1)]
                if fault == 'delay_rank':value[0]['last_selection']['rank'] = 1
                else:value[0]['delay_counts']['0'] = 2
            return value
        monkeypatch.setattr(FakeDummy,'env_method',altered)
        if fault.startswith('delay_'):operation = lambda:runtime.delay_state(vec)
    with pytest.raises((ValueError,RuntimeError)):operation()
    assert runtime.snapshot()['dirty'] is True
    if fault in ('pending','closed'):assert len(vec.sent) == before
    prior = len(vec.sent)
    with pytest.raises(RuntimeError):runtime.counters(vec)
    assert len(vec.sent) == prior


def test_close_failure_never_claims_owned_subprocess_killed(direct):
    runtime,vec = direct
    def fail():raise OSError('synthetic env close failure')
    vec.envs[0].close = fail
    ok,error = runtime.close()
    assert ok is False and 'synthetic env close failure' in error
    assert runtime.snapshot()['closed'] is False and runtime.snapshot()['dirty'] is True


@pytest.mark.parametrize('bad',[None,'missing','unit','negative','malformed'])
def test_memory_boundary_uses_own_rss_pss_or_explicit_unknown(tmp_path,bad):
    path = tmp_path/'smaps_rollup'
    text = '000-fff [rollup]\nRss: 123 kB\nPss: 91 kB\n'
    if bad == 'unit':text = text.replace('kB','MB')
    elif bad == 'negative':text = text.replace('91','-1')
    elif bad == 'malformed':text = 'no RSS/PSS\n'
    if bad != 'missing':path.write_text(text)
    value = r.self_memory25(path)
    if bad is None:
        assert value['rss_bytes'] == 123*1024 and value['pss_bytes'] == 91*1024 and value['error'] is None
    else:assert value['rss_bytes'] is value['pss_bytes'] is None and value['error']


@pytest.mark.parametrize('failure',[False,True])
def test_create_vec_real_wrapper_wiring_owns_one_rank0_and_retains_partial_handle(monkeypatch,tmp_path,failure):
    log = []
    class Wrapper:
        def __init__(self,env,*args):self.env = env;log.append((type(self).__name__,args))
        @property
        def unwrapped(self):return self.env.unwrapped
        def step(self,action):return self.env.step(action)
        def close(self):return self.env.close()
    class Base:
        def __init__(self,**kwargs):
            self.unwrapped,self.kwargs = self,kwargs
            self._training_schedule,self._episode_steps = {'segments':[(0,1000,(0.,0.,0.))]},0
            log.append(('base',kwargs))
        def step(self,action):return 'actor',.7,True,False,{'unchanged':'info'}
        def get_delay_state18(self):return 'actual base cached delay'
        def close(self):log.append(('base closed',))
    class CountedMonitor(Wrapper):pass
    class VelocityLabelWrapper(Wrapper):pass
    gym = types.ModuleType('gymnasium');gym.Wrapper = Wrapper
    monkeypatch.setitem(sys.modules,'gymnasium',gym)
    package = types.ModuleType('d1_rl');package.__path__ = []
    monkeypatch.setitem(sys.modules,'d1_rl',package)
    for name,values in {'d1_rl.delay_env18':{'D1Delay18Env':Base},
        'd1_rl.velocity_labels15':{'VelocityLabelWrapper':VelocityLabelWrapper},
        'train_d1_ppo09':{'CountedMonitor':CountedMonitor}}.items():
        module = types.ModuleType(name);module.__dict__.update(values)
        monkeypatch.setitem(sys.modules,name,module)
    class Constructor(FakeDummy):
        def __init__(self,env_fns):
            super().__init__(env_fns)
            if failure:raise OSError('synthetic post-factory construction failure')
    original = r.checked_dummy_class25
    monkeypatch.setattr(r,'checked_dummy_class25',lambda:original(Constructor))
    monkeypatch.setattr(r,'contract25',lambda:SimpleNamespace(TRANSITIONS=262144))
    monkeypatch.setattr(r.sys,'path',list(sys.path))
    runtime = r.LowMemoryRuntime25()
    plan = {'envs':1,'seed':181,'config':'training/configs/pilot03d.json'}
    if failure:
        with pytest.raises(OSError):runtime.create_vec(plan,262144,tmp_path,ROOT,'zero')
        assert runtime.vec is not None and runtime.snapshot()['dirty'] is True
        assert runtime.close()[0] is False
    else:
        vec = runtime.create_vec(plan,262144,tmp_path,ROOT,'uniform_0_3')
        wrapped = vec.envs[0]
        assert wrapped.step('action') == ('actor',.7,True,False,{'unchanged':'info','training_phase':'stationary_initial'})
        assert wrapped.get_delay_state18() == 'actual base cached delay'
        assert log[:4] == [('base',{'training_delay_mode':'uniform_0_3','config':ROOT/plan['config'],
            'arm':'B','seed':181,'rank':0,'max_transitions':262144}),
            ('CountedMonitor',(str(tmp_path/'env_00.monitor.csv'),)),('VelocityLabelWrapper',()),('PhaseWrapper',())]
        assert runtime.vec is vec and vec.num_envs == 1


@pytest.mark.parametrize('fault',[None,'4rows','bool','float','cap','native','pd','missing'])
def test_one_environment_partial_counter_budget_and_exact_json_schema(fault):
    rows = [counters(23,1)]
    if fault == '4rows':rows *= 4
    elif fault == 'bool':rows[0]['resets'] = True
    elif fault == 'float':rows[0]['resets'] = 1.
    elif fault == 'cap':rows[0] = counters(262145,1)
    elif fault == 'native':rows[0]['native_steps'] = 231
    elif fault == 'pd':rows[0]['pd_updates'] = 120
    elif fault == 'missing':rows[0].pop('resets')
    if fault:
        with pytest.raises(ValueError):r.checked_worker_counts25(rows,262144)
    else:assert r.checked_worker_counts25(rows,262144) == counters(23,1)


def installed_method(name):
    """Execute one real installed SB3 control flow with only synthetic stand-ins."""
    path = ROOT/'.python-deps/stable_baselines3/common/base_class.py'
    if not path.exists():
        spec = importlib.util.find_spec('stable_baselines3')
        if spec is None or spec.origin is None:
            pytest.skip('exact SB3 source is not installed in static CI')
        package = Path(spec.origin).parent
        version = package/'version.txt'
        if not version.exists() or version.read_text().strip() != '2.7.0':
            pytest.skip('installed SB3 control-flow check requires exact version 2.7.0')
        path = package/'common/base_class.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='BaseAlgorithm')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name))
    method.decorator_list,method.returns = [],None
    for node in ast.walk(method):
        if isinstance(node,ast.arg):node.annotation = None
    return method,path


@pytest.mark.parametrize('force_reset',[True,False])
def test_real_installed_sb3_load_rebuilds_one_buffer_before_restoring_original_adam(force_reset):
    method,path = installed_method('load')
    events = []
    original_policy,original_adam = object(),object()
    data = {'policy_class':'saved-policy-class','verbose':0,'observation_space':'saved-627/3-Dict',
        'action_space':'saved-16Box','n_envs':4,'n_steps':1024,'_last_obs':'old obs for 4 envs','use_sde':False}
    class Algorithm:
        def __init__(self,policy,env,device,_init_setup_model):
            assert policy == 'saved-policy-class' and _init_setup_model is False
        @staticmethod
        def _wrap_env(env,verbose):return env
        def _setup_model(self):events.append(('setup',self.n_envs,self.n_steps));self.rollout_buffer = (self.n_steps,self.n_envs)
        def set_parameters(self,params,exact_match,device):
            assert exact_match is True and self.rollout_buffer == (4096,1)
            events.append(('restore',params['policy'],params['policy.optimizer']))
    namespace = {'load_from_zip_file':lambda *args,**kwargs:(copy.deepcopy(data),{'policy':original_policy,'policy.optimizer':original_adam},None),
        '_convert_space':lambda space:space,'check_for_correct_spaces':lambda *args:None}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)  # noqa: S102 -- reviewed installed control flow, synthetic globals only
    model = namespace['load'](Algorithm,'original_complete_control.zip',env=SimpleNamespace(num_envs=1),
        device='cpu',seed=181,force_reset=force_reset,n_steps=4096)
    assert events == [('setup',1,4096),('restore',original_policy,original_adam)]
    assert model._last_obs is None if force_reset else model._last_obs == 'old obs for 4 envs'


def test_real_installed_set_env_rejects_changing_four_environment_loaded_model_to_one():
    method,path = installed_method('set_env')
    namespace = {}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)  # noqa: S102 -- reviewed installed guard, no library import
    model = SimpleNamespace(n_envs=4,verbose=0,_wrap_env=lambda env,verbose:env)
    with pytest.raises(AssertionError):namespace['set_env'](model,SimpleNamespace(num_envs=1))


def test_real_installed_first_learn_reset_replaces_old_four_env_starts_without_zeroing_timesteps():
    method,path = installed_method('_setup_learn')
    namespace = {'time':SimpleNamespace(time_ns=lambda:123),
        'np':SimpleNamespace(ones=lambda shape,dtype:Array(shape,'bool'))}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)  # noqa: S102 -- real reset/offset flow, synthetic env/arrays
    resets = []
    def reset():
        resets.append(True)
        return {'actor':Array((1,627)),'velocity_target':Array((1,3))}
    model = SimpleNamespace(ep_info_buffer=[],ep_success_buffer=[],action_noise=None,num_timesteps=1048576,
        _last_obs=None,_last_episode_starts=Array((4,),'bool'),_vec_normalize_env=None,_custom_logger=True,
        env=SimpleNamespace(num_envs=1,reset=reset),_init_callback=lambda cb,progress_bar:cb)
    total,callback = namespace['_setup_learn'](model,262144,callback='synthetic callback',reset_num_timesteps=False)
    assert total == 1310720 and model.num_timesteps == model._num_timesteps_at_start == 1048576
    assert resets == [True] and model._last_episode_starts.shape == (1,)
    assert model._last_obs['actor'].shape == (1,627) and model._last_obs['velocity_target'].shape == (1,3)
    assert callback == 'synthetic callback'


def test_import_is_cold_and_source_never_constructs_subproc_or_calls_old_four_env_loader():
    code = ('import sys;from tools25.low_memory_runtime25 import LowMemoryRuntime25;'
        'r=LowMemoryRuntime25();assert r.vec is None;'
        'assert not any(k in sys.modules for k in ("torch","numpy","stable_baselines3","mujoco","gymnasium"));'
        'assert r.snapshot()["environment_subprocesses"]==0')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    source = (ROOT/'tools25/low_memory_runtime25.py').read_text()
    assert 'SubprocVecEnv' not in source and 'set_env(' not in source and 'validate_loaded_model' not in source
    assert 'force_reset=True,n_steps=4096' in source
