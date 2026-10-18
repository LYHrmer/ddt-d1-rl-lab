"""Cold/pure29 tests. Real tensor checks require explicit isolated opt-in."""
from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools29 import anchor29 as a


@pytest.fixture
def np():
    return pytest.importorskip('numpy')


def fake_library(np):
    class Tensor(np.ndarray):
        requires_grad=False
        @property
        def device(self):return SimpleNamespace(type='cpu')
        def detach(self):return self
        def clone(self):return self.copy().view(Tensor)
        def cpu(self):return self
        def contiguous(self):return self
        def numpy(self):return np.asarray(self)
    def tensor(value):return np.asarray(value,dtype=np.float32).view(Tensor)
    state={key:tensor(np.zeros(shape,dtype=np.float32)) for key,shape in a.ACTOR_SHAPES.items()}
    # This is a cheap fake routing operation, deliberately not a dense NN.
    def linear(value,weight,bias):
        return tensor(np.broadcast_to(np.asarray(value)[:,:1],(len(value),weight.shape[0]))+np.asarray(bias))
    th=SimpleNamespace(float32=np.float32,from_numpy=tensor,tanh=lambda x:tensor(np.tanh(x)),
        cat=lambda xs,dim:tensor(np.concatenate(xs,axis=dim)),no_grad=contextlib.nullcontext)
    lib=SimpleNamespace(np=np,th=th,F=SimpleNamespace(linear=linear),as_numpy=lambda x:np.asarray(x).copy())
    return lib,state


def fake_buffer(np):
    class Buffer:
        full=True;generator_ready=False;n_envs=1;buffer_size=4096
        def __init__(self):
            self.observations={'actor':np.zeros((4096,1,627),np.float32),'velocity_target':np.zeros((4096,1,3),np.float32)}
            self.observations['actor'][:,0,0]=np.arange(4096,dtype=np.float32)/10000
            for key in ('actions','values','log_probs','advantages','returns'):
                setattr(self,key,np.zeros((4096,1,16) if key=='actions' else (4096,1),np.float32))
        def swap_and_flatten(self,value):
            return value.swapaxes(0,1).reshape((-1,*value.shape[2:])) if value.ndim>2 else value.swapaxes(0,1).reshape(-1,1)
        def to_torch(self,value):return value.copy()
        def _get_samples(self,indices):
            return SimpleNamespace(observations={key:value[indices] for key,value in self.observations.items()},rows=indices.copy())
    return Buffer()


def original_get(np):
    source=(ROOT/'.python-deps/stable_baselines3/common/buffers.py')
    if not source.is_file():
        package=importlib.util.find_spec('stable_baselines3')
        if package is None:pytest.skip('SB3 source absent in static-only CI')
        source=Path(next(iter(package.submodule_search_locations)))/'common/buffers.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DictRolloutBuffer')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get'))
    method.returns=None
    for arg in method.args.args:arg.annotation=None
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['get']


def test_cold_import_does_not_load_frameworks():
    code='import sys;from tools29 import anchor29,train_anchor29;assert not any(n.partition(".")[0] in {"numpy","torch","mujoco","gymnasium","stable_baselines3","onnxruntime"} for n in sys.modules)'
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True,text=True)


def test_teacher_snapshot_thirteen_detached_clones_and_exact_budget(np):
    lib,state=fake_library(np);counts=a.zero_teacher_counts29()
    teacher=a.FrozenActor29(SimpleNamespace(state_dict=lambda:state),lib,counts)
    assert len(teacher.weights)==13 and teacher.initial_sha==a.actor_digest29(state,lib)
    state['action_net.bias'][0]=7.
    assert teacher.current_sha()==teacher.initial_sha and teacher.weights['action_net.bias'][0]==0
    buffer=fake_buffer(np);cache=teacher.cache(buffer,a.INITIAL+4096)
    assert counts==dict(batch_attempts=1,batch_returns=1,rows_attempted=4096,rows_returned=4096,linear_attempts=6,linear_returns=6)
    assert cache['record']['actor_input']['shape']==[4096,627]
    assert all(not value.requires_grad for value in teacher.weights.values())
    with pytest.raises(ValueError,match='generation'):teacher.cache(buffer,a.INITIAL+4096)


def test_same_permutation_same_samples_and_rng_as_real_sb3_get(np):
    lib,state=fake_library(np);teacher=a.FrozenActor29(SimpleNamespace(state_dict=lambda:state),lib,a.zero_teacher_counts29())
    anchored=fake_buffer(np);plain=fake_buffer(np)
    plain.get=MethodType(original_get(np),plain)
    cache=teacher.cache(anchored,a.INITIAL+4096)
    for epoch in range(2):
        np.random.seed(2900+epoch);reference=[x.rows for x in plain.get(256)];next_original=np.random.randint(2**30)
        np.random.seed(2900+epoch);batches=list(a.anchored_minibatches29(anchored,cache,256,np));next_new=np.random.randint(2**30)
        assert next_original==next_new
        assert len(batches)==16
        for indices,(batch,mean,log_std) in zip(reference,batches):
            assert np.array_equal(indices,batch.rows)
            assert np.array_equal(mean,cache['mean'][indices]) and np.array_equal(log_std,cache['log_std'])
            assert np.array_equal(batch.observations['actor'],plain.observations['actor'][indices])
    assert len(cache['record']['epochs'])==2


@pytest.mark.parametrize('fault',['actor','mean','log_std','teacher','incomplete'])
def test_mutated_or_incomplete_cache_is_rejected(np,fault):
    lib,state=fake_library(np);teacher=a.FrozenActor29(SimpleNamespace(state_dict=lambda:state),lib,a.zero_teacher_counts29())
    buffer=fake_buffer(np);cache=teacher.cache(buffer,a.INITIAL+4096)
    if fault=='actor':buffer.observations['actor'][0,0,0]+=1
    elif fault=='mean':cache['mean'][0,0]+=1
    elif fault=='log_std':cache['log_std'][0]+=1
    elif fault=='teacher':teacher.weights['log_std'][0]+=1
    else:buffer.full=False
    with pytest.raises(ValueError):
        if fault=='teacher':teacher.cache(fake_buffer(np),a.INITIAL+8192)
        else:list(a.anchored_minibatches29(buffer,cache,256,np))


@pytest.mark.parametrize('fault',['bool','cap','native','extra'])
def test_real_worker_count_validator_keeps_five_typed_keys(fault):
    row=dict(policy_transitions=32768,native_steps=327680,pd_updates=163840,resets=1,forward_calls=32771)
    assert a.checked_worker_counts29([row])==row
    if fault=='bool':row['resets']=True
    elif fault=='cap':row.update(policy_transitions=32769,native_steps=327690,pd_updates=163845)
    elif fault=='native':row['native_steps']-=50
    else:row['extra']='untrusted'
    with pytest.raises(ValueError):a.checked_worker_counts29([row])


def test_ppo_control_is_direct_original_train_and_only_anchor_is_added():
    tree=ast.parse((ROOT/'tools29/ppo_anchor29.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    branches=[n for n in ast.walk(train) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.anchor_coef == 0.0']
    assert len(branches)==1 and 'result = super().train()' in ast.unparse(branches[0])
    source=(ROOT/'tools29/anchor29.py').read_text()
    assert 'self.policy.evaluate_actions(' in source and 'self.policy.action_dist.distribution' in source
    assert '+self.anchor_coef*anchor_loss)' in source
    assert 'self.policy.get_distribution(' not in source  # no duplicate current actor


def test_real_torch_synthetic_gradient_control_equivalence_and_save_load(tmp_path):
    """Opt-in CPU CI only; never env construction/learn or historical weights.

    Two4096-row synthetic policy batches, one4096-row teacher batch, at most
    48 training minibatches, three32-row comparisons and two1-row save/load
    predictions. This test is outside the registered production budget.
    """
    if os.environ.get('D1_RUN_ANCHOR29_NUMERIC_TESTS')!='1':
        pytest.skip('explicit isolated CPU tensor test not enabled')
    import numpy as np
    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.logger import configure
    from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
    from tools29.ppo_anchor29 import AnchoredVelocityPPO29
    th.set_num_threads(1)
    if th.get_num_interop_threads()!=1:th.set_num_interop_threads(1)
    def model(kind,coef):
        value=kind(env=None,aux_coef=0.,n_steps=4096,batch_size=256,n_epochs=1,learning_rate=1e-4,
            vf_coef=0.,ent_coef=0.,seed=2911,_init_setup_model=False,**({'anchor_coef':coef} if kind is AnchoredVelocityPPO29 else {}))
        value.observation_space=spaces.Dict({'actor':spaces.Box(-np.inf,np.inf,(627,),np.float32),
                                           'velocity_target':spaces.Box(-np.inf,np.inf,(3,),np.float32)})
        value.action_space=spaces.Box(-1.,1.,(16,),np.float32);value.n_envs=1;value._setup_model()
        value.set_logger(configure(folder=None,format_strings=[]))
        for param in value.policy.parameters():param.grad=th.zeros_like(param)
        value.policy.optimizer.step();value.policy.optimizer.zero_grad()
        value.num_timesteps=a.INITIAL+4096
        value._current_progress_remaining=1.
        return value
    def fill(value):
        rng=np.random.default_rng(2912)
        observations={'actor':rng.normal(0,.1,(4096,627)).astype(np.float32),
                      'velocity_target':np.zeros((4096,3),np.float32)}
        with th.no_grad():
            tensors={k:th.from_numpy(v) for k,v in observations.items()}
            actions,values,logs=value.policy(tensors,deterministic=False)
        buffer=value.rollout_buffer;buffer.reset()
        buffer.observations={k:v[:,None].copy() for k,v in observations.items()}
        buffer.actions=actions.numpy()[:,None].copy();buffer.values=values.numpy().reshape(4096,1)
        buffer.log_probs=logs.numpy().reshape(4096,1);buffer.returns=buffer.values.copy()
        buffer.advantages.fill(0.);buffer.full=True;buffer.pos=4096
        return observations
    control=model(AnchoredVelocityPPO29,0.);reference=model(AuxiliaryVelocityPPO,0.)
    reference.policy.load_state_dict(control.policy.state_dict())
    # CPU optimizer.load_state_dict may return the original step tensor and
    # dtype/device-preserving .to() may alias moment tensors. Isolate fixtures.
    reference.policy.optimizer.load_state_dict(copy.deepcopy(control.policy.optimizer.state_dict()))
    def optimizer_storages(value):
        return {item.untyped_storage().data_ptr() for row in value.state.values()
                for item in row.values() if isinstance(item,th.Tensor)}
    assert optimizer_storages(control.policy.optimizer).isdisjoint(optimizer_storages(reference.policy.optimizer))
    assert {p.untyped_storage().data_ptr() for p in control.policy.parameters()}.isdisjoint(
        {p.untyped_storage().data_ptr() for p in reference.policy.parameters()})
    control.configure_anchor29();fill(control)
    control.vf_coef=reference.vf_coef=.5
    control.rollout_buffer.advantages[:,0]=np.sin(np.arange(4096,dtype=np.float32))
    control.rollout_buffer.returns+=.1
    reference.rollout_buffer=copy.deepcopy(control.rollout_buffer)
    initial_control={k:v.clone() for k,v in control.policy.state_dict().items()}
    reference_adam_before=copy.deepcopy(reference.policy.optimizer.state_dict())
    np.random.seed(2913);control.train()
    for key,row in reference_adam_before['state'].items():
        assert all(th.equal(value,reference.policy.optimizer.state_dict()['state'][key][name]) for name,value in row.items())
    np.random.seed(2913);reference.train()
    assert all(th.equal(value,reference.policy.state_dict()[name]) for name,value in control.policy.state_dict().items())
    for key,row in control.policy.optimizer.state_dict()['state'].items():
        assert all(th.equal(value,reference.policy.optimizer.state_dict()['state'][key][name]) for name,value in row.items())
    assert any(not th.equal(value,initial_control[name]) for name,value in control.policy.state_dict().items())
    assert control.anchor_receipt29()['optimizer_updates']==16
    assert control.anchor_receipt29()['teacher']==a.zero_teacher_counts29()
    treatment=model(AnchoredVelocityPPO29,1.);treatment.configure_anchor29();treatment._anchor_poll29=lambda *args:True
    teacher=treatment._anchor_teacher29
    with th.no_grad():treatment.policy.action_net.bias.add_(.1)
    observations=fill(treatment)
    x={k:th.from_numpy(v[:32]) for k,v in observations.items()}
    before_state={k:v.clone() for k,v in treatment.policy.state_dict().items()}
    teacher_state={k:v.clone() for k,v in teacher.weights.items()}
    # Teacher cache is built once by train; compare initial/output KL on the same32 rows.
    with th.no_grad():base_mean=treatment.policy.get_distribution(x).distribution.loc.detach().clone()-.1
    before=float(a.gaussian_kl29(treatment.policy.get_distribution(x).distribution.loc,treatment.policy.log_std,
        base_mean,teacher.weights['log_std'],th))
    np.random.seed(2914);treatment.train()
    after=float(a.gaussian_kl29(treatment.policy.get_distribution(x).distribution.loc,treatment.policy.log_std,
        base_mean,teacher.weights['log_std'],th))
    assert after<before
    assert any(not th.equal(before_state[k],v) for k,v in treatment.policy.state_dict().items() if k.startswith('mlp_extractor.encoder'))
    assert any(not th.equal(before_state[k],v) for k,v in treatment.policy.state_dict().items() if k.startswith('mlp_extractor.policy_net'))
    assert all(th.equal(before_state[k],v) for k,v in treatment.policy.state_dict().items() if k.startswith(('mlp_extractor.value_net','value_net')))
    assert all(th.equal(teacher_state[k],v) and not v.requires_grad and v.grad is None for k,v in teacher.weights.items())
    treatment.save(str(tmp_path/'policy.zip'))
    restored=AuxiliaryVelocityPPO.load(tmp_path/'policy.zip',device='cpu')
    assert type(restored.anchor_coef) is float and restored.anchor_coef==1.
    assert np.array_equal(treatment.predict({k:v[:1] for k,v in observations.items()},deterministic=True)[0],
                          restored.predict({k:v[:1] for k,v in observations.items()},deterministic=True)[0])
    assert not any(name.startswith('_anchor_') for name in restored.__dict__)
