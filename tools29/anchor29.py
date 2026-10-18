"""Cold helpers and the explicit fixed-base Gaussian-KL intervention for29.

NumPy/Torch are supplied by an authorized runtime. Importing this module does
not load a network, tensor library, checkpoint, environment or physics engine.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

INITIAL, TRANSITIONS, SEED = 1310720, 32768, 183
ACTOR_SHAPES = {'log_std': (16,)}
for prefix, layouts in (
    ('mlp_extractor.encoder', ((627,128),(128,64),(64,3))),
    ('mlp_extractor.policy_net', ((60,128),(128,128))),
    ('action_net', ((128,16),)),
):
    for index, (incoming,outgoing) in enumerate(layouts):
        name = prefix if prefix == 'action_net' else prefix+'.'+str(index*2)
        ACTOR_SHAPES[name+'.weight'] = (outgoing,incoming)
        ACTOR_SHAPES[name+'.bias'] = (outgoing,)


def require(value,message):
    if not value:
        raise ValueError(message)


def digest_array29(array):
    view = memoryview(array)
    require(view.c_contiguous,'anchor29 arrays must be contiguous')
    return {'shape':list(view.shape),'format':view.format,'nbytes':view.nbytes,
            'sha256':hashlib.sha256(view.cast('B')).hexdigest()}


def zero_teacher_counts29():
    return dict.fromkeys(('batch_attempts','batch_returns','rows_attempted','rows_returned',
                          'linear_attempts','linear_returns'),0)


def actor_digest29(state,lib):
    require(set(ACTOR_SHAPES) <= set(state),'anchor29 actor state missing')
    digest = hashlib.sha256()
    for name,shape in sorted(ACTOR_SHAPES.items()):
        value=state[name]
        require(tuple(value.shape)==shape and value.dtype==lib.th.float32
                and value.device.type=='cpu','anchor29 actor tensor layout differs: '+name)
        array=lib.as_numpy(value)
        require(bool(lib.np.isfinite(array).all()),'anchor29 actor tensor nonfinite')
        digest.update(name.encode());digest.update(json.dumps(digest_array29(array),sort_keys=True).encode())
    return digest.hexdigest()


class FrozenActor29:
    """Thirteen detached clones; no Module, optimizer, RNG or second load."""
    def __init__(self,policy,lib,counts):
        state=policy.state_dict()
        self.lib,self.counts=lib,counts
        self.weights={name:state[name].detach().clone() for name in ACTOR_SHAPES}
        require(all(not value.requires_grad for value in self.weights.values()),'teacher must not require gradients')
        self.initial_sha=actor_digest29(self.weights,lib)
        require(self.initial_sha==actor_digest29(state,lib),'teacher snapshot changed loaded base')
        self.rollouts=[]

    def current_sha(self):
        return actor_digest29(self.weights,self.lib)

    def cache(self,buffer,generation):
        np,th=self.lib.np,self.lib.th
        require(buffer.full and not buffer.generator_ready and buffer.n_envs==1
                and buffer.buffer_size==4096,'teacher needs a fresh complete single-env rollout')
        require(type(generation) is int and generation==INITIAL+4096*(len(self.rollouts)+1),
                'teacher rollout generation reused or skipped')
        require(self.current_sha()==self.initial_sha,'fixed teacher mutated')
        actor=buffer.swap_and_flatten(buffer.observations['actor'])
        require(actor.shape==(4096,627) and actor.dtype==np.float32 and bool(np.isfinite(actor).all()),
                'teacher actual rollout actor differs')
        self.counts['batch_attempts']+=1;self.counts['rows_attempted']+=4096
        def linear(value,prefix):
            self.counts['linear_attempts']+=1
            result=self.lib.F.linear(value,self.weights[prefix+'.weight'],self.weights[prefix+'.bias'])
            self.counts['linear_returns']+=1
            return result
        with th.no_grad():
            x=th.from_numpy(actor)
            velocity=linear(th.tanh(linear(th.tanh(linear(x,'mlp_extractor.encoder.0')),
                                         'mlp_extractor.encoder.2')),'mlp_extractor.encoder.4')
            features=th.cat((x[:,:57],velocity),dim=1)
            latent=th.tanh(linear(th.tanh(linear(features,'mlp_extractor.policy_net.0')),
                                  'mlp_extractor.policy_net.2'))
            mean=self.lib.as_numpy(linear(latent,'action_net'))
        require(mean.shape==(4096,16) and mean.dtype==np.float32 and bool(np.isfinite(mean).all()),
                'teacher mean nonfinite or wrong shape')
        self.counts['batch_returns']+=1;self.counts['rows_returned']+=4096
        log_std=self.lib.as_numpy(self.weights['log_std'])
        record={'generation':generation,'rows':4096,'actor_input':digest_array29(actor),
                'teacher_mean':digest_array29(mean),'teacher_log_std':digest_array29(log_std),'epochs':[]}
        self.rollouts.append(record)
        return {'buffer':buffer,'generation':generation,'mean':mean,'log_std':log_std,'record':record}


def anchored_minibatches29(buffer,cache,batch_size,np):
    """Same permutation and _get_samples as SB3; index both arrays once."""
    require(buffer is cache['buffer'] and buffer.full and buffer.n_envs==1
            and buffer.buffer_size==4096 and batch_size==256,'anchor29 buffer/cache layout differs')
    # Identical position/order to DictRolloutBuffer.get, including each epoch.
    indices=np.random.permutation(buffer.buffer_size*buffer.n_envs)
    require(np.array_equal(np.sort(indices),np.arange(4096)),'anchor29 permutation incomplete')
    if not buffer.generator_ready:
        for key,obs in buffer.observations.items():
            buffer.observations[key]=buffer.swap_and_flatten(obs)
        for name in ('actions','values','log_probs','advantages','returns'):
            buffer.__dict__[name]=buffer.swap_and_flatten(buffer.__dict__[name])
        buffer.generator_ready=True
    require(digest_array29(buffer.observations['actor'])==cache['record']['actor_input'],
            'teacher cache actor input changed or stale')
    require(digest_array29(cache['mean'])==cache['record']['teacher_mean']
            and digest_array29(cache['log_std'])==cache['record']['teacher_log_std'],
            'teacher cached distributions changed')
    record={'epoch':len(cache['record']['epochs']),'permutation':digest_array29(indices),
            'minibatches_returned':0,'sample_rows_returned':0}
    cache['record']['epochs'].append(record)
    for start in range(0,4096,batch_size):
        selected=indices[start:start+batch_size]
        data=buffer._get_samples(selected)
        base_mean=buffer.to_torch(cache['mean'][selected])
        base_log_std=buffer.to_torch(cache['log_std'])
        record['minibatches_returned']+=1;record['sample_rows_returned']+=len(selected)
        yield data,base_mean,base_log_std


def gaussian_kl29(mean,log_std,base_mean,base_log_std,th):
    """Unclipped KL(base||current), sum16 then mean batch; teacher detached."""
    require(mean.ndim==2 and mean.shape[1]==16 and base_mean.shape==mean.shape
            and tuple(log_std.shape)==tuple(base_log_std.shape)==(16,), 'anchor29 Gaussian layout differs')
    require(not base_mean.requires_grad and not base_log_std.requires_grad,'teacher gradient forbidden')
    current=mean.to(th.float64);reference=base_mean.detach().to(th.float64)
    b=log_std.to(th.float64);a=base_log_std.detach().to(th.float64)
    terms=b-a+.5*th.expm1(2*(a-b))+.5*(reference-current).square()*th.exp(-2*b)
    value=terms.sum(dim=1).mean().to(mean.dtype)
    require(bool(th.isfinite(value)) and float(value.detach())>=-1e-10,'anchor29 KL nonfinite/negative')
    return value


def checked_worker_counts29(rows,transitions=TRANSITIONS):
    from tools25.low_memory_runtime25 import checked_worker_counts25
    return checked_worker_counts25(rows,transitions=transitions)


def initial_identity_valid29(initial,plan):
    try:
        fixed={'num_timesteps':INITIAL,'updates':1088,'optimizer_restored':True,
            'checkpoint_sha256':plan['base_checkpoint']['sha256'],
            'serialized_entries_sha256':plan['baseline_provenance']['serialized_entries_sha256'],
            'seed_before':181,'seed_after':SEED,'seed_changes_policy_or_optimizer':False,
            'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[1e-4,1e-4],
            'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,
            'rollout_layout':{'backend':'DummyVecEnv','envs':1,'n_steps':4096,'rollout_transitions':4096,
                'buffer_class':'DictRolloutBuffer','actor_shape':[4096,1,627],'velocity_target_shape':[4096,1,3],
                'actions_shape':[4096,1,16],'last_obs_cleared':True,'interprocess_rpc':False,'environment_subprocesses':0}}
        require(all(type(initial.get(k)) is type(v) and json.dumps(initial[k],sort_keys=True)==json.dumps(v,sort_keys=True)
                    for k,v in fixed.items()),'initial29 restoration/layout differs')
        keys=('policy_state_sha256','optimizer_state_sha256','policy_state_sha256_after_checks','optimizer_state_sha256_after_checks')
        require(all(type(initial.get(k)) is str and re.fullmatch('[0-9a-f]{64}',initial[k]) for k in keys)
                and initial[keys[0]]==initial[keys[2]] and initial[keys[1]]==initial[keys[3]],'initial29 policy/Adam changed')
        return True
    except (ValueError,KeyError,TypeError):
        return False


def valid_reward_state29(state,steps=None):
    try:
        require(type(state) is dict and set(state)=={'yaw_tracking_weight','steps','original_reward_sum',
            'training_reward_sum','extra_reward_sum','raw_wz_tracking_sum'},'reward29 schema differs')
        require(type(state['yaw_tracking_weight']) is float and state['yaw_tracking_weight']==2.
                and type(state['steps']) is int and 0<=state['steps']<=TRANSITIONS
                and (steps is None or state['steps']==steps),'reward29 weight/counter differs')
        require(all(type(state[k]) in (int,float) and math.isfinite(state[k]) for k in state if k not in ('steps',)),
                'reward29 nonfinite')
        require(state['extra_reward_sum']==0. and state['original_reward_sum']==state['training_reward_sum']
                and 0<=state['raw_wz_tracking_sum']<=state['steps'],'reward29 changed original reward')
        return True
    except (ValueError,KeyError,TypeError):
        return False


def valid_anchor_receipt29(value,coef,completed=False):
    try:
        require(type(coef) is float and coef in (0.,1.) and type(value) is dict,'anchor29 coefficient differs')
        require(set(value)=={'base_actor_sha256','frozen_actor_sha256_after_training','teacher','cache_rollouts',
            'train_calls','train_returns','optimizer_updates','anchor_loss_statistics'},
                'anchor29 receipt fields differ')
        digest=value['base_actor_sha256']
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest)
                and digest==value['frozen_actor_sha256_after_training'],'fixed base actor changed')
        counts=value['teacher'];require(set(counts)==set(zero_teacher_counts29())
            and all(type(x) is int and x>=0 for x in counts.values()),'anchor29 teacher counters untyped')
        rows=value['cache_rollouts'];require(type(rows) is list and len(rows)<=8,'anchor29 cache count exceeds cap')
        require(counts['batch_returns']==len(rows) and counts['batch_returns']<=counts['batch_attempts']<=8
                and counts['rows_attempted']==4096*counts['batch_attempts']
                and counts['rows_returned']==4096*counts['batch_returns']
                and 6*counts['batch_returns']<=counts['linear_returns']<=counts['linear_attempts']
                and counts['linear_attempts']<=6*counts['batch_attempts'],'anchor29 teacher counts differ')
        require(all(type(value[k]) is int and 0<=value[k]<=512 for k in ('train_calls','train_returns','optimizer_updates'))
                and value['train_returns']<=value['train_calls']<=8,'anchor29 train/update counters differ')
        stats=value['anchor_loss_statistics']
        require(type(stats) is dict and set(stats)=={'batches','mean','min','max'}
                and type(stats['batches']) is int and 0<=stats['batches']<=512,'anchor29 loss statistics differ')
        if stats['batches']:
            require(all(type(stats[k]) is float and math.isfinite(stats[k]) for k in ('mean','min','max'))
                    and stats['min']>=-1e-10 and stats['min']-1e-12<=stats['mean']<=stats['max']+1e-12,
                    'anchor29 loss statistics invalid')
        else:
            require(all(stats[k] is None for k in ('mean','min','max')),'anchor29 missing loss samples disguised')
        for index,row in enumerate(rows):
            require(row['generation']==INITIAL+4096*(index+1) and type(row['generation']) is int
                    and type(row['rows']) is int and row['rows']==4096,'anchor29 stale cache generation')
            for key,shape in (('actor_input',[4096,627]),('teacher_mean',[4096,16]),('teacher_log_std',[16])):
                require(row[key]['shape']==shape and row[key]['format']=='f' and type(row[key]['nbytes']) is int
                        and row[key]['nbytes']==4*math.prod(shape)
                        and re.fullmatch('[0-9a-f]{64}',row[key]['sha256']),'anchor29 cache identity differs')
            require(type(row['epochs']) is list and len(row['epochs'])<=4,'anchor29 epoch cap exceeded')
            for epoch_index,epoch in enumerate(row['epochs']):
                require(epoch['epoch']==epoch_index and type(epoch['epoch']) is int
                    and type(epoch['minibatches_returned']) is int and 1<=epoch['minibatches_returned']<=16
                    and type(epoch['sample_rows_returned']) is int and epoch['sample_rows_returned']==256*epoch['minibatches_returned'],
                    'anchor29 minibatch count differs')
                perm=epoch['permutation']
                require(perm['shape']==[4096] and perm['format'] in ('l','q') and perm['nbytes']==32768
                        and re.fullmatch('[0-9a-f]{64}',perm['sha256']),'anchor29 permutation identity differs')
        if coef==0.:
            require(counts==zero_teacher_counts29() and rows==[] and stats['batches']==0,'control29 executed teacher')
        if completed:
            require(value['train_calls']==value['train_returns']==8 and 8<=value['optimizer_updates']<=512,
                    'anchor29 complete PPO/update prefix missing')
        if completed and coef==1.:
            require(counts==dict(batch_attempts=8,batch_returns=8,rows_attempted=32768,rows_returned=32768,
                                linear_attempts=48,linear_returns=48)
                    and all(1<=len(row['epochs'])<=4 for row in rows)
                    and stats['batches']>=value['optimizer_updates'],'anchor29 complete teacher prefix missing')
        return True
    except (ValueError,KeyError,TypeError):
        return False


def anchor_receipt29(model):
    teacher=model._anchor_teacher29
    losses=model._anchor_losses29
    return {'base_actor_sha256':model._anchor_base_sha29,
        'frozen_actor_sha256_after_training':model._anchor_base_sha29 if teacher is None else teacher.current_sha(),
        'teacher':dict(model._anchor_counts29),'cache_rollouts':[] if teacher is None else copy.deepcopy(teacher.rollouts),
        'train_calls':model._anchor_train_calls29,'train_returns':model._anchor_train_returns29,
        'optimizer_updates':model._anchor_optimizer_updates29,
        'anchor_loss_statistics':{'batches':len(losses),'mean':None if not losses else float(math.fsum(losses)/len(losses)),
            'min':None if not losses else float(min(losses)),'max':None if not losses else float(max(losses))}}


def adam_steps29(optimizer):
    states=optimizer.state_dict()['state']
    require(bool(states),'anchor29 trained Adam state absent')
    steps=[]
    for value in states.values():
        raw=value['step']
        number=raw.item() if hasattr(raw,'item') else raw
        require(type(number) in (int,float) and math.isfinite(number) and number>=0 and int(number)==number,
                'anchor29 Adam step invalid')
        steps.append(int(number))
    require(len(set(steps))==1,'anchor29 Adam parameter steps disagree')
    return steps[0]


def poll_anchor29(model,phase,force=False):
    require(callable(model._anchor_poll29) and model._anchor_poll29(phase,force),
            'anchor29 resource/wall stop: '+phase)


def train_with_anchor29(self,lib):
    np,th,F,explained_variance=lib.np,lib.th,lib.F,lib.explained_variance
    require(type(self.anchor_coef) is float and self.anchor_coef==1., "anchor29 training requires beta1")
    poll_anchor29(self,"before_teacher29",True)
    cache=self._anchor_teacher29.cache(self.rollout_buffer,self.num_timesteps)
    self._anchor_cache29=cache
    poll_anchor29(self,"after_teacher29",True)
    # Adapted from stable-baselines3 2.7.0 PPO.train (MIT license,
    # https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/ppo/ppo.py).
    # Copyright (c) 2019 Antonin Raffin; full permission notice is retained
    # in licenses/stable-baselines3-MIT.txt and THIRD_PARTY_NOTICES.md.
    # Continuous Box only: the upstream discrete-action cast is inapplicable.
    # Original auxiliary MSE is retained. The new29 intervention adds fixed-base KL.
    # Teacher cache / shared indices / extra KL are the three explicit seams.
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)
    if self.clip_range_vf is not None:
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
    entropy_losses, pg_losses, value_losses, auxiliary_losses, clip_fractions = [], [], [], [], []
    continue_training = True
    for epoch in range(self.n_epochs):
        approx_kl_divs = []
        for rollout_data,base_mean,base_log_std in anchored_minibatches29(self.rollout_buffer,cache,self.batch_size,np):
            poll_anchor29(self,"anchor29_minibatch")
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, rollout_data.actions)
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            ratio = th.exp(log_prob-rollout_data.old_log_prob)
            policy_loss_1 = advantages*ratio
            policy_loss_2 = advantages*th.clamp(ratio, 1-clip_range, 1+clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            pg_losses.append(policy_loss.item())
            clip_fractions.append(th.mean((th.abs(ratio-1) > clip_range).float()).item())
            if self.clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values+th.clamp(
                    values-rollout_data.old_values, -clip_range_vf, clip_range_vf)
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())
            entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
            entropy_losses.append(entropy_loss.item())
            auxiliary_loss = self.policy.auxiliary_velocity_loss(rollout_data.observations)
            auxiliary_losses.append(auxiliary_loss.item())
            current_distribution=self.policy.action_dist.distribution
            anchor_loss=gaussian_kl29(current_distribution.loc,self.policy.log_std,base_mean,base_log_std,th)
            self._anchor_losses29.append(float(anchor_loss.detach().cpu().item()))
            loss = (policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                    + self.aux_coef*auxiliary_loss+self.anchor_coef*anchor_loss)
            require(bool(th.isfinite(loss)), "nonfinite PPO or auxiliary loss")
            with th.no_grad():
                log_ratio = log_prob-rollout_data.old_log_prob
                approx_kl_div = th.mean((th.exp(log_ratio)-1)-log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)
            if self.target_kl is not None and approx_kl_div > 1.5*self.target_kl:
                continue_training = False
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                break
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += 1
        if not continue_training:
            break
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
    for name, values in (("entropy_loss", entropy_losses), ("policy_gradient_loss", pg_losses),
                         ("value_loss", value_losses), ("auxiliary_velocity_mse", auxiliary_losses),
                         ("approx_kl", approx_kl_divs), ("clip_fraction", clip_fractions)):
        self.logger.record("train/"+name, np.mean(values))
    self.logger.record("train/aux_coef", self.aux_coef)
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
        self.logger.record("train/clip_range_vf", clip_range_vf)
    self.logger.record("train/fixed_base_kl29",self._anchor_losses29[-1])

