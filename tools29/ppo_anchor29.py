"""Runtime-only stable SB3 algorithm identity for the single29 intervention.

The cold parent/trainer never imports this module before explicit execution.
Policy/optimizer tensor names and the deployed network remain those of15.
"""
from types import SimpleNamespace

import numpy as np
import torch as th
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
from tools29.anchor29 import (
    FrozenActor29,
    actor_digest29,
    adam_steps29,
    anchor_receipt29,
    require,
    train_with_anchor29,
    zero_teacher_counts29,
)


class AnchoredVelocityPPO29(AuxiliaryVelocityPPO):
    def __init__(self,*args,anchor_coef=0.,**kwargs):
        require(type(anchor_coef) is float and anchor_coef in (0.,1.),'anchor29 supports only beta0/1')
        self.anchor_coef=anchor_coef
        self._anchor_lib29=SimpleNamespace(np=np,th=th,F=F,explained_variance=explained_variance,
            as_numpy=lambda value:value.detach().cpu().contiguous().numpy().copy())
        self._anchor_teacher29=self._anchor_cache29=self._anchor_poll29=None
        self._anchor_counts29=zero_teacher_counts29()
        self._anchor_losses29=[]
        self._anchor_base_sha29=None
        self._anchor_train_calls29=self._anchor_train_returns29=self._anchor_optimizer_updates29=0
        super().__init__(*args,**kwargs)

    def configure_anchor29(self):
        require(type(self.anchor_coef) is float and self.anchor_coef in (0.,1.)
                and self.aux_coef==0. and self.n_steps==4096 and self.n_envs==1,'anchor29 setup differs')
        require(self._anchor_train_calls29==0 and self._anchor_base_sha29 is None,
                'anchor29 snapshot cannot be reset or resumed')
        self._anchor_base_sha29=actor_digest29(self.policy.state_dict(),self._anchor_lib29)
        if self.anchor_coef==1.:
            self._anchor_teacher29=FrozenActor29(self.policy,self._anchor_lib29,self._anchor_counts29)
            require(self._anchor_teacher29.initial_sha==self._anchor_base_sha29,'anchor29 wrong fixed base')

    def train(self):
        require(self._anchor_base_sha29 is not None,'anchor29 base must be bound before learn')
        self._anchor_train_calls29+=1
        before=adam_steps29(self.policy.optimizer)
        try:
            if self.anchor_coef==0.:
                # Exact original math, shuffle, MSE and optimizer path.
                result=super().train()
            else:
                require(self.anchor_coef==1. and self._anchor_teacher29 is not None,'anchor29 teacher missing')
                result=train_with_anchor29(self,self._anchor_lib29)
            self._anchor_train_returns29+=1
            return result
        finally:
            self._anchor_cache29=None
            after=adam_steps29(self.policy.optimizer)
            require(after>=before,'anchor29 Adam steps went backwards')
            self._anchor_optimizer_updates29+=after-before

    def anchor_receipt29(self):
        return anchor_receipt29(self)

    def _excluded_save_params(self):
        # anchor_coef is deliberately retained in ordinary SB3 data; snapshots,
        # caches and audit callables must never be cloudpickled into checkpoints.
        return super()._excluded_save_params()+[name for name in self.__dict__ if name.startswith('_anchor_')]
