#!/usr/bin/env python3
"""Bounded, random-initialized pure D1 PPO; exclusive run directories."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

import gymnasium
import mujoco
import numpy as np
import psutil
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from d1_rl.factory03 import environment_class, environment_sources

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(path)

class CountedMonitor(Monitor):
    def __init__(self, env, filename):
        super().__init__(env,filename=filename)
        self.counter_path=Path(filename+".counts.json")
        self.publish_counts("created")

    def publish_counts(self,status):
        atomic_json(self.counter_path,{"status":status,"pid":os.getpid(),"counts":self.env.counters()})

    def step(self,action):
        try:
            result=super().step(action)
            if self.env.counters()["policy_transitions"]%128==0: self.publish_counts("running")
            return result
        except BaseException:
            self.publish_counts("interrupted_or_failed")
            raise

    def close(self):
        self.publish_counts("closed")
        super().close()

def make_env(config,arm,seed,rank,limit,monitor_path):
    def factory():
        return CountedMonitor(environment_class(config)(config=config,arm=arm,seed=seed,rank=rank,max_transitions=limit),filename=str(monitor_path))
    return factory

class Ledger(BaseCallback):
    def __init__(self, output, wall_limit):
        super().__init__()
        self.output=output; self.wall_limit=wall_limit
        self.started=time.monotonic(); self.reason=None
        self.terms=defaultdict(float); self.term_n=0
        self.max_rss=0; self.max_uss=0; self.min_available=float("inf")
        self.rollouts=0

    def _on_step(self):
        for info in self.locals.get("infos",[]):
            terms=info.get("reward_terms",{}).get("contributions_per_tick",{})
            for key,value in terms.items(): self.terms[key]+=float(value)
            self.term_n+=1
        if self.n_calls % 64 == 0:
            if time.monotonic()-self.started >= self.wall_limit:
                self.reason="WALL_LIMIT"; return False
            if psutil.virtual_memory().available < 3*1024**3:
                self.reason="RAM_RESERVE_BELOW_3_GIB"; return False
        return True

    def _on_rollout_end(self):
        available=psutil.virtual_memory().available
        rss=uss=0
        for process in [psutil.Process(),*psutil.Process().children(recursive=True)]:
            try:
                mem=process.memory_full_info(); rss+=mem.rss; uss+=mem.uss
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        self.max_rss=max(self.max_rss,rss); self.max_uss=max(self.max_uss,uss)
        self.min_available=min(self.min_available,available)
        if available < 3*1024**3:
            self.reason="RAM_RESERVE_BELOW_3_GIB"
            raise MemoryError(self.reason)
        self.rollouts+=1
        counters=self.training_env.env_method("counters")
        total={k:sum(v[k] for v in counters) for k in counters[0]}
        elapsed=time.monotonic()-self.started
        record={"rollout":self.rollouts,"transitions":self.num_timesteps,
                "elapsed_training_s":elapsed,"transitions_per_s":self.num_timesteps/elapsed,
                "actual_counts":total,"process_tree_rss_bytes":rss,"process_tree_uss_bytes":uss,
                "available_ram_bytes":available,
                "reward_contribution_mean_per_tick":{k:v/max(1,self.term_n) for k,v in self.terms.items()}}
        with (self.output/"ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        atomic_json(self.output/"live_status.json",record)
        self.terms.clear(); self.term_n=0
        if self.rollouts % 8 == 0 or self.rollouts==1:
            print(json.dumps({k:record[k] for k in ("rollout","transitions","elapsed_training_s","transitions_per_s")}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--config",type=Path,default=ROOT/"training/configs/pilot03.json")
    p.add_argument("--arm",choices=["B","R","M"],default="B")
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--envs",type=int,choices=[4,8],required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--max-wall",type=float,required=True)
    p.add_argument("--kind",choices=["preflight","pilot","formal"],required=True)
    p.add_argument("--formal-budget",type=Path)
    a=p.parse_args()
    if a.steps<=0 or a.steps%4096: p.error("steps must be a positive multiple of the 4096-sample rollout")
    limits={"preflight":(32768,600),"pilot":(1048576,5400),"formal":(4194304,10800)}
    if a.steps>limits[a.kind][0] or not 0<a.max_wall<=limits[a.kind][1]: p.error("Per-run contract cap exceeded")
    if a.kind=="formal":
        if a.formal_budget is None: p.error("Formal training requires a frozen --formal-budget")
        budget=json.loads(a.formal_budget.read_text())
        if not budget["pilot_qualified"] or a.steps!=budget["steps_per_run"] or a.seed not in [11,23,37]:
            p.error("Formal budget/gate/seed mismatch")
    if a.kind=="pilot" and a.seed!=101: p.error("Development pilot seed is 101")
    a.output=a.output.resolve(); a.config=a.config.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text())
    snapshot=a.output/"source_snapshot"; snapshot.mkdir()
    sources=[Path(__file__).resolve(), *environment_sources(config),
             ROOT/"training/d1_rl/forkserver_bootstrap.py",a.config,
             ROOT/"tools/export_d1_ppo.py", ROOT/"tools/evaluate_d1_ppo.py"]
    identities={str(path):sha(path) for path in sources}
    for path in sources: shutil.copy2(path,snapshot/path.name)
    scene=ROOT/config["scene"]
    robot=scene.parent/"robot.xml"
    identities.update({str(scene):sha(scene),str(robot):sha(robot),str(ROOT/config["interface_schema"]):sha(ROOT/config["interface_schema"])})
    hyper={"learning_rate":3e-4,"n_steps":4096//a.envs,"batch_size":256,"n_epochs":4,
           "gamma":.99,"gae_lambda":.95,"clip_range":.2,"ent_coef":.005,"vf_coef":.5,
           "max_grad_norm":.5,"policy_arch":[128,128],"activation":"tanh","log_std_init":-1.0,
           "device":"cpu","torch_threads":1,"random_initialization":True,
           "teacher_policy_used":False,"warm_start":False,
           "forkserver_preload":"d1_rl.forkserver_bootstrap"}
    protocol={"started_utc":datetime.now(timezone.utc).isoformat(),"kind":a.kind,"arm":a.arm,
              "seed":a.seed,"envs":a.envs,"planned_transitions":a.steps,"planned_native_steps":a.steps*10,
              "wall_limit_training_s":a.max_wall,"inputs_sha256":identities,"hyperparameters":hyper,
              "environment_class":environment_class(config).__name__,
              "versions":{"mujoco":mujoco.__version__,"gymnasium":gymnasium.__version__,
                          "sb3":stable_baselines3.__version__,"torch":torch.__version__},"config":config}
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    mp.set_forkserver_preload(["d1_rl.forkserver_bootstrap"])
    vec=model=callback=None; status="ERROR"; error=None; counts=None; per_worker=None
    started=time.monotonic()
    try:
        if psutil.virtual_memory().available<3*1024**3: raise MemoryError("Insufficient 3GiB RAM reserve before spawn")
        vec=SubprocVecEnv([make_env(a.config,a.arm,a.seed,i,a.steps//a.envs,a.output/f"env_{i:02d}")
                          for i in range(a.envs)],start_method="forkserver")
        if psutil.virtual_memory().available<3*1024**3: raise MemoryError("Insufficient 3GiB RAM reserve after spawn")
        model=PPO("MlpPolicy",vec,learning_rate=3e-4,n_steps=4096//a.envs,batch_size=256,n_epochs=4,
                  gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.005,vf_coef=.5,max_grad_norm=.5,
                  policy_kwargs={"net_arch":{"pi":[128,128],"vf":[128,128]},
                                 "activation_fn":torch.nn.Tanh,"log_std_init":-1.0},
                  seed=a.seed,device="cpu",verbose=0)
        model.set_logger(configure(str(a.output),["csv"]))
        callback=Ledger(a.output,a.max_wall)
        model.learn(total_timesteps=a.steps,callback=callback,progress_bar=False)
        status="COMPLETE" if model.num_timesteps==a.steps else "PARTIAL"
        if callback.reason: status=callback.reason
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        (a.output/"error.txt").write_text(traceback.format_exc())
    finally:
        if vec is not None:
            try:
                per_worker=vec.env_method("counters")
                counts={k:sum(v[k] for v in per_worker) for k in per_worker[0]}
            except Exception as exc:
                error=(error or "")+f"; counter collection failed: {exc}"
            try:
                vec.close()
            except Exception as exc:
                error=(error or "")+f"; vector close failed: {exc}"
                for worker in vec.processes:
                    if worker.is_alive(): worker.terminate()
                for worker in vec.processes:
                    worker.join(timeout=2)
                    if worker.is_alive(): worker.kill(); worker.join(timeout=1)
        if model is not None: model.save(str(a.output/("policy.zip" if status=="COMPLETE" else "partial_policy.zip")))
        result={"status":status,"error":error,"elapsed_total_s":time.monotonic()-started,
                "trained_transitions":int(model.num_timesteps) if model is not None else 0,
                "actual_counts":counts,"per_worker_counts":per_worker,
                "inputs_unchanged":all(sha(Path(s))==h for s,h in identities.items()),
                "peak_process_tree_rss_bytes":callback.max_rss if callback else None,
                "peak_process_tree_uss_bytes":callback.max_uss if callback else None,
                "min_available_ram_bytes":callback.min_available if callback and np.isfinite(callback.min_available) else None,
                "finished_utc":datetime.now(timezone.utc).isoformat()}
        if callback:
            result["elapsed_training_s"]=time.monotonic()-callback.started
            result["training_transitions_per_s"]=result["trained_transitions"]/result["elapsed_training_s"]
        if counts is not None and counts["native_steps"]!=counts["policy_transitions"]*10:
            result["status"]="INVALID_PARTIAL_NATIVE_INTERVAL"
        atomic_json(a.output/"result.json",result); print(json.dumps(result,indent=2),flush=True)
    return 0 if result["status"]=="COMPLETE" and result["inputs_unchanged"] else 1

if __name__=="__main__": raise SystemExit(main())
