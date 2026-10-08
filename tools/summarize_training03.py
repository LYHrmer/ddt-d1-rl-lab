#!/usr/bin/env python3
"""Summarize existing bounded PPO runs; read-only, zero simulation steps."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(run):
    protocol=json.loads((run/"protocol.json").read_text())
    final_path=run/"result.json"
    final=json.loads(final_path.read_text()) if final_path.exists() else None
    records=[json.loads(line) for line in (run/"ledger.jsonl").read_text().splitlines()]
    latest=records[-1] if records else {}
    episodes=[]
    for path in sorted(run.glob("env_*.monitor.csv")):
        with path.open() as stream:
            next(stream)
            episodes.extend(dict(r=float(x["r"]),l=int(x["l"]),t=float(x["t"]))
                            for x in csv.DictReader(stream))
    episodes.sort(key=lambda x:x["t"])
    tail=episodes[-100:]
    report={"run":str(run),"kind":protocol["kind"],"arm":protocol["arm"],
            "seed":protocol["seed"],"interface":protocol["config"].get("interface_id","03a"),
            "status":final["status"] if final else "RUNNING_OR_INTERRUPTED_NO_FINAL_RECEIPT",
            "planned_transitions":protocol["planned_transitions"],
            "actual_counts":final["actual_counts"] if final else latest.get("actual_counts"),
            "recorded_rollouts":len(records),"completed_episode_count":len(episodes),
            "last_100_episode_mean_seconds":statistics.mean(x["l"]*.02 for x in tail) if tail else None,
            "last_100_episode_mean_reward":statistics.mean(x["r"] for x in tail) if tail else None,
            "last_100_episode_time_limit_fraction":sum(x["l"]==1000 for x in tail)/len(tail) if tail else None,
            "latest_transitions_per_s":latest.get("transitions_per_s"),
            "qualification":"requires separate fixed final-policy evaluation; episode reward is not qualification"}
    return report,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    results=[summarize(path.resolve()) for path in a.runs]
    if a.output:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/"summary.json").write_text(json.dumps([x[0] for x in results],indent=2)+"\n")
        for i,(report,records) in enumerate(results):
            keys=["transitions","elapsed_training_s","transitions_per_s","process_tree_uss_bytes","available_ram_bytes"]
            with (a.output/f"run_{i}_curve.csv").open("x") as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
                writer.writerows({k:row[k] for k in keys} for row in records)
        # Matplotlib is loaded only when artifact output is explicitly requested.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
        for report,records in results:
            label=f'{Path(report["run"]).name} ({report["seed"]})'
            steps=[row["transitions"] for row in records]
            reward=[sum(row["reward_contribution_mean_per_tick"].values()) for row in records]
            axes[0].plot(steps,reward,label=label)
            axes[1].plot(steps,[row["transitions_per_s"] for row in records],label=label)
        axes[0].set(ylabel="Mean reward per policy tick",xlabel="Total environment transitions")
        axes[1].set(ylabel="Transitions / wall second",xlabel="Total environment transitions")
        for axis in axes: axis.grid(alpha=.2);axis.legend(fontsize=7)
        fig.suptitle("Training diagnostics — not a control qualification")
        fig.savefig(a.output/"training_diagnostics.svg");plt.close(fig)
    print(json.dumps([x[0] for x in results],indent=2))

if __name__=="__main__":main()
