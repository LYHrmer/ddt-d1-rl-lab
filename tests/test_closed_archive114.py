from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools94.common94 import write
from tools114 import closed_archive as archive


class TestClosedArchive(unittest.TestCase):
    def cohorts(self, passing):
        rows, stages = [], {}
        for i,seed in enumerate((*archive.DEV,*archive.BLIND)):
            paired = seed in archive.DEV
            status = "COMPLETE" if paired or passing else "NOT_STARTED"
            rows.append(dict(seed=seed, paired=paired, complete=status == "COMPLETE",
                             gate_pass=passing and status == "COMPLETE", stage_status=status))
            stages[f"native_{seed}"] = dict(status=status, reserved_at_unix_ns=i*10 if status == "COMPLETE" else None,
                                            completed_at_unix_ns=i*10+8 if status == "COMPLETE" else None)
        return rows, stages

    def test_failed_development_retains_three_unopened_blind_starts(self):
        rows, stages = self.cohorts(False)
        self.assertFalse(archive._cohorts(rows,stages))
        self.assertEqual([r["stage_status"] for r in rows[3:]],["CONDITIONAL_NOT_OPENED"]*3)
        stages[f"native_{archive.BLIND[1]}"]["status"] = "PARTIAL"
        with self.assertRaises(ValueError):
            archive._cohorts(rows,stages)

    def test_blind_must_follow_last_passing_development_and_all_must_close(self):
        rows, stages = self.cohorts(True)
        self.assertTrue(archive._cohorts(rows,stages))
        first = stages[f"native_{archive.BLIND[0]}"]
        first["reserved_at_unix_ns"] = 21
        with self.assertRaises(ValueError):
            archive._cohorts(rows,stages)
        first.update(reserved_at_unix_ns=None,status="NOT_STARTED")
        with self.assertRaises(ValueError):
            archive._cohorts(rows,stages)

    def test_completed_child_under_failed_guard_is_partial_and_cannot_qualify(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = archive.BLIND[0]
            name = f"native_{seed}"
            folder = root/name
            folder.mkdir()
            _, _, unit = archive.locations(name,1)
            header = dict(stage=name, attempt=1, unit=unit, registration_sha256="protocol", reserved_at_unix_ns=20)
            write(folder/"reservation.json",header)
            write(folder/"report.json",dict(**header, status="COMPLETE",completed_at_unix_ns=40,
                                            result={"counts":archive.LIMIT,"motion":{"passed":True}}))
            guard = "guard_"+name+"_01"
            guards = {guard:dict(stage=name,attempt=1,child_started=True,status="INCOMPLETE",error="guard failed")}
            stage = archive._stage(name,root,guards,root/"film"/archive.MOVIE.name,"protocol")
            row = archive._case(seed,stage,folder,guards)
            self.assertEqual(stage["status"],"PARTIAL")
            self.assertFalse(row["gate_pass"])
            self.assertIsNone(row["motion"])
            self.assertEqual(row["counts"]["physics"],6000)

    def test_unknown_partial_counts_consume_a_full_episode_upper_bound(self):
        seed = archive.DEV[0]
        stage = {f"native_{seed}":dict(reserved_at_unix_ns=5)}
        caps = dict(archive.LIMIT,episodes=1)
        usage = archive._budgets([dict(seed=seed,counts=None)],stage,{"maximum_totals":caps})
        self.assertEqual(usage["counted"]["physics"],0)
        self.assertEqual(usage["upper_bound"]["physics"],6000)
        self.assertEqual(usage["unknown_counts_seeds"],[seed])
        caps["physics"] = 5999
        with self.assertRaises(ValueError):
            archive._budgets([dict(seed=seed,counts=None)],stage,{"maximum_totals":caps})

    def test_finished_native_source_requires_the_film_to_be_resolved(self):
        with self.assertRaises(ValueError):
            archive._film(Path("."),Path("movie"),{"status":"NOT_STARTED"},{"status":"COMPLETE"})
        result = archive._film(Path("."),Path("movie"),{"status":"PARTIAL"},{"status":"COMPLETE"})
        self.assertEqual(result["status"],"PARTIAL")

    def test_synthetic_archive_keeps_partials_and_refuses_to_replace_a_different_target(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw, public, movie = tmp/"evidence", tmp/"public", tmp/"assets/parking114.mp4"
            plan, index = tmp/"plan.json", public/"closed_loop.json"
            raw.mkdir()
            write(plan,{"maximum_totals":dict(episodes=6, **{k:v*6 for k,v in archive.LIMIT.items()})})
            for seed in archive.DEV:
                name = f"native_{seed}"
                folder, guard = raw/name, raw/f"guard_{name}_01"
                _, _, unit = archive.locations(name,1)
                folder.mkdir()
                guard.mkdir()
                write(folder/"reservation.json",dict(registration_sha256="test", unit=unit, attempt=1, stage=name,
                                                     reserved_at_unix_ns=15,baseline_sha256=str(seed)))
                write(folder/"partial_counts.json",dict(counts=dict.fromkeys(archive.LIMIT,0),error="test stop"))
                write(guard/"report.json",dict(scope=f"/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}.service",
                      supervisor_sha256=archive.GUARD_SHA,schema="pressure-supervisor83-v1",timeout_s=600,
                      command=["/usr/bin/python3","-B","-m","tools114.closed_run","worker",name,"--attempt","1"],
                      status="INCOMPLETE",child_started=True,error="test stop",child_reaped=True,
                      owned_cgroup_children_empty=True))
                (guard/"resources.jsonl").write_text("{}\n")
            with patch.multiple(archive,BASE=raw,PUBLIC=public,MOVIE=movie,PLAN=plan,INDEX=index,
                                protocol=lambda:"test",baseline=lambda s:(None,{"trace_sha256":str(s)})):
                archive.publish()
                summary = archive.verify()
                self.assertEqual(summary["partial_stages"],[f"native_{s}" for s in archive.DEV])
                self.assertFalse(summary["film_delivered"])
                self.assertFalse(summary["blind_opened"])
                self.assertFalse(movie.exists())
                different = public/f"native_{archive.DEV[0]}"/"partial_counts.json"
                different.write_text('{"preserve":true}')
                with self.assertRaises(ValueError):
                    archive.publish()
                self.assertEqual(different.read_text(),'{"preserve":true}')
