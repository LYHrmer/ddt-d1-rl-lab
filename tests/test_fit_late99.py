"""Exercise both round-99 fits with an archived scalar toy plant and synthetic tensors."""
import json

import pytest

torch = pytest.importorskip("torch")

from tools99 import common99, fit99


class ToyPlant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("scale", torch.ones(1))

    def step(self, state, action, terrain):
        assert terrain.shape == (len(state),) and bool((terrain == 1).all())
        x = state[:, :1] + self.scale * self.residual(action[:, :1])
        return torch.cat((x, state[:, 1:]), dim=1)


def fresh(weights):
    plant = ToyPlant()
    plant.load_state_dict(weights)
    return plant.eval().requires_grad_(False)


@pytest.fixture
def toy_archive(tmp_path, monkeypatch):
    seed = 94011
    base = tmp_path / "endpoint98"
    fit = base / "fit_01"
    prior = fit / f"seed_{seed}"
    endpoint = prior / "endpoint"
    endpoint.mkdir(parents=True)
    old = prior / "initial.pt"
    old.write_bytes(b"fixed toy policy")
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    digest = fit99.fit96.digest
    monkeypatch.setattr(common99, "PLAN", plan)
    monkeypatch.setattr(common99, "PLAN_SHA", digest(plan))
    monkeypatch.setattr(fit99, "ROOT", tmp_path)
    monkeypatch.setattr(fit99, "seed_dir", lambda s: prior)
    monkeypatch.setattr(fit99.data99, "BASE98", base)
    monkeypatch.setattr(fit99.fit96, "_fresh_world", fresh)
    monkeypatch.setattr(fit99, "UPDATES", 1)

    parent = ToyPlant()
    opt = torch.optim.Adam(parent.residual.parameters(), lr=fit99.learn94.LR)
    param = parent.residual.weight
    param.grad = torch.ones_like(param)
    opt.step()
    opt.state[param]["step"].fill_(175)
    with torch.no_grad():
        param.zero_()
    ref = fit99.fit96._save(endpoint / "world.pt", parent, opt)
    arm = {"status": "COMPLETE", "updates_attempted": 25, "updates_completed": 25, "attempted_model_steps": 7200,
           "completed_model_steps": 7200, "lambda_endpoint": 1., "checkpoint": ref}
    fit99.write(endpoint / "receipt.json", arm)

    origin = torch.zeros(18, 45)
    origin[:, 0] = 1.1
    origin[:, 3] = 1.
    batch = {"start": origin, "applied": torch.zeros(18, 16, 16), "target": origin[:, None].repeat(1, 16, 1),
             "paired_x_m": torch.zeros(3, 8, dtype=torch.float64)}
    actions = torch.full((6, 16, 16), .1)
    actions[:3] *= -1.
    targets = (.02 * torch.arange(1, 17, dtype=torch.float64)).repeat(3, 1)
    flags = [{"arm": "m1", "reset": 96103, "tick": 32, "flags": ["illegal", "terminated"]}]
    data = fit99.data99.TrainingData(batch, actions, targets, 1.e-6, flags)
    monkeypatch.setattr(fit99.data99, "load_training_data", lambda s: data)
    paths = [{"path": f"synthetic_{i}"} for i in range(18)]
    source = {"policy94": {"sha256": digest(old)}, "direction95_tensors_sha256": "toy",
              "ridge_and_stats_sha256": fit99.mapping_sha256(dict(parent.named_buffers())), "training_traces": paths}
    fit99.write(prior / "receipt.json", {"seed": seed, "status": "COMPLETE", "arms": {"endpoint": arm}, "source": source,
                "normalization": {"denominator_m2": data.early_denom}, "batch": {"tensors_sha256": fit99.mapping_sha256(batch)}})
    files = {str(path.relative_to(fit)): digest(path) for path in
             (prior / "receipt.json", endpoint / "receipt.json", endpoint / "world.pt")}
    fit99.write(fit / "seal.json", {"stage": "fit", "files": files})
    monkeypatch.setattr(fit99.data99, "FIT98_SEAL", digest(fit / "seal.json"))
    return seed, endpoint / "world.pt", data


def test_equal_forks_from_175_use_one_weight_and_preserve_provenance(tmp_path, toy_archive):
    seed, parent_file, data = toy_archive
    dest = tmp_path / "fit99"
    before = fit99.fit96.digest(parent_file)
    report = fit99.fit_seed(seed, dest)
    assert report["status"] == "COMPLETE", report["error"]
    assert report["updates_attempted"] == report["updates_completed"] == 2
    assert report["attempted_model_steps"] == report["completed_model_steps"] == 768
    assert report["policy_updates"] == report["ppo_updates"] == 0
    assert report["flagged_endpoints"] == data.flagged_endpoints
    assert len(report["source"]["training_traces"]) == 18
    assert report["source"]["parent_checkpoint"]["sha256"] == before == fit99.fit96.digest(parent_file)
    assert report["normalization"]["early_denominator_m2"] == data.early_denom
    assert report["normalization"]["late_denominator_m2"] == float(data.late_targets.square().mean())
    assert report["batch"]["prefix_tensors_sha256"] == fit99.mapping_sha256(data.prefix_batch)
    values = {}
    for arm, weight in fit99.WEIGHTS.items():
        row = report["arms"][arm]
        first, = row["losses"]
        assert row["lambda_late"] == weight and first["update"] == 1
        assert row["attempted_model_steps"] == row["completed_model_steps"] == 384
        assert row["initial_adam_sha256"] == report["source"]["parent_checkpoint"]["adam_sha256"]
        assert row["initial_weights_sha256"] == report["source"]["parent_checkpoint"]["weights_sha256"]
        assert json.loads((dest / arm / "progress.jsonl").read_text()) == first
        assert first["total"] == pytest.approx(first["state_h8"] + first["response_h8"] + first["state_h16"]
                                              + weight * first["response_h17_32"])
        assert first["response_h17_32"] == pytest.approx(1.)
        model = fit99.load_fit(seed, dest, arm)
        assert not any(p.requires_grad for p in model.parameters())
        assert model.scale.item() == 1.
        values[arm] = model.residual.weight.item()
    assert values["control"] < 0. < values["late"]
    assert report["arms"]["late"]["losses"][0]["preclip_gradient_norm"] > 1.
    with pytest.raises(FileExistsError):
        fit99.fit_seed(seed, dest)
    with (dest / "late/world.pt").open("ab") as output:
        output.write(b"altered")
    with pytest.raises(ValueError, match="changed source file"):
        fit99.load_fit(seed, dest, "late")


@pytest.mark.parametrize("both_fail", [True, False])
def test_suffix_failure_preserves_attempts_and_blocks_the_sibling(tmp_path, toy_archive, monkeypatch, both_fail):
    seed, parent_file, _ = toy_archive
    started = 0

    def failing(weights):
        nonlocal started
        started += 1
        world = fresh(weights)
        if started == 1 and not both_fail:
            return world
        original = world.step
        def step(state, action, terrain):
            if len(state) == 6:
                raise RuntimeError("synthetic suffix fault")
            return original(state, action, terrain)
        world.step = step
        return world

    monkeypatch.setattr(fit99.fit96, "_fresh_world", failing)
    dest = tmp_path / "partial99"
    report = fit99.fit_seed(seed, dest)
    assert report["status"] == "PARTIAL" and report["updates_completed"] == int(not both_fail)
    assert fit99.fit96.digest(parent_file) == report["source"]["parent_checkpoint"]["sha256"]
    for name, row in report["arms"].items():
        if name == "control" and not both_fail:
            assert row["status"] == "COMPLETE" and (dest / name / "world.pt").is_file()
        else:
            assert row["status"] == "PARTIAL" and "synthetic suffix fault" in row["error"]
            assert row["updates_attempted"] == 1 and not row["losses"]
            assert row["attempted_model_steps"] == 294 and row["completed_model_steps"] == 288
            assert row["checkpoint"] is None and (dest / name / row["partial_checkpoint"]["file"]).is_file()
        with pytest.raises(ValueError, match="incomplete"):
            fit99.load_fit(seed, dest, name)
    with pytest.raises(FileExistsError):
        fit99.fit_seed(seed, dest)


def test_corrupt_endpoint_is_rejected_before_any_model_step(tmp_path, toy_archive):
    seed, parent_file, _ = toy_archive
    with parent_file.open("ab") as out:
        out.write(b"broken parent checkpoint")
    report = fit99.fit_seed(seed, tmp_path / "bad_parent")
    assert report["status"] == "PARTIAL" and "changed source file" in report["error"]
    assert report["updates_attempted"] == report["attempted_model_steps"] == 0
    assert all(row["status"] == "NOT_RUN" for row in report["arms"].values())
