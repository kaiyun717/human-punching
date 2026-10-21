"""Untrained native-RSL export fixtures; no physics, rollouts or GPU jobs."""
from copy import deepcopy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from human_punching.wbc.checkpointing import (BoundaryProgress, build_checkpoint_contract,
                                             save_boundary_checkpoint)
from human_punching.wbc.export_bundle import package_tracker, verify_tracker, _planner, _xml_assets, _member
from human_punching.wbc.provenance import PROJECT_ROOT, sha256_file, write_json
from human_punching.wbc.runtime.bundle import load_runtime_bundle


@pytest.fixture(scope="module")
def fixture_inputs(tmp_path_factory, task10_common_bundle):
    root = tmp_path_factory.mktemp("export_inputs")
    # Public-API-generated library remains deliberately unqualified for training.
    bundle = task10_common_bundle
    torch.manual_seed(10010)
    observations = TensorDict({"actor": torch.randn(8, 6)*2+1,
                               "critic": torch.randn(8, 9)}, batch_size=[8])
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(observations, groups, "actor", 48, hidden_dims=(16, 8),
                     obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution",
                                                               "init_std": .7, "std_type": "scalar"})
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=(16, 8), obs_normalization=True)
    actor.update_normalization(observations)
    critic.update_normalization(observations)
    storage = RolloutStorage("rl", 8, 24, observations, [48], "cpu")
    algorithm = PPO(actor, critic, storage, learning_rate=.0003, device="cpu")
    # Populate real Adam serialization state without PPO rollouts or training.
    for parameter in list(actor.parameters())+list(critic.parameters()):
        parameter.grad = torch.full_like(parameter, .01)
    algorithm.optimizer.step()
    algorithm.optimizer.zero_grad()
    source = root/"fixture.py"
    source.write_text("# UNTRAINED EXPORT TEST FIXTURE\n")
    interfaces = {
        "actor_observation_schema": {"schema": "human_wbc_observations_v1", "groups": {
            "actor": {"shape": [6], "terms": [{"name": "fixture", "shape": [6]}]}}},
        "action_schema": {"joint_names": list(bundle.joint_names), "dimension": 48, "raw_clip": 1.},
        "simulation_contract": {"physics_dt_s": .005, "decimation": 4},
        "learner_configuration": {"rollout_steps": 24, "epochs": 5, "minibatches": 4}}
    contract = build_checkpoint_contract(body_path=bundle.model_path,
        controller=bundle.controller_contract(), reference_manifest=bundle.root/"manifest.json",
        implementation_sources={"fixture.py": source}, **interfaces)
    runner = SimpleNamespace(alg=algorithm, device="cpu", gpu_global_rank=0, gpu_world_size=1,
        current_learning_iteration=7, logger=SimpleNamespace(tot_timesteps=0, tot_time=0),
        env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0)))
    checkpoint = root/"checkpoint"
    receipt = save_boundary_checkpoint(checkpoint, runner, contract=contract,
        progress=BoundaryProgress(8, "canonical", 8), sampler_identity={"fixture": True})
    jit, onnx = root/"policy_jit.pt", root/"policy.onnx"
    torch.jit.script(actor.as_jit().cpu().eval()).save(str(jit))
    wrapper = actor.as_onnx(verbose=False).cpu().eval()
    torch.onnx.export(wrapper, wrapper.get_dummy_inputs(), str(onnx), input_names=wrapper.input_names,
        output_names=wrapper.output_names, opset_version=18, dynamo=False)
    planner_root = root/"planner"
    planner_root.mkdir()
    (planner_root/"fixture.py").write_text("# Source closure fixture; not a physical planner.\n")
    np.savez(planner_root/"data.npz", values=np.array([1, 2, 3]))
    planner_manifest = write_json(planner_root/"manifest.json", {
        "schema": "human_wbc_planner_bundle_v1", "entry_point": "fixture:prepare",
        "runtime_dependencies": ["numpy"],
        "portability_limitations": ["Fixture only; no planner implementation or physical qualification."],
        "files": [{"path": name, "sha256": sha256_file(planner_root/name), "role": role}
                  for name, role in (("fixture.py", "source"), ("data.npz", "data"))]})
    domain_manifest = write_json(root/"domain.json", {
        "schema": "human_wbc_domain_manifest_v1", "qualified": False, "cells": [],
        "families": ["jab", "cross", "left_hook", "right_hook"],
        "reference_manifest_sha256": sha256_file(bundle.root/"manifest.json"),
        "model_sha256": sha256_file(bundle.model_path), "controller_sha256": bundle.controller_sha256(),
        "reference_generator_sha256": sha256_file(planner_manifest)})
    return dict(policy_jit=jit, policy_onnx=onnx, boundary_checkpoint=checkpoint,
        boundary_manifest_sha256=receipt["manifest_sha256"], reference_bundle=bundle.root,
        interface_contracts=interfaces, planner_manifest=planner_manifest,
        domain_manifest=domain_manifest, implementation_sources={"fixture.py": source})


def test_complete_untrained_native_rsl_fixture_and_independent_torch_load(tmp_path, fixture_inputs):
    output = tmp_path/"tracker"
    receipt = package_tracker(output, **fixture_inputs)
    manifest = json.loads((output/"manifest.json").read_text())
    assert receipt["status"] == "passed" and receipt["file_count"] > 20
    assert manifest["qualification"] == "engineering_fixture"
    assert manifest["trained_policy"] is False
    assert manifest["acceptance_qualified"] is manifest["portability_qualified"] is False
    assert manifest["coupled_runtime_qualified"] is manifest["promoted"] is False
    assert (output/"planner/data.npz").is_file()
    assert (output/"dependencies/requirements-wbc.lock").is_file()
    assert (output/"learner/rank_00.pt").is_file()
    bundle = load_runtime_bundle(output/"references")
    assert len(bundle.clips) == 4 and bundle.body["root_actuated"] is False
    with pytest.raises(ValueError, match="unqualified"):
        bundle.require_training_qualification()
    native = torch.jit.load(str(output/"policy/policy_jit.pt"), map_location="cpu").eval()
    assert native(torch.zeros(2, 6)).shape == (2, 48)
    assert "obs_normalizer._mean" in native.state_dict()
    parity = json.loads((output/"evidence/synthetic_export_parity.json").read_text())
    assert parity["sample_count"] == 32 and parity["max_action_abs_error"] <= 1e-5
    assert parity["ten_thousand_real_observation_gate_passed"] is False
    assert verify_tracker(output, manifest_sha256=receipt["manifest_sha256"])["file_count"] == receipt["file_count"]
    with pytest.raises(FileExistsError):
        package_tracker(output, **fixture_inputs)


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_closed_inventory_detects_corruption_missing_or_extra_files(tmp_path, fixture_inputs, change):
    output = tmp_path/"tracker"
    receipt = package_tracker(output, **fixture_inputs)
    target = output/"planner/data.npz"
    if change == "modified":
        target.write_bytes(target.read_bytes()+b"changed")
    elif change == "missing":
        target.unlink()
    else:
        (output/"untracked.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory|hash/size"):
        verify_tracker(output, manifest_sha256=receipt["manifest_sha256"])


def test_false_trained_label_rejects_unqualified_library(tmp_path, fixture_inputs):
    # Production cannot use a reduced source map even before its reference gate.
    with pytest.raises(ValueError, match="complete current human source inventory"):
        package_tracker(tmp_path/"false_train", **fixture_inputs, qualification="trained_candidate")
    with pytest.raises(ValueError, match="Unknown export label"):
        package_tracker(tmp_path/"false_acceptance", **fixture_inputs, qualification="accepted")
    with pytest.raises(ValueError, match="must not attach apparent training"):
        package_tracker(tmp_path/"false_evidence", **fixture_inputs,
                        training_evidence=fixture_inputs["domain_manifest"])


def test_contract_drift_fails_before_loading_exported_policy(tmp_path, fixture_inputs, monkeypatch):
    inputs = deepcopy(fixture_inputs)
    inputs["interface_contracts"]["action_schema"]["raw_clip"] = 2.
    monkeypatch.setattr(torch.jit, "load", lambda *a, **kw: pytest.fail("Contract gate must precede JIT"))
    with pytest.raises(ValueError, match="interface differs"):
        package_tracker(tmp_path/"drift", **inputs)


def test_changed_jit_normalizer_cannot_be_paired_with_other_checkpoint(tmp_path, fixture_inputs):
    inputs = deepcopy(fixture_inputs)
    actor = torch.jit.load(str(inputs["policy_jit"]), map_location="cpu")
    actor.obs_normalizer._mean.add_(.25)
    path = tmp_path/"wrong_actor.pt"
    actor.save(str(path))
    inputs["policy_jit"] = path
    with pytest.raises(ValueError, match="actor/normalizer differs"):
        package_tracker(tmp_path/"wrong", **inputs)


def test_planner_dependency_escape_missing_file_and_hash_drift(tmp_path, fixture_inputs):
    path = tmp_path/"planner.json"
    source = json.loads(fixture_inputs["planner_manifest"].read_text())
    source["files"][0]["path"] = "../escape.py"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Invalid portable member"):
        _planner(path)
    source["files"][0]["path"] = "missing.py"
    path.write_text(json.dumps(source))
    with pytest.raises(FileNotFoundError):
        _planner(path)
    (tmp_path/"missing.py").write_text("wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        _planner(path)


def test_body_external_asset_escape_is_rejected(tmp_path):
    xml = tmp_path/"body.xml"
    xml.write_text('<mujoco><asset><mesh file="../unbundled.stl"/></asset></mujoco>')
    with pytest.raises(ValueError, match="Invalid portable member"):
        _xml_assets(SimpleNamespace(root=tmp_path, model_path=xml))


def test_checkpoint_hash_corruption_precedes_pickle_load(tmp_path, fixture_inputs, monkeypatch):
    root = tmp_path/"checkpoint"
    shutil.copytree(fixture_inputs["boundary_checkpoint"], root)
    (root/"rank_00.pt").write_bytes(b"corrupt")
    inputs = {**fixture_inputs, "boundary_checkpoint": root}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Hash gate before pickle"))
    with pytest.raises(ValueError, match="rank file or receipt"):
        package_tracker(tmp_path/"corrupt", **inputs)


def test_mount_alias_allowed_but_internal_member_symlink_rejected(tmp_path):
    real = tmp_path/"real"
    real.mkdir()
    (real/"data.txt").write_text("same NFS data")
    alias = tmp_path/"alias"
    alias.symlink_to(real, target_is_directory=True)
    assert _member(alias, "data.txt") == real/"data.txt"
    (real/"linked.txt").symlink_to(real/"data.txt")
    with pytest.raises(ValueError, match="Symlink is not a portable artifact"):
        _member(alias, "linked.txt")
