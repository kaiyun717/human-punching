"""Learning packets remain bounded, explicit, and separate from job submission."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).parents[1] / "scripts/task10_engineering_training.py"
    spec = importlib.util.spec_from_file_location("task10_learning_packet_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("minutes,gib", [(True, 20), (9, 20), (1441, 20), (60., 20), (60, True), (60, 3), (60, 33)])
def test_invalid_resource_budget_rejected(wrapper, minutes, gib):
    with pytest.raises(ValueError):
        wrapper.resource_budget(minutes, gib)


def test_preparation_freezes_inputs_without_submitting_and_marks_hour_boundary(wrapper, monkeypatch, tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    for name in ("task10_engineering_training.py", "task10_engineering_training.sbatch"):
        (root / "scripts" / name).write_text("fixture, not executable learning\n")
    payload = root / "evidence.json"
    payload.write_text('{"synthetic_fixture":true}\n')
    monkeypatch.setattr(wrapper, "ROOT", root)
    monkeypatch.setattr(wrapper.os, "fsencode", lambda _: b"short-fixture-path")
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("Preparation must not execute"))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **k: pytest.fail("Preparation must not submit"))
    from human_punching.wbc.runtime.engineering_recipe import LEGACY_RECIPE
    module = SimpleNamespace(checked_engineering_plan=lambda *a, **k: {
        "engineering_recipe": LEGACY_RECIPE.manifest(), "engineering_recipe_sha256": LEGACY_RECIPE.sha256,
        "synthetic_fixture": True, "input_files": {"evidence.json": wrapper.sha256_file(payload)},
        "projection": {"learning_seconds": 100., "measured_initialization_seconds": 20., "panel_seconds": 40.}})
    monkeypatch.setitem(sys.modules, "human_punching.wbc.engineering_training", module)
    for minutes in (60, 61):
        args = SimpleNamespace(bundle=payload, engineering_admission=payload, observer_report=payload,
            bridge_report=payload, capacity_report=payload, iterations=2000, seed=101,
            resume=None, resume_manifest_sha256=None, output=root / f"packet_{minutes}",
            wall_minutes=minutes, products_gib=20, engineering_recipe=LEGACY_RECIPE.recipe_id,
            training_initialization_report=None)
        record = wrapper.prepare(args)
        assert record["long_session_approval_required_before_submission"] is (minutes > 60)
        assert record["automatic_followup"] is False and record["promotion"] is False
        assert record["resources"]["application_wall_seconds"] == minutes * 60 - 120
        assert record["files"]["evidence.json"] == wrapper.sha256_file(args.output / "source_snapshot/evidence.json")
        assert record["status"] == "prepared_not_submitted"
        assert record["engineering_recipe"] == LEGACY_RECIPE.recipe_id
        assert record["engineering_recipe_sha256"] == LEGACY_RECIPE.sha256
        assert record["training_initialization_report"] is None
        assert record["resources"]["nodes"] == 1 and record["resources"]["tasks"] == 1
        assert "--nodes=1" in record["submission_command"] and "--ntasks=1" in record["submission_command"]
        assert "--gres=gpu:l40s:2" in record["submission_command"]
        assert not any(item.startswith("--gpus=") for item in record["submission_command"])


def test_measured_complete_session_cannot_be_hidden_in_a_short_allocation(wrapper):
    plan = {"projection": {"learning_seconds": 3500., "measured_initialization_seconds": 100.,
                           "panel_seconds": 400.}}
    with pytest.raises(ValueError, match="complete measured session"):
        wrapper.measured_wall_budget(plan, wrapper.resource_budget(60, 20))
    estimate = wrapper.measured_wall_budget(plan, wrapper.resource_budget(90, 20))
    assert estimate["required_application_seconds"] == 5000
    assert estimate["shutdown_reserve_seconds"] == 120


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_wall_projection_rejects_missing_or_nonphysical_measurement(wrapper, value):
    plan = {"projection": {"learning_seconds": value, "measured_initialization_seconds": 10.,
                           "panel_seconds": 10.}}
    with pytest.raises(ValueError, match="Positive finite"):
        wrapper.measured_wall_budget(plan, wrapper.resource_budget(60, 20))


def test_execution_requires_allocation_before_any_native_or_process_call(wrapper, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema": wrapper.SCHEMA}))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No native process"))
    with pytest.raises(ValueError, match="allocated"):
        wrapper.execute(packet)


def test_execution_rejects_changed_allocation_budget(wrapper, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    resources = wrapper.resource_budget(60, 20)
    resources["application_wall_seconds"] = 7200
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema": wrapper.SCHEMA,
        "scope": "canonical_four_family_scratch_engineering", "production_qualified": False,
        "automatic_followup": False, "promotion": False, "resources": resources}))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No native process"))
    with pytest.raises(ValueError, match="resource bounds"):
        wrapper.execute(packet)


def test_revised_preparation_forwards_and_freezes_recipe_evidence(wrapper, monkeypatch, tmp_path):
    from human_punching.wbc.runtime.engineering_recipe import REVISED_RECIPE
    root = tmp_path / "project"; (root / "scripts").mkdir(parents=True)
    for name in ("task10_engineering_training.py", "task10_engineering_training.sbatch"):
        (root / "scripts" / name).write_text("synthetic wrapper, not a learning program")
    evidence = root / "initialization.json"; evidence.write_text('{"synthetic":true}')
    monkeypatch.setattr(wrapper, "ROOT", root)
    monkeypatch.setattr(wrapper.os, "fsencode", lambda _: b"short-fixture-path")
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No native execution"))
    seen = {}
    def checked(*args, **kwargs):
        seen.update(kwargs)
        return {"engineering_recipe": REVISED_RECIPE.manifest(), "engineering_recipe_sha256": REVISED_RECIPE.sha256,
            "input_files": {evidence.name: wrapper.sha256_file(evidence)},
            "inputs": {"training_initialization_report": {"path": evidence.name, "sha256": wrapper.sha256_file(evidence)}},
            "projection": {"learning_seconds": 20., "measured_initialization_seconds": 20., "panel_seconds": 20.}}
    monkeypatch.setitem(sys.modules, "human_punching.wbc.engineering_training",
                        SimpleNamespace(checked_engineering_plan=checked))
    args = SimpleNamespace(bundle=evidence, engineering_admission=evidence, observer_report=evidence,
        bridge_report=evidence, capacity_report=evidence, iterations=25, seed=101, resume=None,
        resume_manifest_sha256=None, output=root / "prepared", wall_minutes=30, products_gib=20,
        engineering_recipe=REVISED_RECIPE.recipe_id, training_initialization_report=evidence)
    record = wrapper.prepare(args)
    assert seen["engineering_recipe"] == REVISED_RECIPE.recipe_id
    assert seen["training_initialization_report"] == evidence
    assert record["engineering_recipe_sha256"] == REVISED_RECIPE.sha256
    assert record["training_initialization_report"] == evidence.name
    assert record["files"][evidence.name] == wrapper.sha256_file(args.output / "source_snapshot" / evidence.name)
    assert record["long_session_approval_required_before_submission"] is False
