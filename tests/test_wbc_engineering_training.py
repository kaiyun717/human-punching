"""Outer orchestration failures/counters; no simulation or learning is run."""
from types import SimpleNamespace as NS
import json

import pytest

from human_punching.wbc import engineering_training as m


def fake_runner():
    env = NS(common_step_counter=17)
    return NS(env=NS(unwrapped=env), current_learning_iteration=0,
              gpu_global_rank=0, gpu_world_size=1, is_distributed=False)


def panel(done, launches=0):
    return {"completed_iterations": done, "launched_attempts": launches, "stopping_metrics": None}


def learn_for(runner, seen):
    def learn(*, num_learning_iterations, init_at_random_ep_len):
        assert init_at_random_ep_len is False
        seen.append((runner.current_learning_iteration, num_learning_iterations))
        runner.current_learning_iteration += num_learning_iterations-1
        runner.env.unwrapped.common_step_counter += num_learning_iterations*24
    return learn


def test_chunks_keep_next_iteration_and_stop_after_four_unlaunched_panels():
    runner, seen, saved, pending = fake_runner(), [], [], []
    result = m.run_engineering_chunks(runner, iterations=2000, completed=0,
        inherited_learn=learn_for(runner, seen),
        validate_boundary=lambda done: {"ok": True, "panel": panel(done)},
        checkpoint_boundary=lambda *args: saved.append(args),
        pending_boundary=lambda *args: pending.append(args))
    assert seen == [(0, 250), (250, 250), (500, 250), (750, 250)]
    assert result["completed_iterations"] == 1000
    assert result["stop_reason"] == "four_panel_all_unlaunched_diagnosis"
    assert [row[0] for row in pending] == [row[0] for row in saved] == [250, 500, 750, 1000]
    assert result["production_accepted"] is False


def test_final_partial_boundary_and_resume_preserve_counts():
    runner, seen, saved = fake_runner(), [], []
    result = m.run_engineering_chunks(runner, iterations=273, completed=100,
        inherited_learn=learn_for(runner, seen),
        validate_boundary=lambda done: {"ok": True, "panel": panel(done, 12)},
        checkpoint_boundary=lambda *args: saved.append(args))
    assert seen == [(100, 150), (250, 123)]
    assert result["completed_iterations"] == 373
    assert [row[0] for row in saved] == [250, 373]
    assert not result["stop_learning"]


def test_failed_validation_keeps_full_completed_boundary_and_never_continues():
    runner, order, seen = fake_runner(), [], []
    def invalid(_):
        order.append("validation")
        raise ValueError("physical evidence missing")
    with pytest.raises(RuntimeError, match="physical evidence missing"):
        m.run_engineering_chunks(runner, iterations=500, completed=0,
            inherited_learn=learn_for(runner, seen), validate_boundary=invalid,
            pending_boundary=lambda *args: order.append("pending_checkpoint"),
            checkpoint_boundary=lambda done, histories, status, decision: order.append(
                (done, status["ok"], decision["stop_reason"])))
    assert seen == [(0, 250)]
    assert order == ["pending_checkpoint", "validation", (250, False, "validation_failed")]


@pytest.mark.parametrize("bad_counter", ["iteration", "rollout"])
def test_inherited_count_mismatch_cannot_be_reported_complete(bad_counter):
    runner = fake_runner()
    def bad(**kwargs):
        runner.current_learning_iteration = 250 if bad_counter == "iteration" else 249
        runner.env.unwrapped.common_step_counter += 250*24-(bad_counter == "rollout")
    with pytest.raises(RuntimeError, match="completed iterations"):
        m.run_engineering_chunks(runner, iterations=250, completed=0, inherited_learn=bad,
            validate_boundary=lambda _: pytest.fail("No validation after bad counts"),
            checkpoint_boundary=lambda *args: pytest.fail("No claimed completed boundary"))


def test_partial_panels_do_not_count_as_four_aligned_plateau_panels():
    assert not m.engineering_stop([panel(i) for i in (250, 500, 750, 999)])["stop_learning"]
    assert not m.engineering_stop([panel(i, 1) for i in (250, 500, 750, 1000)])["stop_learning"]
    assert not m.engineering_stop([panel(i) for i in (250, 500, 1000, 1250)])["stop_learning"]


def summary(launches):
    episodes = [{"launched": i < launches, "outward_minimum_target_distance_m": .13 if i < launches else None,
        "contact_ttc_error_s": None, "recovery_cycle_s": None, "requested_ttc_s": .7} for i in range(64)]
    return {"engineering_checkpoint_selection_eligible": True, "all_terminal": True,
        "episodes": episodes, "attempts": 64, "launched": launches,
        "families": {f: {"attempts": 16, "strict_cycle_success": 0}
                     for f in ("jab", "cross", "left_hook", "right_hook")}}


@pytest.mark.parametrize("launches", [0, 1, 63])
def test_unlaunched_physical_ttc_stays_null_and_not_general_plateau(launches):
    source = summary(launches)
    result = m.stopping_panel(source, 250)
    assert result["stopping_metrics"] is None
    assert all(row["contact_ttc_error_s"] is None for row in result["summary"]["episodes"])


def test_full_launch_failures_use_labelled_stopping_penalties_only():
    source = summary(64)
    result = m.stopping_panel(source, 250)
    assert result["stopping_metrics"]["timing_error"] == 4.
    assert result["stopping_metrics"]["recovery_time"] == pytest.approx(4.7)
    assert all(row["contact_ttc_error_s"] is None for row in source["episodes"])


def test_missing_terminal_attempt_prevents_stopping_summary():
    source = summary(64); source["all_terminal"] = False
    with pytest.raises(ValueError, match="64"):
        m.stopping_panel(source, 250)


def test_single_rank_status_is_unique_atomic_and_records_failure(tmp_path, monkeypatch):
    # File protocol only, no torch distributed initialization.
    monkeypatch.setattr(m, "writable_path", lambda path: path)
    monkeypatch.setattr(m, "write_json", lambda path, value: path.write_text(json.dumps(value)))
    runner = fake_runner()
    def broken():
        raise ValueError("missing trace")
    first = m.coordinated_panel(runner, tmp_path, completed=250,
        expectation_identity={"contract": "x"}, panel=broken, timeout_s=10.)
    second = m.coordinated_panel(runner, tmp_path, completed=250,
        expectation_identity={"contract": "x"}, panel=lambda: {"valid": True}, timeout_s=10.)
    assert first["ok"] is False and "missing trace" in first["error"]
    assert second["ok"] is True
    assert first["identity"]["nonce"] != second["identity"]["nonce"]
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob("*.incomplete"))


@pytest.mark.parametrize("budget", [0., -1., float("inf"), float("nan")])
def test_panel_protocol_requires_finite_bound(tmp_path, budget):
    with pytest.raises(ValueError, match="wall budget"):
        m.coordinated_panel(fake_runner(), tmp_path, completed=250, expectation_identity={},
                            panel=lambda: None, timeout_s=budget)


@pytest.mark.parametrize("iterations,seed,completed", [(True, 1, 0), (0, 1, 0), (2001, 1, 0),
    (1, -1, 0), (2000, 1, 1), (1, 1, True)])
def test_canonical_budget_rejects_invalid_or_cumulative_overflow(iterations, seed, completed):
    with pytest.raises(ValueError):
        m._bounds(iterations, seed, completed)


def test_module_cli_exists_and_does_not_start_learning_for_help():
    with pytest.raises(SystemExit) as raised:
        m.main(["--help"])
    assert raised.value.code == 0


@pytest.mark.parametrize("recipe_id", ["legacy_guard_start_v1", "reference_start_guard_max_v1"])
def test_sampler_and_curriculum_bind_recipe_without_confusing_strict_launches(recipe_id):
    from human_punching.wbc.runtime.engineering_recipe import resolve_engineering_recipe
    recipe = resolve_engineering_recipe(recipe_id)
    sampler = m.sampler_identity(101, recipe_id)
    curriculum = m.training_curriculum(recipe_id)
    assert sampler["engineering_recipe"] == curriculum["engineering_recipe"] == recipe.manifest()
    assert sampler["engineering_recipe_sha256"] == curriculum["engineering_recipe_sha256"] == recipe.sha256
    assert sampler["reference_start_probability"] == recipe.training.reference_start_probability
    assert sampler["initial_support_grace_ticks"] == recipe.training.initial_support_grace_ticks
    assert sampler["training_starts_count_as_strict_ready_launches"] is False
    assert sampler["capacity_or_historical_weights_allowed"] is False
    assert curriculum["strict_validation_recipe"] == "legacy_guard_start_v1"
    assert curriculum["interior_rsi"] is False and curriculum["mid_cycle_state_replacement"] is False
    assert curriculum["guard_joint_posture_metric"] == recipe.training.guard_joint_posture_metric
    assert curriculum["guard_joint_posture_weight"] == recipe.training.guard_joint_posture_weight


@pytest.mark.parametrize("recipe_id,report", [("legacy_guard_start_v1", "unused.json"),
    ("reference_start_guard_max_v1", None)])
def test_recipe_requires_exact_training_initialization_evidence_scope(recipe_id, report):
    with pytest.raises(ValueError, match="native training initialization evidence"):
        m._recipe_request(recipe_id, report)


def plan_for(recipe_id):
    from human_punching.wbc.runtime.engineering_recipe import resolve_engineering_recipe
    recipe = resolve_engineering_recipe(recipe_id)
    return {"seed": 101, "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
        "sampler_identity": m.sampler_identity(101, recipe_id),
        "inputs": {"training_initialization_report": {"path": "synthetic.json", "sha256": "f"*64}}
            if recipe.training.reference_start_probability else {},
        "measurement_contract": {"payload": {"reference_and_body": {
            "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256}}}}


@pytest.mark.parametrize("changed", ["recipe", "measured_recipe", "sampler", "initialization_evidence"])
def test_worker_rejects_recipe_drift_before_native_construction(changed):
    plan = plan_for("reference_start_guard_max_v1")
    legacy = plan_for("legacy_guard_start_v1")
    if changed == "recipe":
        plan["engineering_recipe"]["training"]["reference_start_probability"] = .5
    elif changed == "measured_recipe":
        plan["measurement_contract"] = legacy["measurement_contract"]
    elif changed == "sampler":
        plan["sampler_identity"] = legacy["sampler_identity"]
    else:
        del plan["inputs"]["training_initialization_report"]
    with pytest.raises(ValueError):
        m._plan_recipe(plan)


def resume_fixture(tmp_path, monkeypatch, recipe_id):
    """Synthetic metadata only; tests plan admission, never a valid learned payload."""
    from dataclasses import asdict
    from human_punching.wbc.checkpointing import CheckpointContract, CONTINUATION
    from human_punching.wbc.runtime.bundle import SOURCE_PINS
    root = tmp_path / "checkpoint"; root.mkdir()
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    def evidence(path):
        return {"path": str(path.relative_to(tmp_path)), "sha256": m.sha256_file(path)}
    monkeypatch.setattr(m, "_evidence", evidence)
    def read(record):
        path = tmp_path / record["path"]
        assert m.sha256_file(path) == record["sha256"]
        return path
    monkeypatch.setattr(m, "read_artifact", read)
    sources = {"synthetic_source.py": "b"*64}
    contract = CheckpointContract(*(["a"*64]*7), m.digest(sources), dict(SOURCE_PINS))
    progress = {"stage": "canonical", "completed_iterations": 250, "stage_completed_iterations": 250,
        "curriculum": m.training_curriculum(recipe_id),
        "validation": {"scope": m.SCOPE, "resume_allowed": True, "panels": []}}
    ranks = []
    for rank in range(2):
        member = root / f"rank_{rank}.bin"; member.write_bytes(b"synthetic, never load as learner")
        ranks.append({"rank": rank, "path": member.name, "sha256": m.sha256_file(member),
            **{key: "c"*64 for key in ("contract_sha256", "progress_sha256", "sampler_identity_sha256",
                                      "learned_parameters_sha256", "model_buffers_sha256")}})
    manifest = {"schema": "human_wbc_boundary_checkpoint_v1", "status": "complete", "world_size": 2,
        "contract": asdict(contract), "contract_sha256": contract.sha256, "progress": progress,
        "sampler_identity": m.sampler_identity(101, recipe_id), "continuation": CONTINUATION,
        "physical_state_saved": False, "ranks": ranks}
    return root, sources, manifest


@pytest.mark.parametrize("recipe_id", ["legacy_guard_start_v1", "reference_start_guard_max_v1"])
def test_resume_plan_accepts_only_same_recipe_and_curriculum(tmp_path, monkeypatch, recipe_id):
    root, sources, manifest = resume_fixture(tmp_path, monkeypatch, recipe_id)
    path = root / "manifest.json"; path.write_text(json.dumps(manifest))
    resumed, completed, files = m._resume_plan(root, m.sha256_file(path), seed=101,
        reference_hash="a"*64, source_hashes=sources, engineering_recipe=recipe_id)
    assert completed == 250 and len(files) == 3
    assert resumed["progress"]["curriculum"] == m.training_curriculum(recipe_id)
    other = "legacy_guard_start_v1" if recipe_id != "legacy_guard_start_v1" else "reference_start_guard_max_v1"
    with pytest.raises(ValueError, match="unchanged engineering lineage"):
        m._resume_plan(root, m.sha256_file(path), seed=101, reference_hash="a"*64,
            source_hashes=sources, engineering_recipe=other)
    manifest["progress"]["curriculum"]["reference_start_probability"] = .5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unchanged engineering lineage"):
        m._resume_plan(root, m.sha256_file(path), seed=101, reference_hash="a"*64,
            source_hashes=sources, engineering_recipe=recipe_id)
