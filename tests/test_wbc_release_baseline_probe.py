"""Source-equivalence and fail-closed diagnostic tests; no native integration."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.release_baseline import convert_release_baseline, frozen_source
from human_punching.wbc.release_baseline_probe import (
    CONDITIONS, ENVIRONMENTS, FAMILIES, SOLVERS, ReleaseRecorder, _probe_types,
    audit_release_arrays, build_release_probe_cfg, interpolate_legacy,
    legacy_observations, load_frozen_actor, load_release_inputs, run_release_baseline_probe, verify_release_model,
    _copy, _tensor_view, _write_report, diagnostic_episodes, execution_counts, historical_physics_steps, probe_conditions,
)
from human_punching.wbc.runtime.bundle import load_runtime_bundle

RELEASE = PROJECT_ROOT/"outputs/four-punch-milestone"


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp("probe_reference")/"bundle"
    convert_release_baseline(RELEASE, path)
    bundle = load_runtime_bundle(path)
    return bundle, load_release_inputs(bundle, RELEASE)


def test_scope_exactly_twelve_paired_sixteen_trial_conditions(baseline):
    bundle, inputs = baseline
    assert len(CONDITIONS) == 12 and ENVIRONMENTS == 16
    for family in FAMILIES:
        assert [row[1:] for row in CONDITIONS if row[0] == family] == [
            ("historical", True), ("accepted", True), ("accepted", False)]
        assert inputs[family]["initial_qpos"].shape == (16, 55)
        assert inputs[family]["initial_qvel"].shape == (16, 54)
        actor = load_frozen_actor(inputs[family], "cpu")
        assert not actor.training and not any(p.requires_grad for p in actor.parameters())
        assert actor(torch.zeros(16, 369)).shape == (16, 48)
    with pytest.raises(ValueError, match="unqualified"):
        bundle.require_training_qualification()


def test_raw_warp_copy_and_tensor_view_do_not_use_item_indexing(monkeypatch):
    import warp as wp
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    raw = wp.array(values, dtype=wp.float32, device="cpu")
    np.testing.assert_array_equal(_copy(raw), values)
    torch.testing.assert_close(_tensor_view(raw), torch.tensor(values))
    proxy = SimpleNamespace(_tensor=torch.tensor(values))
    torch.testing.assert_close(_tensor_view(proxy), torch.tensor(values))
    with pytest.raises(TypeError, match="Expected Torch"):
        _tensor_view(values)


def test_executed_counts_keep_failed_preintegration_separate_from_physics():
    recorder = ReleaseRecorder()
    recorder.rows = [{}]
    recorder.contacts = [{}]
    initial = np.zeros(16, np.float32)
    result = execution_counts(recorder, completed_policy_steps=0, initial_time=initial, final_time=initial,
                              final_reference_tick=np.zeros(16, int), final_sim_step_counter=0)
    assert result["actual_physics_steps"] == result["actual_actor_updates"] == 0
    assert result["actual_integrated_duration_s"] == 0.
    recorder.policies.append({})
    recorder.pre.extend([{}, {}, {}])
    recorder.rows.extend([{}, {}])
    result = execution_counts(recorder, completed_policy_steps=0, initial_time=initial,
                              final_time=np.full(16, .010, np.float32),
                              final_reference_tick=np.full(16, 3), final_sim_step_counter=3)
    assert result["actual_actor_updates"] == 1
    assert result["actual_physics_steps"] == result["actual_postintegration_records"] == 2
    assert result["actual_completed_policy_steps"] == 0
    assert result["actual_preintegration_records"] == result["actual_sim_step_counter"] == 3
    assert result["actual_integrated_duration_s"] == .010


def test_executed_counts_allow_native_float32_accumulation_without_planned_duration():
    recorder = ReleaseRecorder()
    clock = np.zeros(16, np.float32)
    for _ in range(1200):
        clock += np.float32(.005)
    result = execution_counts(recorder, completed_policy_steps=300,
                              initial_time=np.zeros(16), final_time=clock)
    assert result["actual_physics_steps"] == 1200
    assert result["actual_integrated_duration_s"] == 6.
    assert result["actual_native_elapsed_s"][0] != 6.
    unavailable = execution_counts(recorder, completed_policy_steps=0)
    assert unavailable["actual_physics_steps"] is None


def test_phase0_panels_preserve_all_conditions_and_paired_controls(baseline):
    bundle, _ = baseline
    assert probe_conditions("contract") == probe_conditions("diagnostic")
    parity = probe_conditions("parity")
    assert len(parity) == 16
    for family in FAMILIES:
        assert [row[1:] for row in parity if row[0] == family] == [
            ("historical", True, "learned"), ("historical", True, "zero"),
            ("accepted", True, "learned"), ("accepted", True, "zero")]
    full = load_release_inputs(bundle, RELEASE, num_envs=64)
    assert all(value["initial_qpos"].shape == (64, 55) for value in full.values())
    with pytest.raises(ValueError, match="mode"):
        probe_conditions("training")


@pytest.mark.parametrize("family,expected_steps,expected_actor_calls", [
    ("jab", 967, 242), ("cross", 961, 241), ("left_hook", 990, 248), ("right_hook", 982, 246)])
def test_historical_horizon_matches_accepted_release_and_migration(baseline, family, expected_steps, expected_actor_calls):
    _, inputs = baseline
    duration = inputs[family]["clip"]["legacy_episode_duration_s"]
    required = historical_physics_steps(duration)
    assert required == expected_steps
    assert (required+3)//4 == expected_actor_calls
    assert 0 <= expected_actor_calls*4-required < 4
    for root, variants in ((RELEASE/family, ("evaluation", "baseline")),
                           (PROJECT_ROOT/"outputs/task10/migration_20261019_04/runs"/family, ("learned", "baseline"))):
        for variant in variants:
            path = root/variant/"rollout_00.npz"
            if not path.is_file():
                pytest.skip("Historical horizon evidence is unavailable: "+str(path))
            with np.load(path, allow_pickle=False) as archive:
                assert int(archive["physics_step_count"]) == required
                assert len(archive["time_s"]) == required+1
                assert len(archive["policy_update_step_indices"]) == expected_actor_calls
                np.testing.assert_array_equal(archive["policy_update_step_indices"], np.arange(0, required, 4))


def test_historical_horizon_keeps_floating_boundary_step():
    assert historical_physics_steps(4.8) == 960
    assert historical_physics_steps(np.nextafter(4.8, np.inf)) == 961
    assert historical_physics_steps(4.800000000000001) == 961
    for invalid in (0., -1., float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite positive"):
            historical_physics_steps(invalid)


def test_p0p01_cross_saved_trace_remains_short_of_original_horizon(baseline):
    saved = PROJECT_ROOT/"outputs/task10b/p0p01/run/parity/cross_historical_ff_learned/raw_trace.npz"
    if not saved.is_file():
        pytest.skip("Preserved p0p01 short-horizon artifact is unavailable")
    _, inputs = baseline
    with np.load(saved, allow_pickle=False) as archive:
        observed = len(archive["time"])-1
    required = historical_physics_steps(inputs["cross"]["clip"]["legacy_episode_duration_s"])
    assert observed == 960 and required == 961
    assert observed < required <= ((required+3)//4)*4


def test_report_encoding_failure_never_creates_partial_json(tmp_path):
    path = tmp_path/"invalid"/"audit.json"
    with pytest.raises(TypeError, match="JSON serializable"):
        _write_report(path, {"numpy_boolean": np.bool_(True)})
    assert not path.exists() and not path.parent.exists()
    with pytest.raises(ValueError, match="JSON compliant"):
        _write_report(path, {"nonfinite": float("nan")})
    assert not path.exists()
    _write_report(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": True}
    with pytest.raises(FileExistsError):
        _write_report(path, {"passed": False})
    assert json.loads(path.read_text()) == {"passed": True}


def test_p0c05_saved_native_diagnostics_all_serialize_strictly_without_integration(baseline, monkeypatch):
    import mujoco
    saved = PROJECT_ROOT/"outputs/task10b/p0c05/run/diagnostic/jab_historical_ff"
    if not (saved/"raw_trace.npz").is_file():
        pytest.skip("Preserved p0c05 native regression artifact is unavailable")
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("Saved-trace regression cannot integrate"))
    monkeypatch.setattr(mujoco, "mj_step1", lambda *a: pytest.fail("Saved-trace regression cannot integrate"))
    monkeypatch.setattr(mujoco, "mj_step2", lambda *a: pytest.fail("Saved-trace regression cannot integrate"))
    _, inputs = baseline
    data = inputs["jab"]
    with np.load(saved/"raw_trace.npz", allow_pickle=False) as trace:
        arrays = {key: trace[key].copy() for key in trace.files}
    origins = []
    for world in range(16):
        with np.load(saved/f"episode_{world:02d}.npz", allow_pickle=False) as episode:
            origins.append(episode["target_w"]-np.asarray(data["clip"]["target_w"]))
    model = mujoco.MjModel.from_binary_path(str(saved/"compiled_model.mjb"))
    episodes, reports = diagnostic_episodes(model, arrays, data, np.asarray(origins))
    assert len(episodes) == len(reports) == 16
    decoded = json.loads(json.dumps({"diagnostics": reports}, allow_nan=False))
    assert len(decoded["diagnostics"]) == 16
    assert any(row["final_quiet_guard_dwell_s"] > 0 for row in reports)
    for world, row in enumerate(reports):
        assert row["trial_id"] == world
        assert type(row["final_quiet_guard_dwell_s"]) is float
        assert type(row["final_one_second_guard"]) is bool
        assert not any(isinstance(value, np.generic) for value in row.values())


@pytest.mark.parametrize("family", FAMILIES)
def test_observation_adapter_exactly_matches_frozen_source_with_raw_previous_actions(baseline, family):
    _, inputs = baseline
    source = inputs[family]
    reference = {key: torch.tensor(value, dtype=torch.float32) for key, value in source["legacy"].items()
                 if np.issubdtype(value.dtype, np.number)}
    module, _ = frozen_source(RELEASE, family)
    clock = torch.tensor([0., .505, 2.125, source["clip"]["legacy_episode_duration_s"]])
    def ref(key, offset=0.):
        value = module.interpolate_reference(reference["t"], reference[key], clock+offset)
        if key == "qpos":
            value[:, 3:7] /= value[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return value
    q = ref("qpos")
    v = ref("qvel")+.07
    gloves = ref("glove_pos")+.01
    previous = torch.arange(4*48, dtype=torch.float32).reshape(4, 48)/37.-2.
    fake = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q, qvel=v, site_xpos=gloves)),
        ref=ref, qadr=np.arange(7, 55), vadr=np.arange(6, 54), clock=clock,
        duration=source["clip"]["legacy_episode_duration_s"], hands=[0, 1], last_action=previous)
    expected = module.JabEnv.observe(fake)
    actual = legacy_observations(q, v, gloves, previous, clock, reference, fake.duration)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    clipped_previous = legacy_observations(q, v, gloves, previous.clamp(-1, 1), clock, reference, fake.duration)
    assert not torch.equal(actual, clipped_previous)
    v[1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        legacy_observations(q, v, gloves, previous, clock, reference, fake.duration)


def test_interpolation_source_float32_and_endpoint_hold():
    times = torch.tensor([0., .0167, .0333])
    values = torch.arange(3*54).reshape(3, 54).float()
    query = torch.tensor([-1., .005, .020, 3.])
    module, _ = frozen_source(RELEASE, "jab")
    torch.testing.assert_close(interpolate_legacy(times, values, query),
        module.interpolate_reference(times, values, query), rtol=0, atol=0)


def test_both_solver_profiles_are_local_and_keep_wbc_clocks_and_no_learning(baseline, monkeypatch):
    import mujoco
    from human_punching.wbc.runtime.task import build_human_env_cfg
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("CPU tests cannot integrate"))
    bundle, inputs = baseline
    for solver, ff in (("historical", True), ("accepted", True), ("accepted", False)):
        cfg = build_release_probe_cfg(bundle, ReleaseRecorder(), inputs["cross"], solver=solver, feedforward=ff)
        assert cfg.scene.num_envs == 16 and cfg.decimation == 4
        assert not cfg.auto_reset and not cfg.events and not cfg.terminations and not cfg.rewards
        assert cfg.commands["motion"].family == "cross"
        assert cfg.commands["motion"].feedforward_enabled is ff
        assert cfg.metrics["release_probe"].per_substep
        assert cfg.sim.nconmax == 128 and cfg.sim.njmax == 512
        for key, value in SOLVERS[solver].items():
            assert getattr(cfg.sim.mujoco, key) == value
    standard = build_human_env_cfg(bundle)
    assert standard.sim.mujoco.cone == "pyramidal" and standard.sim.mujoco.iterations == 10
    with pytest.raises(ValueError, match="Unknown"):
        build_release_probe_cfg(bundle, ReleaseRecorder(), inputs["jab"], solver="invented", feedforward=True)


def test_family_guard_reference_and_feedforward_hold_do_not_erase_endpoint_values():
    command_type, _, _ = _probe_types()
    command = command_type.__new__(command_type)
    command.physics_steps = torch.tensor([0, 100, 1000])
    command.trajectory_ids = torch.tensor([0, 1, 2])
    command.master_starts = torch.tensor([0, 3, 6])
    command.master_lengths = torch.tensor([3, 3, 3])
    command.master = {"joint_feedforward": torch.arange(9*48).float().reshape(9, 48),
                      "qvel": torch.arange(9*54).float().reshape(9, 54)}
    command.cfg = SimpleNamespace(feedforward_enabled=True)
    torch.testing.assert_close(command.sample_master("joint_feedforward"), command.master["joint_feedforward"][[0, 5, 8]])
    torch.testing.assert_close(command.sample_master("qvel"), command.master["qvel"][[0, 5, 8]])
    command.cfg.feedforward_enabled = False
    assert torch.count_nonzero(command.sample_master("joint_feedforward")) == 0
    assert torch.count_nonzero(command.sample_master("qvel")) > 0
    with pytest.raises(RuntimeError, match="episode initialization"):
        command._write_reference_state_to_sim(torch.tensor([0]))


def controller_arrays(bundle, inputs, *, feedforward=True):
    count, n = 9, 8
    with np.load(bundle.path(inputs["clip"]["master_path"])) as master:
        qref = np.tile(master["qpos"][:n, None], (1, 16, 1)).astype(np.float32)
        vref = np.tile(master["qvel"][:n, None], (1, 16, 1)).astype(np.float32)
        ff = np.tile(master["joint_feedforward"][:n, None], (1, 16, 1)).astype(np.float32)
    if not feedforward:
        ff[:] = 0
    actions = np.empty((2, 16, 48), np.float32)
    actions[0], actions[1] = 1.5, -2.
    pre_actions = np.repeat(actions, 4, axis=0)
    limits = np.asarray(bundle.body["joint_limits_rad"])
    target = np.clip(qref[..., 7:]+.25*np.clip(pre_actions, -1, 1), limits[:, 0], limits[:, 1])
    drive = np.asarray(bundle.body["kp"])*target+np.asarray(bundle.body["kd"])*vref[..., 6:]+ff
    q = np.tile(inputs["initial_qpos"].astype(np.float32)[None], (count, 1, 1))
    v = np.tile(inputs["initial_qvel"].astype(np.float32)[None], (count, 1, 1))
    caps = np.asarray(bundle.body["effort"])
    force = np.clip(drive-np.asarray(bundle.body["kp"])*q[:-1, :, 7:]
                    -np.asarray(bundle.body["kd"])*v[:-1, :, 6:], -caps, caps)
    return dict(time=np.tile((np.arange(count)*.005)[:, None], (1, 16)).astype(np.float32),
        qpos=q, qvel=v, ctrl=np.concatenate((drive[:1], drive)),
        actuator_force=np.concatenate((np.zeros((1, 16, 48)), force)), qfrc_actuator=np.zeros((count, 16, 54)),
        qfrc_applied=np.zeros((count, 16, 54)), xfrc_applied=np.zeros((count, 16, 24, 6)),
        overflow=np.zeros((count, 16), int), reference_physics_steps=np.tile(np.arange(count)[:, None], (1, 16)),
        reset_counts=np.ones((count, 16), int), sim_step_index=np.arange(count),
        policy_step_index=np.maximum(np.arange(count)-1, 0)//4,
        raw_action=np.concatenate((np.zeros((1, 16, 48)), pre_actions)),
        trajectory_id=np.zeros((count, 16), int), joint_position_target=np.concatenate((target[:1], target)),
        joint_velocity_target=np.concatenate((vref[:1, :, 6:], vref[..., 6:])),
        joint_effort_target=np.concatenate((ff[:1], ff)), pre_qref=qref, pre_vref=vref,
        pre_qpos=q[:-1], pre_qvel=v[:-1],
        pre_feedforward=ff, pre_reference_tick=np.tile(np.arange(n)[:, None], (1, 16)),
        pre_actor_action=pre_actions, policy_observation=np.zeros((2, 16, 369)), policy_action=actions,
        policy_previous_raw_action=np.concatenate((np.zeros((1, 16, 48)), actions[:1])),
        policy_reference_tick=np.tile(np.arange(2)[:, None]*4, (1, 16)))


@pytest.mark.parametrize("feedforward", [True, False])
def test_learned_action_controller_audit_requires_no_fake_zero_actions(baseline, feedforward):
    bundle, inputs = baseline
    arrays = controller_arrays(bundle, inputs["jab"], feedforward=feedforward)
    result = audit_release_arrays(arrays, bundle=bundle, inputs=inputs["jab"], policy_steps=2, feedforward=feedforward)
    assert result["status"] == "passed", result
    assert "zero_learned_residual" not in result["checks"]
    assert result["physics_audit"]["status"] == "passed"
    assert result["physics_audit"]["zero_residual_fixture_check"] == "not_applicable_to_frozen_actor"


@pytest.mark.parametrize("key,index,value,gate", [
    ("policy_previous_raw_action", (1, 0, 0), 1., "previous_raw_action_semantics"),
    ("pre_actor_action", (2, 0, 0), .1, "actor_held_for_four_physics_steps"),
    ("raw_action", (3, 0, 0), .1, "recorded_actor_reaches_action_manager"),
    ("pre_reference_tick", (3, 0), 2, "preintegration_reference_clock"),
    ("joint_position_target", (1, 0, 0), .5, "joint_target_clipping"),
    ("joint_effort_target", (1, 0, 0), 15., "feedforward_reaches_actuator"),
    ("ctrl", (1, 0, 0), -1000., "native_affine_drive"),
    ("actuator_force", (1, 0, 0), 7., "native_affine_force_law"),
    ("qpos", (0, 0, 0), .7, "fixed_initial_bank_full_pose_and_origin"),
    ("qfrc_applied", (3, 0, 0), 1., "zero_external_generalized_force"),
    ("reset_counts", (3, 0), 2, "one_initial_reset"),
])
def test_controller_and_clock_mutations_fail(baseline, key, index, value, gate):
    bundle, inputs = baseline
    arrays = controller_arrays(bundle, inputs["jab"])
    arrays[key][index] = value
    result = audit_release_arrays(arrays, bundle=bundle, inputs=inputs["jab"], policy_steps=2, feedforward=True)
    assert result["status"] == "failed" and result["checks"][gate] is False


def test_missing_or_nonfinite_actor_evidence_fails(baseline):
    bundle, inputs = baseline
    arrays = controller_arrays(bundle, inputs["jab"])
    del arrays["policy_action"]
    assert audit_release_arrays(arrays, bundle=bundle, inputs=inputs["jab"], policy_steps=2, feedforward=True)["status"] == "failed"
    arrays = controller_arrays(bundle, inputs["jab"])
    arrays["policy_observation"][0, 0, 0] = np.nan
    assert audit_release_arrays(arrays, bundle=bundle, inputs=inputs["jab"], policy_steps=2, feedforward=True)["status"] == "failed"


@pytest.mark.parametrize("field,value", [("diagnostic_scope", "production"), ("production_qualified", True),
                                         ("feedforward_mode", "disabled_zero"), ("release_manifest_sha256", "0"*64)])
def test_mismatched_bundle_scope_is_rejected(baseline, field, value):
    from dataclasses import replace
    bundle, _ = baseline
    other = replace(bundle, manifest={**bundle.manifest, field: value})
    with pytest.raises(ValueError, match="migration bundle"):
        load_release_inputs(other, RELEASE)


def test_allocation_guard_precedes_output_creation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path/"never_created"
    with pytest.raises(RuntimeError, match="existing explicitly allocated"):
        run_release_baseline_probe("missing", "missing", output)
    assert not output.exists()


def test_legacy_runtime_hash_is_mandatory_even_for_loaded_bundle(baseline):
    from dataclasses import replace
    bundle, _ = baseline
    clips = [dict(clip) for clip in bundle.clips]
    clips[0]["legacy_runtime_sha256"] = "0"*64
    with pytest.raises(ValueError, match="Source-grid runtime hash"):
        load_release_inputs(replace(bundle, clips=tuple(clips)), RELEASE)


@pytest.mark.parametrize("solver", ["historical", "accepted"])
def test_compiled_scene_and_solver_parity_without_integrating(baseline, solver, monkeypatch):
    import mujoco
    from mjlab.scene import Scene
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No CPU integration"))
    bundle, inputs = baseline
    cfg = build_release_probe_cfg(bundle, ReleaseRecorder(), inputs["jab"], solver=solver, feedforward=True)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    result = verify_release_model(model, bundle, solver=solver)
    assert result["status"] == "passed", result
    model.body_mass[model.body("robot/Hips").id] += 1.
    with pytest.raises(ValueError, match="body_mass_inertia"):
        verify_release_model(model, bundle, solver=solver)
