"""CPU-only recorder/config tests; no native GPU probe is executed here."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching.wbc.native_probe import (
    RAW_FIELDS, NativeProbeRecorder, _gloves_from_qpos, audit_probe_arrays,
    build_probe_cfg, run_native_probe,
)


def raw_fixture():
    count, envs = 9, 2
    qpos = np.zeros((count, envs, 55))
    qpos[..., 2:4] = [.9, 1.]
    return {
        "time": np.tile((np.arange(count)*.005)[:, None], (1, envs)).astype(np.float32),
        "qpos": qpos, "qvel": np.zeros((count, envs, 54)),
        "ctrl": np.zeros((count, envs, 48)),
        "actuator_force": np.zeros((count, envs, 48)),
        "qfrc_actuator": np.zeros((count, envs, 54)),
        "qfrc_applied": np.zeros((count, envs, 54)),
        "xfrc_applied": np.zeros((count, envs, 24, 6)),
        "overflow": np.zeros((count, envs), int),
        "reference_physics_steps": np.tile(np.arange(count)[:, None], (1, envs)),
        "reset_counts": np.ones((count, envs), int), "sim_step_index": np.arange(count),
        "policy_step_index": np.maximum(np.arange(count) - 1, 0) // 4,
        "raw_action": np.zeros((count, envs, 48)),
        "trajectory_id": np.tile(np.arange(envs), (count, 1)),
        "joint_position_target": np.zeros((count, envs, 48)),
        "joint_velocity_target": np.zeros((count, envs, 48)),
        "joint_effort_target": np.zeros((count, envs, 48)),
    }


def audit(arrays):
    return audit_probe_arrays(arrays, policy_steps=2, physics_dt_s=.005,
                              effort_limits=np.full(48, 20.))


def test_audit_clocks_forces_and_untrained_fall_diagnostic():
    arrays = raw_fixture()
    assert audit(arrays)["status"] == "passed"
    arrays["qpos"][4:, 0, 2] = .2
    result = audit(arrays)
    assert result["status"] == "passed"
    assert result["fall_diagnostic_environments"] == 1
    assert not result["fall_diagnostic_is_acceptance_gate"]


@pytest.mark.parametrize("name,index,value,check", [
    ("actuator_force", (1, 0, 0), 20.1, "actual_actuator_force_caps"),
    ("qfrc_actuator", (1, 0, 0), 1., "unactuated_root"),
    ("xfrc_applied", (1, 0, 3, 0), 1., "zero_external_body_wrench"),
    ("qfrc_applied", (1, 0, 7), 1., "zero_external_generalized_force"),
    ("reference_physics_steps", (2, 0), 1, "one_reference_sample_per_physics_step"),
    ("reset_counts", (2, 0), 2, "one_initial_reset"),
    ("time", (2, 0), .011, "physics_clock"),
    ("overflow", (2, 0), 1, "no_native_overflow"),
    ("policy_step_index", (2,), 1, "policy_counter"),
    ("raw_action", (2, 0, 0), .1, "zero_learned_residual"),
    ("trajectory_id", (2, 0), 1, "committed_trajectory_id"),
])
def test_audit_rejects_actual_evidence_violations(name, index, value, check):
    arrays = raw_fixture()
    arrays[name][index] = value
    result = audit(arrays)
    assert result["status"] == "failed"
    assert not result["checks"][check]


def test_truncated_trace_and_nonfinite_arrays_do_not_pass():
    arrays = raw_fixture()
    arrays = {key: value[:-1] for key, value in arrays.items()}
    assert audit(arrays)["status"] == "failed"
    arrays = raw_fixture()
    arrays["qvel"][4, 0, 1] = float("nan")
    assert not audit(arrays)["checks"]["finite"]


def test_missing_controller_data_and_malformed_state_do_not_pass():
    arrays = raw_fixture()
    del arrays["joint_effort_target"]
    assert audit(arrays)["missing_fields"] == ["joint_effort_target"]
    arrays = raw_fixture()
    arrays["qvel"] = arrays["qvel"][..., :-1]
    assert not audit(arrays)["checks"]["recorded_state_and_controller_shapes"]


def test_recorder_copies_raw_data_and_supported_reset_callback():
    torch = pytest.importorskip("torch")
    arrays = raw_fixture()
    source = SimpleNamespace(**{key: torch.as_tensor(arrays[key][0].copy()) for key in RAW_FIELDS})
    command = SimpleNamespace(physics_steps=torch.zeros(2, dtype=torch.long),
                              trajectory_ids=torch.tensor([0, 1]))
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 48),
        joint_vel_target=torch.zeros(2, 48), joint_effort_target=torch.zeros(2, 48)))
    env = SimpleNamespace(sim=SimpleNamespace(data=source), scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(action=torch.zeros(2, 48)),
        _sim_step_counter=0, common_step_counter=0, num_envs=2, device="cpu")
    recorder = NativeProbeRecorder(2)
    callback = recorder.metric_function()
    assert deepcopy(callback) is callback
    callback.reset(torch.tensor([0, 1]))
    assert callback(env).shape == (2,)
    source.qpos[:, 2] = 0
    assert np.all(recorder.arrays()["qpos"][0, :, 2] == .9)
    assert np.all(recorder.arrays()["reset_counts"] == 1)
    callback.reset(slice(0, 1))
    assert np.array_equal(recorder.reset_counts, [2, 1])
    callback.reset()
    assert np.array_equal(recorder.reset_counts, [3, 2])


def test_pinned_metrics_manager_calls_reset_and_exactly_one_record_per_substep():
    torch = pytest.importorskip("torch")
    from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg

    arrays = raw_fixture()
    source = SimpleNamespace(**{key: torch.as_tensor(arrays[key][0].copy()) for key in RAW_FIELDS})
    command = SimpleNamespace(physics_steps=torch.zeros(2, dtype=torch.long),
                              trajectory_ids=torch.tensor([0, 1]))
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 48),
        joint_vel_target=torch.zeros(2, 48), joint_effort_target=torch.zeros(2, 48)))
    env = SimpleNamespace(sim=SimpleNamespace(data=source), scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(action=torch.zeros(2, 48)),
        _sim_step_counter=0, common_step_counter=0, num_envs=2, device="cpu")
    recorder = NativeProbeRecorder(2)
    manager = MetricsManager({"probe": MetricsTermCfg(
        func=recorder.metric_function(), per_substep=True)}, env)
    manager.reset()  # The real manager passes slice(None) to the hook.
    assert np.array_equal(recorder.reset_counts, [1, 1])
    for _ in range(4):
        manager.compute_substep()
    manager.compute()
    assert len(recorder.rows) == 4


def test_offline_glove_fk_uses_recorded_qpos_without_a_physics_step(monkeypatch):
    import mujoco

    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("FK cannot run dynamics"))
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot/Hips"><freejoint/><geom type="sphere" size=".1"/>
        <site name="robot/LeftHand_center" pos="0 .2 0"/>
        <site name="robot/RightHand_center" pos="0 -.2 0"/>
      </body></worldbody></mujoco>""")
    qpos = np.zeros((2, 1, 7))
    qpos[..., 3] = 1.
    qpos[1, 0, :3] = [1., 2., 3.]
    points = _gloves_from_qpos(model, qpos)
    assert np.allclose(points[1, 0], [[1., 2.2, 3.], [1., 1.8, 3.]])
    assert np.allclose(points[0, 0], [[0., .2, 0.], [0., -.2, 0.]])


def test_allocation_guard_precedes_output_creation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    output = tmp_path / "never-created"
    with pytest.raises(RuntimeError, match="existing explicitly allocated"):
        run_native_probe("missing-bundle", output)
    assert not output.exists()


def test_real_configuration_adds_recorder_without_physics(monkeypatch, tmp_path):
    pytest.importorskip("wbc_mjlab")
    import mujoco
    from human_punching.wbc.references import PROJECT_ROOT, convert_release
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("CPU tests cannot run dynamics"))
    bundle = tmp_path / "converted"
    convert_release(PROJECT_ROOT / "outputs/four-punch-milestone", bundle)
    recorder = NativeProbeRecorder(16)
    cfg = build_probe_cfg(bundle, recorder)
    assert cfg.scene.num_envs == 16 and cfg.decimation == 4
    assert not cfg.auto_reset and not cfg.terminations and not cfg.rewards
    assert cfg.metrics["native_probe"].per_substep
    assert callable(cfg.metrics["native_probe"].func.reset)
