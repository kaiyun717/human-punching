"""Synthetic recorded-state fixtures exercise evaluator gates, not physical feasibility."""

import mujoco
import numpy as np
import pytest

from human_punching.jab_evaluation import (
    _interpolated_contact_time,
    _simulation_clock_evidence,
    chest_relative_site_state,
    evaluate_rollout,
    trailing_dwell,
)


@pytest.fixture
def trial():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.1"/><worldbody>
      <body name="Hips" pos="0 0 1"><freejoint name="root"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="LeftFoot_sole" pos="0 0.1 -1"/>
        <site name="RightFoot_sole" pos="0 -0.1 -1"/>
        <body name="Chest" pos="0 0 0.5">
          <site name="RightHand_center" pos="-0.4 0 0"/>
        <body><joint name="arm" axis="0 0 1" limited="true" range="-120 120"/>
          <geom type="sphere" size="0.08" pos="0.4 0 0" mass="0.1"/>
          <site name="LeftHand_center" pos="0.4 0 0"/>
        </body></body>
      </body>
    </worldbody><actuator><motor joint="arm" ctrllimited="true" ctrlrange="-10 10"/></actuator></mujoco>
    """)
    count = 11
    qpos = np.tile(model.qpos0, (count, 1))
    qpos[3:6, 7] = [0.7, np.pi / 2, 0.7]
    arrays = {
        "time_s": np.arange(count) * model.opt.timestep,
        "qpos": qpos,
        "qvel": np.zeros((count, model.nv)),
        "ctrl": np.zeros((count, model.nu)),
        "qfrc_actuator": np.zeros((count, model.nv)),
        "qfrc_applied": np.zeros((count, model.nv)),
        "xfrc_applied": np.zeros((count, model.nbody, 6)),
        "support_contact": np.ones((count, 2), dtype=bool),
        "head_center_w": np.array([0, 0.4, 1.5]),
        "state_reset_count": 1,
        "physics_step_count": count - 1,
        "evaluation_kind": "synthetic_test",
    }
    reference = {name: np.copy(arrays[name]) for name in ("time_s", "qpos", "qvel", "head_center_w")}
    reference.update(launch_time=0.2, return_start=0.6)
    return model, arrays, reference


def test_trailing_dwell_counts_time_not_samples():
    assert trailing_dwell([0, 0.1, 0.3, 0.4], [True, False, True, True]) == pytest.approx(0.1)
    assert trailing_dwell([0, 0.1], [False, True]) == 0
    assert trailing_dwell([0, 0.1], [True, False]) == 0
    with pytest.raises(ValueError):
        trailing_dwell([0, 0], [True, True])


def test_complete_trial_contact_and_recovery(trial):
    model, arrays, reference = trial
    metrics = evaluate_rollout(model, arrays, reference)
    assert metrics["pilot_pass"]
    assert metrics["first_sampled_contact_after_reset_s"] == pytest.approx(0.4)
    assert metrics["final_guard"]["trailing_recovered_dwell_s"] == pytest.approx(0.4)
    assert metrics["reference_tracking"]["glove_position_rms_m"] == pytest.approx([0, 0])
    assert metrics["root_tilt_max_rad"] == pytest.approx(0)


def test_missing_physics_evidence_cannot_pass(trial):
    model, arrays, reference = trial
    for key in ("support_contact", "xfrc_applied", "state_reset_count"):
        arrays.pop(key)
    result = evaluate_rollout(model, arrays, reference)
    assert not result["pilot_pass"]
    assert not result["acceptance_checks"]["no_external_forces"]
    assert not result["acceptance_checks"]["guard_recovered_with_recorded_support"]
    assert not result["acceptance_checks"]["one_guard_reset_and_every_physics_step_recorded"]
    assert result["recorded_double_support_fraction"] is None


def test_late_root_force_and_oversized_control_fail(trial):
    model, arrays, reference = trial
    arrays["qfrc_actuator"][-1, 2] = 0.01
    arrays["ctrl"][3, 0] = 10.2
    result = evaluate_rollout(model, arrays, reference)
    assert not result["acceptance_checks"]["unactuated_root"]
    assert not result["acceptance_checks"]["bounded_recorded_control"]
    assert result["control_limit_violation_max"] == pytest.approx(0.2)


def test_motion_must_finish_in_slow_guard_after_contact(trial):
    model, arrays, reference = trial
    arrays["qvel"][-1, 6] = 1.0
    result = evaluate_rollout(model, arrays, reference)
    assert result["final_guard"]["trailing_recovered_dwell_s"] == 0
    assert not result["acceptance_checks"]["guard_recovered_with_recorded_support"]


def test_no_contact_is_not_a_successful_guard_return(trial):
    model, arrays, reference = trial
    result = evaluate_rollout(model, arrays, reference, head_center=[10, 0, 1.5])
    assert not result["any_glove_contact"]
    assert result["final_guard"]["trailing_recovered_dwell_s"] == 0
    assert not result["acceptance_checks"]["guard_recovered_with_recorded_support"]


def test_nonfinite_and_incorrect_timestamps_rejected(trial):
    model, arrays, reference = trial
    arrays["qpos"][2, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate_rollout(model, arrays, reference)
    arrays["qpos"][2, 0] = 0
    arrays["time_s"][2] = arrays["time_s"][1]
    with pytest.raises(ValueError, match="increase strictly"):
        evaluate_rollout(model, arrays, reference)


def test_downsampled_recording_fails_substep_evidence(trial):
    model, arrays, reference = trial
    arrays["time_s"] *= 2
    result = evaluate_rollout(model, arrays, reference)
    assert not result["recording_at_every_model_step"]
    assert not result["acceptance_checks"]["one_guard_reset_and_every_physics_step_recorded"]


def test_recorded_runtime_timestep_and_reference_speed(trial):
    model, arrays, reference = trial
    arrays["time_s"] *= 2
    arrays["physics_timestep_s"] = 0.2
    arrays["reference_speed"] = 0.5
    result = evaluate_rollout(model, arrays, reference)
    assert result["pilot_pass"]
    assert result["model_xml_timestep_s"] == pytest.approx(0.1)
    assert result["evaluated_physics_timestep_s"] == pytest.approx(0.2)
    assert result["reference_tracking"]["glove_position_rms_m"] == pytest.approx([0, 0])


def test_actual_torque_can_fail_even_with_bounded_control(trial):
    model, arrays, reference = trial
    arrays["qfrc_actuator"][3, 6] = 11
    result = evaluate_rollout(model, arrays, reference)
    assert result["acceptance_checks"]["bounded_recorded_control"]
    assert not result["acceptance_checks"]["bounded_recorded_joint_torque"]
    assert result["actual_joint_torque_limit_violation_max_nm"] == pytest.approx(1)


def test_relative_hand_velocity_removes_rotating_translating_chest():
    rotation = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    center = np.array([[2., -3., 1.]])
    linear = np.array([[1., 2., -0.3]])
    angular = np.array([[0., 0., 2.]])
    relative_position = np.array([[[0.4, 0.2, 0.1], [-0.4, 0.1, 0.2]]])
    offset = np.einsum("...ij,...kj->...ki", rotation, relative_position)
    world_velocity = linear[:, None, :] + np.cross(angular[:, None, :], offset)
    state = chest_relative_site_state(center, rotation, linear, angular,
                                      center[:, None, :] + offset,
                                      np.broadcast_to(rotation[:, None], (1, 2, 3, 3)),
                                      world_velocity)
    np.testing.assert_allclose(state["position"], relative_position, atol=1e-12)
    np.testing.assert_allclose(state["rotation"], np.broadcast_to(np.eye(3), (1, 2, 3, 3)))
    np.testing.assert_allclose(state["linear_velocity"], 0, atol=1e-12)


def test_small_joint_error_can_still_fail_chest_relative_glove_guard(trial):
    model, arrays, reference = trial
    # 0.10 rad passes the joint guard threshold but moves this 0.4 m hand by 4 cm.
    arrays["qpos"][-3:, 7] = 0.10
    result = evaluate_rollout(model, arrays, reference)
    assert result["final_guard"]["joint_position_error_max_rad"] < 0.12
    assert result["final_guard"]["glove_position_in_chest_error_m"][0] > 0.03
    assert not result["acceptance_checks"]["guard_recovered_with_recorded_support"]


def test_mujoco_jacobian_relative_velocity_cancels_rigid_root_motion(trial):
    model, arrays, reference = trial
    arrays["qvel"][-1, :6] = [1, 2, -0.3, 0, 0, 2]
    result = evaluate_rollout(model, arrays, reference)
    assert result["final_guard"]["root_linear_speed_m_s"] > 1
    np.testing.assert_allclose(result["final_guard"]["glove_relative_linear_speed_m_s"], 0, atol=1e-12)


def test_late_return_contact_does_not_satisfy_attack_gate(trial):
    model, arrays, reference = trial
    result = evaluate_rollout(model, arrays, reference, attack_window_reference_s=[0.1, 0.3])
    assert result["any_glove_contact"]
    assert not result["attack_window_glove_contact"]
    assert not result["acceptance_checks"]["attack_window_contact"]


def test_missing_attack_annotation_cannot_pass(trial):
    model, arrays, reference = trial
    reference.pop("return_start")
    result = evaluate_rollout(model, arrays, reference)
    assert result["any_glove_contact"]
    assert not result["acceptance_checks"]["attack_window_contact"]
    assert result["attack_window_after_reset_s"] is None


def test_native_affine_pd_audits_output_force_not_drive(trial):
    model, arrays, reference = trial
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[0, :3] = [0, -100, -10]
    model.actuator_ctrllimited[:] = False
    model.actuator_forcelimited[:] = True
    model.actuator_forcerange[0] = [-10, 10]
    # A target-drive value can exceed the torque cap while its force is bounded.
    arrays["ctrl"][:] = 1000
    arrays["actuator_force"] = np.full((len(arrays["time_s"]), 1), 9.0)
    arrays["qfrc_actuator"][:, 6] = 9
    result = evaluate_rollout(model, arrays, reference)
    assert result["pilot_pass"]
    assert result["actuator_control_kind"] == "native_affine_pd_force_limited"
    assert result["actual_joint_torque_max_nm"] == pytest.approx(9)
    arrays["actuator_force"][3, 0] = 11
    arrays["qfrc_actuator"][3, 6] = 11
    result = evaluate_rollout(model, arrays, reference)
    assert not result["acceptance_checks"]["bounded_recorded_control"]
    assert not result["acceptance_checks"]["bounded_recorded_joint_torque"]


def test_affine_pd_requires_recorded_force_and_consistent_transmission(trial):
    model, arrays, reference = trial
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_ctrllimited[:] = False
    model.actuator_forcelimited[:] = True
    model.actuator_forcerange[0] = [-10, 10]
    model.actuator_gear[0, 0] = 2
    result = evaluate_rollout(model, arrays, reference)
    assert not result["acceptance_checks"]["bounded_recorded_control"]
    assert not result["acceptance_checks"]["recorded_actuator_force_matches_joint_torque"]
    arrays["actuator_force"] = np.full((len(arrays["time_s"]), 1), 9.0)
    arrays["qfrc_actuator"][:, 6] = 18
    assert evaluate_rollout(model, arrays, reference)["pilot_pass"]
    arrays["qfrc_actuator"][3, 6] = 17
    result = evaluate_rollout(model, arrays, reference)
    assert result["acceptance_checks"]["bounded_recorded_joint_torque"]
    assert not result["acceptance_checks"]["recorded_actuator_force_matches_joint_torque"]


def test_required_five_ms_clock_checks_runtime_and_actual_policy_updates(trial):
    model, arrays, reference = trial
    model.opt.timestep = 0.005
    arrays["time_s"] = np.arange(len(arrays["time_s"])) * .005
    arrays["physics_timestep_s"] = .005
    arrays["simulation_time_s"] = arrays["time_s"].copy()
    arrays["policy_timestep_s"] = .020
    arrays["policy_update_step_indices"] = np.array([0, 4, 8])
    kwargs = dict(required_physics_timestep_s=.005, required_policy_timestep_s=.020)
    result = evaluate_rollout(model, arrays, reference, **kwargs)
    assert result["acceptance_checks"]["required_physics_timestep"]
    assert result["acceptance_checks"]["required_policy_update_interval"]
    model.opt.timestep = .002
    arrays["policy_update_step_indices"] = np.array([0, 2, 4, 6, 8])
    result = evaluate_rollout(model, arrays, reference, **kwargs)
    assert not result["acceptance_checks"]["required_physics_timestep"]
    assert not result["acceptance_checks"]["required_policy_update_interval"]


def test_nominal_five_ms_labels_do_not_hide_wrong_simulator_clock(trial):
    model, arrays, reference = trial
    model.opt.timestep = .005
    arrays["time_s"] = np.arange(len(arrays["time_s"])) * .005
    arrays["physics_timestep_s"] = .005
    arrays["simulation_time_s"] = np.arange(len(arrays["time_s"])) * .002
    result = evaluate_rollout(model, arrays, reference, required_physics_timestep_s=.005)
    assert result["recording_at_every_model_step"]
    assert not result["acceptance_checks"]["required_physics_timestep"]
    assert not result["simulator_clock_evidence"]["verified"]


def test_contact_timing_uses_annotated_active_hand_and_effective_speed(trial):
    model, arrays, reference = trial
    reference.update(punch_family="left_hook", active_hand="Left", source_speed=.8)
    result = evaluate_rollout(model, arrays, reference)
    assert result["punch_family"] == "left_hook"
    assert result["effective_source_speed"] == pytest.approx(.8)
    assert result["contact_timing"]["signed_first_contact_error_s"] == 0
    assert result["contact_timing"]["signed_interpolated_contact_error_s"] == 0
    assert result["precision_targets"]["contact_timing_target_met"]
    assert result["reference_tracking"]["active_glove_position_rms_m"] == 0
    reference["active_hand"] = "Right"
    result = evaluate_rollout(model, arrays, reference)
    assert result["any_glove_contact"]
    assert not result["acceptance_checks"]["active_hand_attack_contact"]
    assert result["contact_timing"]["actual_first_attack_contact_after_reset_s"] is None


def test_contact_interpolation_respects_threshold_and_attack_boundary():
    times = np.array([2.0, 2.005, 2.010])
    distances = np.array([.20, .15, .13])
    assert _interpolated_contact_time(times, distances, np.ones(3, bool)) == pytest.approx(.0075)
    # An excluded earlier sample cannot invent a contact before the attack window.
    assert _interpolated_contact_time(times, distances, np.array([False, False, True])) == pytest.approx(.010)
    assert _interpolated_contact_time(times, np.ones(3), np.ones(3, bool)) is None


def test_declared_native_controller_cannot_pass_with_torque_motor_xml(trial):
    model, arrays, reference = trial
    arrays["controller_mode"] = "implicit_pd"
    result = evaluate_rollout(model, arrays, reference)
    assert not result["acceptance_checks"]["recorded_controller_matches_compiled_actuator"]


def test_actual_float32_simulator_clock_roundoff_is_distinguished_from_substeps():
    nominal = np.arange(1201) * .005
    actual = np.concatenate([np.zeros(1, np.float32),
                             np.cumsum(np.full(1200, .005, np.float32), dtype=np.float32)])
    result = _simulation_clock_evidence({"simulation_time_s": actual}, nominal, .005)
    assert result["verified"]
    assert result["maximum_elapsed_clock_error_s"] > 1e-6
    assert not _simulation_clock_evidence({"simulation_time_s": actual * .4}, nominal, .005)["verified"]


def test_nonuniform_retiming_does_not_claim_one_uniform_source_speed(trial):
    model, arrays, reference = trial
    reference.update(active_hand="Left", source_speed=1.0,
                     timing_kind="nonuniform_retimed", local_source_speed_min=.4,
                     local_source_speed_max=1.0, retiming_duration_ratio=1.5)
    result = evaluate_rollout(model, arrays, reference)
    assert result["effective_source_speed"] is None
    assert result["reference_timing"]["local_source_speed_min"] == .4
    assert result["reference_timing"]["retiming_duration_ratio"] == 1.5
