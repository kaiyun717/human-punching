"""Pure transport algebra and mocked FK/Jacobian tests; no MuJoCo models."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching import phase2_target_path_reference_v2 as path
from human_punching.phase2_target_reference import RetargetSettings, RetargetRejection


NAMES = tuple("Left" + name for name in (
    "Shoulder_x", "Shoulder_y", "Shoulder_z", "Arm_x", "Arm_y", "Arm_z",
    "ForeArm_z", "ForeArm_x", "Hand_x", "Hand_y", "Hand_z"))


def settings():
    return path.SequentialSettings(RetargetSettings("Left", NAMES, maximum_iterations=80))


@pytest.mark.parametrize("change", [dict(maximum_iterations=40), dict(maximum_joint_delta_rad=.21),
    dict(maximum_iteration_delta_rad=.03), dict(active_position_tolerance_m=.0006),
    dict(maximum_target_delta_m=.13), dict(optimized_joint_names=NAMES[:-1])])
def test_recipe_cannot_silently_change_reviewed_bounds(change):
    with pytest.raises(ValueError):
        path.SequentialSettings(replace(settings().base, **change))


def test_transport_preserves_normalized_correction_and_clips_joint_bounds():
    previous = np.array([.05, -.025])
    np.testing.assert_allclose(path.transported_preference(previous, .5, .8, [-.16, -.16], [.16, .16]), [.08, -.04])
    np.testing.assert_allclose(path.transported_preference(previous, .5, .8, [-.01, -.01], [.01, .01]), [.01, -.01])
    np.testing.assert_array_equal(path.transported_preference(previous, 0, .8, [-.16, -.16], [.16, .16]), [0, 0])
    np.testing.assert_array_equal(path.transported_preference(previous, .5, 0, [0, 0], [0, 0]), [0, 0])


def test_transport_remains_bounded_near_zero_envelope():
    correction = path.transported_preference([1e-10], 1e-12, .1, [-.02], [.02])
    np.testing.assert_allclose(correction, [.02])
    for e in [-.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            path.transported_preference([.01], .5, e, [-.1], [.1])


@pytest.fixture
def mock_problem(monkeypatch):
    model = SimpleNamespace(nq=22, nv=21, jnt_range=np.tile([-2., 2.], (11, 1)))
    qa, va = np.arange(7, 18), np.arange(6, 17)
    scale = [1.]
    def data_factory(_):
        return SimpleNamespace(site_xpos=np.zeros((2, 3)), site_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)))
    def kinematics(_, data, pose):
        data.site_xpos[0] = scale[0] * pose[7:10]
        data.site_xpos[1] = [.1, -.2, .3]
    def jacobian(_, data, jp, jr, site):
        jp[:] = 0; jr[:] = 0
        if site == 0:
            jp[:, 6:9] = scale[0] * np.eye(3)
    def velocities(_, t, q):
        v = np.zeros((len(t), model.nv)); v[:, 6:] = np.gradient(q[:, 7:], t, axis=0)
        return v
    monkeypatch.setattr(path.mujoco, "MjData", data_factory)
    monkeypatch.setattr(path, "_kinematics", kinematics)
    monkeypatch.setattr(path.mujoco, "mj_jacSite", jacobian)
    monkeypatch.setattr(path, "resolve_joint_mask", lambda *_: dict(joint_ids=np.arange(11),
        qpos_addresses=qa, dof_addresses=va, active_site=0, inactive_site=1))
    monkeypatch.setattr(path, "tangent_velocities", velocities)
    t = np.arange(81) * .005
    q = np.zeros((len(t), model.nq)); q[:, 3] = 1; q[:, 18:] = [.1, .2, -.1, -.2]
    source = dict(time_s=t, qpos=q, head_center=np.zeros(3), active_hand_index=0, guard_hold_start=.36)
    return model, source, scale


def solve(problem, delta=(.01, 0, 0), **kwargs):
    model, source, _ = problem
    return path.retarget_sequential(model, source, delta, source_outward_start_s=.05,
        source_outward_end_s=.35, settings=settings(), **kwargs)


def test_mocked_candidate_preserves_protected_state_and_zero_envelope_holds(mock_problem):
    _, source, _ = mock_problem
    before = source["qpos"].copy(); candidate, receipt = solve(mock_problem)
    fixed = np.r_[0:7, 18:22]
    np.testing.assert_array_equal(candidate["qpos"][:, fixed], before[:, fixed])
    hold = (source["time_s"] <= .05) | (source["time_s"] >= .35)
    np.testing.assert_array_equal(candidate["qpos"][hold], before[hold])
    np.testing.assert_array_equal(source["qpos"], before)
    assert any(frame["warm_start_used"] for frame in receipt["frame_geometry"])
    assert max(frame["active_error_m"] for frame in receipt["frame_geometry"]) <= .0005
    assert receipt["maximum_joint_correction_rad"] <= .2
    assert receipt["algorithm_version"] == path.VERSION and receipt["physical_qualification"] is False
    assert receipt["feedforward_recomputed"] is False
    np.testing.assert_allclose(candidate["qacc"], np.gradient(candidate["qvel"], source["time_s"], axis=0, edge_order=2))
    assert receipt["maximum_5ms_correction_step_rad"] == np.abs(np.diff(candidate["qpos"][:, 7:18]-before[:, 7:18], axis=0)).max()
    steps = np.r_[0., np.abs(np.diff(candidate["qpos"][:, 7:18]-before[:, 7:18], axis=0)).max(axis=1)]
    np.testing.assert_allclose([frame["correction_step_rad"] for frame in receipt["frame_geometry"]], steps)
    assert len(receipt["envelope_transition_neighborhoods"]) == 4
    assert all(event["sample_indices"] for event in receipt["envelope_transition_neighborhoods"])
    assert receipt["peak_optimized_acceleration_joint"] in NAMES


def test_mocked_zero_target_is_exact_source_fixed_point(mock_problem):
    candidate, receipt = solve(mock_problem, delta=(0, 0, 0))
    np.testing.assert_array_equal(candidate["qpos"], mock_problem[1]["qpos"])
    assert receipt["maximum_joint_correction_rad"] == 0


def test_unsolved_bound_keeps_iteration_residual_and_saturation_evidence(mock_problem):
    mock_problem[2][0] = .01
    with pytest.raises(RetargetRejection) as caught:
        solve(mock_problem, delta=(.03, 0, 0))
    d = caught.value.disposition
    assert d["reason"] == "sequential_upper_body_geometry_unsolved"
    assert d["iterations"] == 80 and d["exit_reason"] == "iteration_cap"
    assert len(d["active_residual_history_m"]) == 81 and d["active_error_m"] > .0005
    assert "LeftShoulder_x" in d["saturated_joint_names"]
    assert len(d["final_correction_rad"]) == 11 and d["physical_qualification"] is False


def test_deadline_callback_reaches_inside_solver_iterations(mock_problem):
    mock_problem[2][0] = .01
    count = [0]
    def stop():
        count[0] += 1
        if count[0] == 30:
            raise RuntimeError("synthetic deadline")
    with pytest.raises(RuntimeError, match="deadline"):
        solve(mock_problem, delta=(.03, 0, 0), checkpoint=stop)
    assert count[0] == 30


def test_moving_preparation_is_rejected_without_model_execution(mock_problem):
    mock_problem[1]["qpos"][1, 7] = .01
    with pytest.raises(RetargetRejection, match="moving_preparation"):
        solve(mock_problem)
