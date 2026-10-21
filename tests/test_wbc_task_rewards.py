"""Physical reward invariants, independent of simulator integration."""
import math

import pytest
import torch

from human_punching.wbc.runtime.task_rewards import (
    DEFAULT_WEIGHTS, POLICY_DT, RewardScales, guard_scores, task_reward_rates,
)
from task10_telemetry_helpers import state


def fixture(n=2):
    current = state(n=n, dtype=torch.float32)
    goal = {key: value.clone() for key, value in current.items()}
    window = {key: torch.zeros(n) for key in (
        "root_linear_speed_peak_m_s", "root_angular_speed_peak_rad_s",
        "joint_speed_peak_rad_s", "glove_guard_speed_peak_m_s",
        "joint_limit_excess_rad", "effort_excess_nm",
    )}
    window["double_support_fraction"] = torch.ones(n)
    window["loaded_patch_slip_m"] = torch.zeros(n, 2)
    context = {
        "guard_mask": torch.ones(n, dtype=torch.bool),
        "strike_mask": torch.zeros(n, dtype=torch.bool),
        "new_contact": torch.zeros(n, dtype=torch.bool),
        "new_recovery": torch.zeros(n, dtype=torch.bool),
        "active_hand": torch.zeros(n, dtype=torch.int64),
        "target_w": torch.tensor([[1., 0., 1.]]).expand(n, -1).clone(),
        "ttc_s": torch.full((n,), .7), "elapsed_s": torch.zeros(n),
        "reference_glove_w": current["glove_position_w"].clone(),
    }
    return current, goal, window, context, torch.full((3,), 10.)


def rates(args):
    return task_reward_rates(*args)


def test_nominal_guard_and_sign_equivalent_orientations():
    current, goal, window, context, caps = fixture()
    current["root_quat_wxyz"].neg_()
    current["glove_quat_guard_frame"].neg_()
    scores = guard_scores(current, goal, window)
    assert all(torch.equal(value, torch.ones(2)) for value in scores.values())
    assert set(DEFAULT_WEIGHTS) == set(rates((current, goal, window, context, caps)))


@pytest.mark.parametrize("error_key,output", [
    ("root_position_w", "guard_root_position"),
    ("glove_position_guard_frame", "guard_hands_position"),
    ("support_anchor_position_w", "guard_footprint"),
    ("joint_position", "guard_joint_posture"),
])
def test_guard_errors_are_per_environment_and_reduce_score(error_key, output):
    args = fixture()
    args[0][error_key][0] += .05
    value = rates(args)[output]
    assert 0 < value[0] < value[1] == 1.


@pytest.mark.parametrize("key", ["root_linear_speed_peak_m_s", "root_angular_speed_peak_rad_s",
                                "joint_speed_peak_rad_s", "glove_guard_speed_peak_m_s"])
def test_quiet_final_endpoint_cannot_hide_earlier_velocity_spike(key):
    args = fixture()
    args[2][key][0] = 1.
    assert rates(args)["guard_quiet"][0] < rates(args)["guard_quiet"][1] == 1.
    # Final kinematics remain entirely quiet; the four-tick peak changes reward.
    assert not args[0]["joint_velocity"].any()


def test_heel_height_is_not_translational_step_and_slip_requires_loaded_patch():
    args = fixture()
    args[0]["support_anchor_position_w"][0, :, 2] += .08
    assert torch.equal(rates(args)["guard_footprint"], torch.ones(2))
    # The observer already excludes unloaded anchors. Only its loaded travel
    # enters the reward; arbitrary airborne velocity does not create slip cost.
    args[0]["support_anchor_velocity_w"][0] = 20.
    assert not rates(args)["loaded_slip_cost"].any()
    args[2]["loaded_patch_slip_m"][1, 0] = .003
    assert rates(args)["loaded_slip_cost"][1] > 0


@pytest.mark.parametrize("metric", ["rms", "maximum"])
def test_masks_exclude_guard_shaping_during_outward_strike_and_terminal_rows(metric):
    args = fixture()
    args[3]["guard_mask"][:] = False
    args[3]["strike_mask"][0] = True
    result = task_reward_rates(*args, guard_joint_posture_metric=metric)
    assert all(not value.any() for key, value in result.items() if key.startswith("guard_"))
    assert result["strike_glove_tracking"].tolist() == [1., 0.]
    assert all(value[1] == 0 for value in result.values())


def test_first_contact_uses_5ms_event_time_and_is_not_consumed_on_read():
    args = fixture()
    args[3]["guard_mask"][:] = False
    args[3]["strike_mask"][:] = True
    args[3]["new_contact"][:] = True
    args[3]["elapsed_s"][:] = .72  # final endpoint is 20 ms late
    args[3]["contact_error_s"] = torch.tensor([0., .005])
    a, b = rates(args), rates(args)
    assert a["first_contact_bonus"][0]*POLICY_DT == 1.
    assert torch.isclose(a["first_contact_bonus"][1]*POLICY_DT, torch.tensor(math.exp(-(.005/.02)**2)))
    assert torch.equal(a["first_contact_bonus"], b["first_contact_bonus"])
    del args[3]["contact_error_s"]
    assert not rates(args)["first_contact_bonus"].any()


def test_recovery_bonus_survives_terminal_phase_but_is_counted_once_per_event():
    args = fixture()
    args[3]["guard_mask"][:] = False
    args[3]["new_recovery"][0] = True
    assert (rates(args)["guard_recovered_bonus"]*POLICY_DT).tolist() == [1., 0.]
    args[3]["new_recovery"][:] = False
    assert not rates(args)["guard_recovered_bonus"].any()


def test_inactive_glove_clearance_and_named_active_hand():
    args = fixture()
    args[3]["active_hand"][:] = torch.tensor([0, 1])
    args[0]["glove_position_w"][:, 1] = args[3]["target_w"]
    result = rates(args)
    assert result["inactive_clearance_cost"][0] > 0
    assert result["inactive_clearance_cost"][1] == 0


def test_invalid_prelaunch_hand_never_earns_strike_or_contact_bonus():
    args = fixture()
    args[3]["active_hand"][:] = -1
    args[3]["new_contact"][:] = True
    args[3]["strike_mask"][:] = True
    args[3]["contact_error_s"] = torch.zeros(2)
    result = rates(args)
    assert not result["first_contact_bonus"].any()
    assert not result["strike_glove_tracking"].any()
    assert torch.equal(result["guard_quiet"], torch.ones(2))


def test_substep_limit_peak_counts_even_with_endpoint_inside_limits():
    args = fixture()
    args[2]["joint_limit_excess_rad"][0] = .021
    args[2]["effort_excess_nm"][1] = 2.
    result = rates(args)
    assert result["joint_limit_cost"][0] > 1.
    assert result["effort_limit_cost"].tolist() == [0., 4.]


def test_no_input_mutation_and_nonfinite_world_does_not_poison_batch():
    args = fixture()
    args[0]["root_position_w"][0, 0] = float("nan")
    before = [{k: v.clone() for k, v in a.items()} if isinstance(a, dict) else a.clone() for a in args]
    result = rates(args)
    assert all(torch.isfinite(v).all() for v in result.values())
    assert result["guard_root_position"].tolist() == [0., 1.]
    for original, copy in zip(args, before):
        if isinstance(original, dict):
            for key in original:
                torch.testing.assert_close(original[key], copy[key], equal_nan=True)
        else:
            assert torch.equal(original, copy)


def test_malformed_context_and_scales_fail_closed():
    args = fixture()
    args[3]["strike_mask"] = torch.zeros(2)
    with pytest.raises(ValueError, match="boolean"):
        rates(args)
    with pytest.raises(ValueError, match="positive"):
        RewardScales(timing_s=0.)


def test_maximum_joint_metric_distinguishes_one_outlier_from_equal_rms_guard():
    args = fixture()
    current, goal, window, _, _ = args
    # Same RMS: the first pose has one inadmissible joint, while every joint in
    # the second remains below the unchanged 0.12-rad guard tolerance.
    current["joint_position"][0, 0] = .18
    current["joint_position"][1] = .18 / math.sqrt(3)
    errors = (current["joint_position"]-goal["joint_position"]).abs().amax(-1)
    assert (errors > .12).tolist() == [True, False]
    historical = rates(args)
    explicit_rms = task_reward_rates(*args, guard_joint_posture_metric="rms")
    maximum = task_reward_rates(*args, guard_joint_posture_metric="maximum")
    assert torch.isclose(historical["guard_joint_posture"][0], historical["guard_joint_posture"][1])
    assert all(torch.equal(historical[key], explicit_rms[key]) for key in historical)
    torch.testing.assert_close(maximum["guard_joint_posture"], torch.exp(-(errors/.24).square()))
    assert maximum["guard_joint_posture"][0] < maximum["guard_joint_posture"][1]
    assert all(torch.equal(historical[key], maximum[key]) for key in historical if key != "guard_joint_posture")
    torch.testing.assert_close(
        guard_scores(current, goal, window, guard_joint_posture_metric="maximum")["guard_joint_posture"],
        maximum["guard_joint_posture"])


@pytest.mark.parametrize("metric", [None, True, 1, "max", "mean", []])
def test_invalid_joint_metric_fails_closed_even_with_disabled_masks(metric):
    args = fixture()
    args[3]["guard_mask"].zero_()
    with pytest.raises(ValueError, match="posture metric"):
        task_reward_rates(*args, guard_joint_posture_metric=metric)
    with pytest.raises(ValueError, match="posture metric"):
        guard_scores(*args[:3], guard_joint_posture_metric=metric)
