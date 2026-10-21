"""Synthetic evidence validates metric logic, never physical performance."""

from dataclasses import replace

import numpy as np
import pytest

from human_punching.wbc.contracts import LaunchSnapshot, PunchRequest
from human_punching.wbc.evaluation import EpisodeResult, evaluate_episode, summarize_results


@pytest.fixture
def trial():
    qpos = np.zeros(36)
    qpos[2:4] = [.75, 1]
    snap = LaunchSnapshot("sample", qpos, np.zeros(35), (0, 0, 1.23), (0, 0, 0),
                          (.75, 0, 0), (0, 0, 0, 1), "command:42")
    request = PunchRequest("jab", .6, snap, "g0")
    times = np.arange(381)*.005
    n = len(times)
    fraction = np.interp(times, [0, .1, .6, .8, 1.9], [0, 0, 1, 0, 0])
    gloves = np.tile([[.35, .10, 1.23], [.35, -.10, 1.23]], (n, 1, 1))
    gloves[:, 0, 0] -= .21*fraction
    gloves[:, 0, 1] -= .10*fraction
    # At the exact contact tick use the geometric inclusive boundary directly.
    gloves[120, 0] = [.14, 0, 1.23]
    arrays = {
        "time_s": times, "glove_position_w": gloves,
        "reference_glove_position_w": gloves.copy(),
        "root_position_w": np.tile([.75, 0, .9], (n, 1)),
        "root_quat_wxyz": np.tile([1., 0, 0, 0], (n, 1)),
        "root_linear_velocity_w": np.zeros((n, 3)),
        "root_angular_velocity_w": np.zeros((n, 3)),
        "glove_position_guard_frame": gloves - np.array([.75, 0, .9]),
        "glove_quat_guard_frame": np.tile([1., 0, 0, 0], (n, 2, 1)),
        "glove_velocity_guard_frame": np.gradient(gloves, .005, axis=0),
        "support_anchor_position_w": np.tile([[.75, .2, 0], [.75, -.2, 0]], (n, 1, 1)),
        "support_patch_contact": np.ones((n, 2), bool),
        "support_contact": np.ones((n, 2), bool),
        "external_wrench": np.zeros((n, 3, 6)), "root_actuator_force": np.zeros((n, 6)),
        "joint_position": fraction[:, None]*np.array([[.5, .2]]),
        "joint_velocity": np.gradient(fraction, .005)[:, None]*np.array([[.5, .2]]),
        "joint_limits": np.array([[-1, 1], [-1, 1]]),
        "effort": np.zeros((n, 2)), "effort_limits": np.array([20., 20.]),
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4,
        "integration_substeps": 1, "physics_step_count": n-1, "outward_end_s": .65,
        "inter_character_collision_enabled": False, "evidence_kind": "synthetic_test",
        "admission_passed": True, "domain_version": "test-fixture",
    }
    for key in ("external_support", "reset_event", "forbidden_collision", "human_g1_collision",
                "numerical_failure", "contact_overflow"):
        arrays[key] = np.zeros(n, bool)
    arrays["reset_event"][0] = True
    return request, arrays


def evaluate(trial):
    request, arrays = trial
    return evaluate_episode(request, arrays, training_seed=1, recipe_id="r1", spatial_cell_id="s0",
                            ttc_cell_id="t0")


def test_full_cycle_and_inclusive_contact(trial):
    result = evaluate(trial)
    assert result.complete_cycle_success, result
    assert not result.acceptance_eligible  # Synthetic fixtures cannot become acceptance evidence.
    assert result.metrics["first_active_contact_s"] == pytest.approx(.6)
    assert result.metrics["guard_dwell_s"] >= 1
    assert result.metrics["outward_glove_rms_m"] == 0


def test_npz_scalar_shape_is_decoded_without_relaxing_boolean_evidence(trial):
    request, arrays = trial
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    assert evaluate((request, arrays)).complete_cycle_success
    arrays["inter_character_collision_enabled"] = np.asarray(0)
    assert evaluate((request, arrays)).failure_categories == ("malformed_evidence",)


@pytest.mark.parametrize("key", ["external_wrench", "support_contact", "reset_event", "effort_limits"])
def test_missing_evidence_never_passes(trial, key):
    trial[1].pop(key)
    result = evaluate(trial)
    assert not result.complete_cycle_success
    assert result.failure_categories == ("missing_evidence",)


@pytest.mark.parametrize("field,index,value,category", [
    ("external_wrench", (250, 2, 0), 1., "external_force"),
    ("root_actuator_force", (250, 0), 1., "actuated_root"),
    ("reset_event", 200, True, "mid_cycle_reset"),
    ("external_support", 200, True, "external_support"),
    ("effort", (250, 0), 21., "effort_limit"),
    ("joint_position", (250, 0), 1.1, "joint_limit"),
    ("forbidden_collision", 250, True, "forbidden_collision"),
    ("contact_overflow", 250, True, "contact_overflow"),
])
def test_late_failures_cannot_be_hidden_by_earlier_contact(trial, field, index, value, category):
    trial[1][field][index] = value
    result = evaluate(trial)
    assert not result.complete_cycle_success
    assert category in result.failure_categories
    assert result.metrics["first_active_contact_s"] == pytest.approx(.6)


def test_inactive_hazard_and_first_event(trial):
    trial[1]["glove_position_w"][50, 1] = [0, 0, 1.23]
    result = evaluate(trial)
    assert not result.complete_cycle_success
    assert result.first_event == "inactive_glove_contact"
    assert result.first_event_time_s == .25


def test_timing_is_measured_not_playback_multiplier(trial):
    req, arrays = trial
    result = evaluate((replace(req, ttc_s=.63), arrays))
    assert "contact_timing" in result.failure_categories
    arrays["time_s"][50] += .001
    assert evaluate(trial).failure_categories == ("malformed_evidence",)


def test_no_hidden_steps_or_pose_replacement(trial):
    trial[1]["integration_substeps"] = 5
    assert evaluate(trial).failure_categories == ("malformed_evidence",)


def test_contact_on_return_and_force_cancellation_cannot_pass(trial):
    trial[1]["outward_end_s"] = .5
    assert "contact_outside_outward" in evaluate(trial).failure_categories
    trial[1]["external_wrench"][250, :2, 0] = [1., -1.]
    assert "external_force" in evaluate(trial).failure_categories


def test_declared_numeric_limit_tolerances(trial):
    trial[1]["joint_position"][250, 0] = 1.019
    trial[1]["effort"][250, 0] = 20.00003
    result = evaluate(trial)
    assert "joint_limit" not in result.failure_categories
    assert "effort_limit" not in result.failure_categories
    trial[1]["joint_position"][250, 0] = 1.021
    trial[1]["effort"][250, 0] = 20.0002
    result = evaluate(trial)
    assert "joint_limit" in result.failure_categories
    assert "effort_limit" in result.failure_categories


def test_heel_lift_does_not_count_as_horizontal_patch_slip(trial):
    trial[1]["support_anchor_position_w"][50:70, 0, 2] += .05
    assert evaluate(trial).complete_cycle_success
    trial[1]["support_anchor_position_w"][60, 0, 0] += .031
    assert "foot_slip" in evaluate(trial).failure_categories
    trial[1]["support_patch_contact"][60, 0] = False
    result = evaluate(trial)
    assert "foot_reposition" in result.failure_categories
    assert "foot_slip" not in result.failure_categories


def test_impact_probes_do_not_count_as_primary_acceptance(trial):
    trial[1]["evidence_kind"] = "physics_rollout"
    assert evaluate(trial).acceptance_eligible
    trial[1]["inter_character_collision_enabled"] = True
    trial[1]["human_g1_collision"][120] = True
    result = evaluate(trial)
    assert result.complete_cycle_success
    assert not result.acceptance_eligible
    assert result.metrics["human_g1_contact_samples"] == 1


def test_family_seed_sample_gates_and_pairing():
    rows = [EpisodeResult(family, seed, f"r{i}", "s0", "t0", f"command:{i}", True, True, (),
                          request_sha256="a"*64)
            for family in ("jab", "cross", "left_hook", "right_hook")
            for seed in (1, 2, 3) for i in range(10)]
    kwargs = dict(expected_seeds=(1, 2, 3), expected_trials_per_family_seed=10,
                  expected_cells_per_family_seed=1, bootstrap_samples=20)
    summary = summarize_results(rows, **kwargs)
    assert summary["all_gates_passed"]
    assert summary["family_seed"]["jab/1"]["trajectory_cluster_95"] == (1., 1.)
    assert not summarize_results(rows[:-1], **kwargs)["all_gates_passed"]
    duplicate = rows.copy()
    duplicate[0] = replace(duplicate[0], recipe_id="r1")
    assert not summarize_results(duplicate, **kwargs)["all_gates_passed"]
    mismatched = rows.copy()
    mismatched[0] = replace(mismatched[0], request_sha256="b"*64)
    assert not summarize_results(mismatched, **kwargs)["all_gates_passed"]
