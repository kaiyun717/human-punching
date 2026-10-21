"""Offline historical measurement adapter tests; no simulated physics steps."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.release_baseline_evaluation import (
    evaluator_identity, evaluate_historical_panel, historical_precision, recorded_support,
)


def contact_fixture():
    names = {"terrain": 0, "robot/physical/LeftFoot": 1, "robot/physical/RightFoot": 2}
    model = SimpleNamespace(ngeom=4, geom=lambda name: SimpleNamespace(id=names[name]))
    arrays = dict(time=np.zeros((3, 2)), contact_offsets=np.array([0, 0, 4, 6]),
        contact_geom=np.array([[0, 1], [2, 0], [0, 1], [0, 3], [1, 0], [0, 2]]),
        contact_worldid=np.array([0, 0, 1, 1, 1, 1]),
        contact_force=np.array([[2., 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0],
            [1.e-6, 8, 0, 0, 0, 0], [90., 0, 0, 0, 0, 0],
            [1.1e-6, 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0]]))
    return model, arrays


def test_support_uses_contact_frame_normal_force_and_actual_world_ownership():
    model, arrays = contact_fixture()
    actual = recorded_support(model, arrays, 2)
    expected = np.zeros((3, 2, 2), bool)
    expected[1, 0] = True
    expected[2, 1] = True
    np.testing.assert_array_equal(actual, expected)
    # Tangential force is irrelevant; no first-pyramidal-ray approximation.
    arrays["contact_force"][:, 1:] = 1.e5
    np.testing.assert_array_equal(recorded_support(model, arrays, 2), expected)


@pytest.mark.parametrize("key,index,value", [
    ("contact_offsets", 2, -1), ("contact_worldid", 0, 2),
    ("contact_geom", (0, 0), 4), ("contact_force", (0, 0), float("nan")),
])
def test_support_rejects_incomplete_or_invalid_saved_contacts(key, index, value):
    model, arrays = contact_fixture()
    arrays[key][index] = value
    with pytest.raises(ValueError, match="Malformed recorded"):
        recorded_support(model, arrays, 2)


def test_strict_release_precision_requires_both_tracking_windows_and_timing():
    row = dict(reference_tracking={"active_glove_position_rms_m": .02,
                                  "attack_active_glove_position_rms_m": .02},
               contact_timing={"absolute_interpolated_contact_error_s": .02},
               precision_targets={"active_glove_rms_target_met": True, "contact_timing_target_met": True})
    assert all(historical_precision(row).values())
    for group, key in (("reference_tracking", "active_glove_position_rms_m"),
                       ("reference_tracking", "attack_active_glove_position_rms_m"),
                       ("contact_timing", "absolute_interpolated_contact_error_s")):
        other = deepcopy(row)
        other[group][key] = .020001
        assert not all(historical_precision(other).values())
        other[group][key] = None
        assert not all(historical_precision(other).values())


def test_evaluator_is_pinned_and_unchanged():
    assert set(evaluator_identity()) == {"human_punching.jab_evaluation", "human_punching.contact"}


@pytest.fixture
def saved_nominal(monkeypatch):
    """Use real archived state/force evidence; mock support only at adapter seam.

    The historical archive has already decoded support rather than raw contact
    rows. Contact decoding is independently tested above with explicit forces.
    """
    import mujoco
    from human_punching.wbc.release_baseline_probe import load_release_inputs
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No native/CPU integration"))
    release = PROJECT_ROOT / "outputs/four-punch-milestone"
    inputs = load_release_inputs(PROJECT_ROOT / "outputs/task10/release_baseline_02", release)["jab"]
    inputs = {**inputs, "initial_qpos": inputs["initial_qpos"][:1],
              "initial_qvel": inputs["initial_qvel"][:1], "trial_ids": [0]}
    model = mujoco.MjModel.from_binary_path(str(PROJECT_ROOT /
        "outputs/task10/rb01/run/jab_historical_ff/compiled_model.mjb"))
    with np.load(release / "jab/evaluation/rollout_00.npz", allow_pickle=False) as data:
        old = {key: data[key].copy() for key in data.files}
    # Extend beyond the accepted historical endpoint with clearly bad states.
    # They must remain visible in raw accounting without changing old gates.
    arrays = {key: np.concatenate([old[key][:, None], old[key][-1:, None]], axis=0)
              for key in ("qpos", "qvel", "ctrl", "actuator_force", "qfrc_actuator",
                          "qfrc_applied", "xfrc_applied")}
    # The WBC entity has one extra attachment body; retained trial has no
    # external wrench, so embed its zero evidence into that expanded array.
    assert not np.any(old["xfrc_applied"])
    arrays["xfrc_applied"] = np.zeros((len(arrays["qpos"]), 1, model.nbody, 6), np.float32)
    arrays["qpos"][-1, 0, 2] = 0.
    arrays.update(time=np.r_[old["simulation_time_s"], old["simulation_time_s"][-1] + .005][:, None],
        reset_counts=np.ones((len(arrays["qpos"]), 1), int),
        policy_reference_tick=old["policy_update_step_indices"][:, None],
        policy_action=np.zeros((len(old["policy_update_step_indices"]), 1, 48)))
    support = np.concatenate([old["support_contact"][:, None], old["support_contact"][-1:, None]])
    monkeypatch.setattr("human_punching.wbc.release_baseline_evaluation.recorded_support",
                        lambda model, arrays, worlds: support)
    return model, arrays, inputs, old


def test_original_horizon_and_unchanged_evaluator_preserve_nominal_acceptance(saved_nominal, tmp_path):
    model, arrays, inputs, old = saved_nominal
    result = evaluate_historical_panel(model, arrays, inputs, np.zeros((1, 3)), tmp_path / "panel",
                                      residual_mode="learned", solver_profile="historical")
    assert result["strict_historical_release_pass_count"] == 1
    assert result["pilot_pass_count"] == 1
    assert result["historical_physics_steps"] == int(old["physics_step_count"])
    assert result["excluded_terminal_padding_steps"] == 1
    assert result["trials"][0]["trial_id"] == 0
    with np.load(tmp_path / "panel/rollout_00.npz") as normalized:
        np.testing.assert_array_equal(normalized["qpos"], old["qpos"])
        assert len(normalized["qpos"]) == len(old["qpos"])


def test_early_abort_or_changed_bank_cannot_be_reported_as_historical_panel(saved_nominal, tmp_path):
    model, arrays, inputs, _ = saved_nominal
    incomplete = {key: value[:1] for key, value in arrays.items()}
    with pytest.raises(ValueError, match="Incomplete original-horizon"):
        evaluate_historical_panel(model, incomplete, inputs, np.zeros((1, 3)), tmp_path / "incomplete",
                                  residual_mode="learned", solver_profile="historical")
    arrays["qpos"][0, 0, 7] += .01
    with pytest.raises(ValueError, match="initial bank"):
        evaluate_historical_panel(model, arrays, inputs, np.zeros((1, 3)), tmp_path / "changed",
                                  residual_mode="learned", solver_profile="historical")
