"""CPU fixtures establish Phase 1 semantics, not physical/G1 qualification."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from human_punching.functional_recovery import (
    FunctionalGuardThresholds, FunctionalRecoveryStateMachine, derive_guard_nominal,
    evaluate_functional_trace, measure_functional_guard,
)


@pytest.fixture
def nominal_fixture():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".005"/><worldbody>
    <body name="Hips" pos="0 0 1"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
      <geom name="physical/LeftFoot" type="box" pos="0 .15 -.95" size=".15 .08 .05" mass=".1"/>
      <geom name="physical/RightFoot" type="box" pos="0 -.15 -.95" size=".15 .08 .05" mass=".1"/>
      <body name="Chest" pos="0 0 .4"><geom type="sphere" size=".1" mass="1"/>
        <body name="Head" pos="0 0 .3"><geom type="sphere" size=".08" mass=".1"/></body>
        <site name="RightHand_center" pos="-.4 0 0"/>
        <body><joint name="arm" axis="0 0 1"/>
          <geom type="sphere" size=".08" pos=".4 0 0" mass=".1"/>
          <site name="LeftHand_center" pos=".4 0 0"/>
        </body>
      </body>
    </body></worldbody></mujoco>""")
    q = np.tile(model.qpos0, (241, 1))
    v = np.zeros((len(q), model.nv))
    support = np.ones((len(q), 2), bool)
    nominal = derive_guard_nominal(model, q[0], family="synthetic_jab",
                                  provenance={"scope": "isolated synthetic test only"})
    return model, q, v, support, nominal


def test_initial_guard_and_readiness_never_complete_unexecuted_punch():
    machine = FunctionalRecoveryStateMachine()
    for t in np.arange(101)*.005:
        result = machine.observe(t, True, True)
        assert not result["complete"]
        assert result["recovery_time_s"] is None
    assert result["ready"]
    assert result["readiness_time_s"] == pytest.approx(.2)


def test_recovery_without_contact_is_a_completed_miss(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    q[45:80, 7] = .8  # Explicit outward motion, followed by an untouched guard.
    before_q, before_v, before_support = q.copy(), v.copy(), support.copy()
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    summary, events = evaluate_functional_trace(np.arange(len(q))*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["functional_complete"]
    assert summary["contact_outcome"] == "miss"
    assert not summary["active_hand_outward_contact"]
    assert summary["first_active_hand_outward_contact_s"] is None
    assert summary["recovery_time_s"] == pytest.approx(.6)
    assert not events["complete"][0]
    assert summary["diagnostic_stability_complete"]
    assert summary["diagnostic_stability_pass"]
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(v, before_v)
    np.testing.assert_array_equal(support, before_support)


def test_recovery_dwell_cannot_include_preoutward_hold():
    machine = FunctionalRecoveryStateMachine()
    for t in np.arange(61)*.005:
        result = machine.observe(t, True, True, launch=bool(t == .2), outward_complete=bool(t == .3))
    assert result["ready"]
    assert result["recovery_dwell_s"] == 0
    assert not result["complete"]
    for t in np.arange(61, 101)*.005:
        result = machine.observe(t, True, True)
    assert result["complete"]
    assert result["recovery_time_s"] == pytest.approx(.5)


@pytest.mark.parametrize("interruption", ["pose", "support"])
def test_interrupted_dwell_restarts_and_interval_force_alignment(interruption):
    machine = FunctionalRecoveryStateMachine()
    for i in range(121):
        result = machine.observe(i*.005, not (interruption == "pose" and i == 90),
                                 not (interruption == "support" and i == 90),
                                 launch=i == 40, outward_complete=i == 60)
    assert not result["complete"]  # Broken at .45, .6 is only .15 s later.
    expected = .655 if interruption == "pose" else .650
    for i in range(121, 141):
        result = machine.observe(i*.005, True, True)
    assert result["complete"]
    assert result["recovery_time_s"] == pytest.approx(expected)


def test_readiness_admission_is_separate_from_quiet_recovery(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    v[39:45, 3] = .3
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    assert measured["readiness_admission"][40]
    assert not measured["operational_guard"][40]
    summary, _ = evaluate_functional_trace(np.arange(len(q))*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["launch_time_s"] == .2
    assert summary["functional_complete"]
    assert summary["recovery_time_s"] == pytest.approx(.6)


def test_launch_rejection_and_outward_order(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    q[40, 7] = 1.
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    summary, _ = evaluate_functional_trace(np.arange(len(q))*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["launch_rejected"]
    assert summary["launch_time_s"] is None
    assert not summary["functional_complete"]
    machine = FunctionalRecoveryStateMachine()
    with pytest.raises(ValueError, match="follow an actual launch"):
        machine.observe(0., True, True, outward_complete=True)


def test_diagnostics_do_not_become_completion_gates(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    for name in ("hand_orientation_error_rad", "torso_orientation_error_rad", "joint_speed_rad_s"):
        measured[name][:] = 1.
    measured["root_height_fraction"][:] = .85
    measured["com_support_margin_m"][:] = -.01
    measured["frozen_target_glove_distance_m"][:] = .15
    summary, _ = evaluate_functional_trace(np.arange(len(q))*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["functional_complete"]
    assert not any(summary["diagnostic_final_checks"].values())
    assert summary["contact_outcome"] == "miss"


def test_operational_speed_and_position_boundaries(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    v[0, 0], v[1, 0] = .05, .050001
    v[2, 3], v[3, 3] = .12, .120001
    # Rigid root translation/rotation must not appear as Chest-relative hand speed.
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    np.testing.assert_array_equal(measured["operational_root_linear_speed"][:2], [True, False])
    np.testing.assert_array_equal(measured["operational_root_angular_speed"][2:4], [True, False])
    np.testing.assert_allclose(measured["hand_relative_speed_m_s"][:4], 0., atol=1.e-15)
    assert nominal["glove_centers_chest_m"] == [[.4, 0., 0.], [-.4, 0., 0.]]
    assert all(face["local_normal_axis"] == 2 and face["local_normal_sign"] == -1 for face in nominal["sole_faces"])
    np.testing.assert_allclose(np.asarray(nominal["nominal_sole_corners_w"])[..., 2], 0., atol=1.e-15)


def test_nonfinite_state_and_missing_support_cannot_pass(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    q[30, 0] = np.nan
    measured = measure_functional_guard(model, q, v, support, nominal, [10., 0., 1.4])
    assert not measured["operational_finite"][30]
    assert not measured["operational_guard"][30]
    summary, _ = evaluate_functional_trace(np.arange(len(q))*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["recovered_after_outward"]
    assert not summary["physical_finite_upright"]
    assert not summary["functional_complete"]
    with pytest.raises(ValueError, match="support flags"):
        measure_functional_guard(model, q, v, None, nominal, [10., 0., 1.4])


def test_unobserved_stability_tail_is_diagnostic_not_rejection(nominal_fixture):
    model, q, v, support, nominal = nominal_fixture
    measured = measure_functional_guard(model, q[:141], v[:141], support[:141], nominal, [10., 0., 1.4])
    summary, _ = evaluate_functional_trace(np.arange(141)*.005, measured,
        launch_time_s=.2, outward_complete_time_s=.4, active_hand=0)
    assert summary["functional_complete"]
    assert summary["diagnostic_stability_observed_s"] == pytest.approx(.1)
    assert not summary["diagnostic_stability_complete"]
    assert summary["diagnostic_stability_pass"] is None


def test_predicate_version_and_force_identity_are_explicit():
    cfg = FunctionalGuardThresholds()
    assert replace(cfg, version="reviewed_fixture_v1").version == "reviewed_fixture_v1"
    with pytest.raises(ValueError, match="historical"):
        replace(cfg, support_normal_force_n=1.)
    with pytest.raises(ValueError, match="version"):
        replace(cfg, version="")
