"""Independent auditor failures must remain visible; CPU arrays only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("phase0_independent_auditor", Path(__file__).resolve().parents[1] /
                                            "scripts/task10b_phase0_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    steps, worlds = 8, 1
    initial = np.zeros((worlds, 55), np.float32)
    initial[:, 2:4] = 1.
    velocity = np.zeros((worlds, 54), np.float32)
    actions = np.full((2, worlds, 48), 1.25, np.float32)
    actions[1] = -1.5
    held = np.repeat(actions, 4, axis=0)
    qref = initial[0][None].repeat(steps, 0)
    qref[:, 7:] = .1
    vref = velocity[0][None].repeat(steps, 0)
    vref[:, 6:] = .2
    ff = np.ones((steps, 48), np.float32)
    target = qref[:, None, 7:] + .25*np.clip(held, -1, 1)
    drive = 10*target + 2*vref[:, None, 6:] + ff[:, None]
    force = drive.copy()
    schema = dict(kp=[10]*48, kd=[2]*48, torque_limits=[[-100, 100]]*48,
                  _compiled_joint_limits=[[-2, 2]]*48)
    arrays = dict(time=(np.arange(steps+1)*.005).astype(np.float32)[:, None],
        qpos=np.tile(initial[None], (steps+1, 1, 1)), qvel=np.tile(velocity[None], (steps+1, 1, 1)),
        ctrl=np.concatenate((drive[:1], drive)), actuator_force=np.concatenate((np.zeros_like(force[:1]), force)),
        qfrc_actuator=np.zeros((steps+1, worlds, 54)), qfrc_applied=np.zeros((steps+1, worlds, 54)),
        xfrc_applied=np.zeros((steps+1, worlds, 24, 6)), overflow=np.zeros((steps+1, worlds)),
        reset_counts=np.ones((steps+1, worlds)), reference_physics_steps=np.arange(steps+1)[:, None],
        sim_step_index=np.arange(steps+1), policy_step_index=np.maximum(np.arange(steps+1)-1, 0)//4,
        pre_qpos=np.tile(initial[None], (steps, 1, 1)), pre_qvel=np.tile(velocity[None], (steps, 1, 1)),
        pre_qref=qref[:, None].copy(), pre_vref=vref[:, None].copy(), pre_feedforward=ff[:, None].copy(),
        pre_reference_tick=np.arange(steps)[:, None], pre_actor_action=held,
        policy_action=actions, policy_previous_raw_action=np.concatenate((np.zeros_like(actions[:1]), actions[:1])),
        policy_reference_tick=np.array([[0], [4]]), raw_action=np.concatenate((np.zeros_like(held[:1]), held)),
        joint_position_target=np.concatenate((target[:1], target)),
        joint_velocity_target=np.concatenate((vref[:1, None, 6:], vref[:, None, 6:])),
        joint_effort_target=np.concatenate((ff[:1, None], ff[:, None])))
    arrays["qfrc_actuator"][1:, :, 6:] = force
    master = dict(time_s=np.arange(steps)*.005, qpos=qref, qvel=vref, joint_feedforward=ff)
    return arrays, schema, master, initial, velocity


def test_independent_array_reconstruction_accepts_source_contract():
    arrays, schema, master, qpos, qvel = fixture()
    result, _ = audit.audit_arrays(arrays, schema, master, qpos, qvel, feedforward=True, residual="learned")
    assert all(result["checks"].values()), result


@pytest.mark.parametrize("field,index,value,gate", [
    ("time", (4, 0), .021, "real_native_clock"),
    ("pre_qvel", (1, 0, 6), .2, "force_interval_state_pairing"),
    ("policy_previous_raw_action", (1, 0, 0), 1., "previous_raw_action"),
    ("actuator_force", (1, 0, 0), 1., "native_force"),
    ("pre_feedforward", (1, 0, 0), 0., "immutable_references"),
])
def test_changed_clock_force_history_reference_is_not_silent(field, index, value, gate):
    arrays, schema, master, qpos, qvel = fixture()
    arrays[field][index] = value
    result, _ = audit.audit_arrays(arrays, schema, master, qpos, qvel, feedforward=True, residual="learned")
    assert not result["checks"][gate], result


def test_condition_coverage_requires_all_four_families_and_both_zero_panels():
    assert len(audit.expected_conditions("contract")) == 12
    parity = audit.expected_conditions("parity")
    assert len(parity) == 16
    for family in audit.FAMILIES:
        assert sum(row[0] == family and row[3] == "zero" for row in parity) == 2


def test_independent_observation_contains_raw_previous_output_and_endpoint_reference():
    q = np.zeros((2, 55), np.float32)
    q[:, 2:4] = 1.
    v = np.zeros((2, 54), np.float32)
    ref = dict(t=np.array([0, 1], np.float32), qpos=q, qvel=v,
               glove_pos=np.zeros((2, 2, 3), np.float32))
    previous = np.full((2, 48), 1.5, np.float32)
    obs = audit.observation(q, v, ref["glove_pos"], previous, np.array([0, 3], np.float32), ref, 4.)
    assert obs.shape == (2, 369) and np.isfinite(obs).all()
    clipped = audit.observation(q, v, ref["glove_pos"], np.clip(previous, -1, 1), np.array([0, 3], np.float32), ref, 4.)
    assert not np.array_equal(obs, clipped)


def test_saved_origin_forward_translation_avoids_float32_cancellation():
    source = np.zeros(55, np.float32)
    source[0] = np.float32(.3)
    origin = np.array([1., 1., 0.], np.float32)
    raw = source.copy()
    raw[:3] += origin
    assert not np.array_equal(raw[:3] - source[:3], origin)
    assert audit.recorded_origin_matches_initial(raw, source, origin)
    assert not audit.recorded_origin_matches_initial(raw, source, origin + .001)
    altered = raw.copy()
    altered[7] = .01
    assert not audit.recorded_origin_matches_initial(altered, source, origin)
