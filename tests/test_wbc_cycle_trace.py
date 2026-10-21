"""CPU kinematics and synthetic contact fixtures; no physics execution."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.contracts import LaunchSnapshot, PunchRequest
from human_punching.wbc.cycle_trace import (
    CycleTraceRecorder, TraceBindings, classify_contacts, derive_kinematics,
)
from human_punching.wbc.evaluation import evaluate_episode


@pytest.fixture(scope="module")
def model(task10_common_bundle):
    return mujoco.MjModel.from_xml_path(str(task10_common_bundle.model_path))


@pytest.fixture
def bindings():
    return TraceBindings(prefix="", floor_geoms=("floor",), human_geom_prefix="physical/")


def test_fresh_fk_and_analytic_relative_velocity(model, bindings):
    q = model.qpos0.copy()
    v = np.zeros(model.nv)
    # Rigid root rotation plus translation: glove/chest relative velocity is zero.
    v[:6] = [.2, -.1, .03, .3, -.2, .8]
    fk = derive_kinematics(model, q[None], v[None], bindings)
    np.testing.assert_allclose(fk["glove_velocity_guard_frame"], 0, atol=1e-12)
    # Articulated velocity must agree with derivative in the rotating chest frame.
    v[model.joint("Chest_z").dofadr[0]] = .7
    v[model.joint("LeftArm_x").dofadr[0]] = .4
    epsilon = 1e-6
    qp, qm = q.copy(), q.copy()
    mujoco.mj_integratePos(model, qp, v, epsilon)
    mujoco.mj_integratePos(model, qm, v, -epsilon)
    fk = derive_kinematics(model, q[None], v[None], bindings)
    perturbed = derive_kinematics(model, np.stack((qm, qp)), np.stack((v, v)), bindings)
    derivative = np.diff(perturbed["glove_position_guard_frame"], axis=0)[0] / (2*epsilon)
    np.testing.assert_allclose(fk["glove_velocity_guard_frame"][0], derivative, atol=1e-8)


def test_fresh_fk_ignores_live_derived_state_and_named_order(model, bindings):
    q = np.tile(model.qpos0, (2, 1))
    q[1, :3] += [1, 2, 3]
    fk = derive_kinematics(model, q, np.zeros((2, model.nv)), bindings)
    np.testing.assert_allclose(fk["glove_position_w"][1] - fk["glove_position_w"][0], [[1, 2, 3]]*2)
    np.testing.assert_allclose(fk["glove_position_guard_frame"][1], fk["glove_position_guard_frame"][0], atol=1e-12)
    assert fk["glove_position_w"][0, 0, 1] > fk["glove_position_w"][0, 1, 1]


def contact_fixture(model, pairs, positions, normal_forces):
    return {"geom": np.asarray([[model.geom(a).id, model.geom(b).id] for a, b in pairs], int).reshape(-1, 2),
            "pos": np.asarray(positions, float).reshape(-1, 3),
            "force": np.asarray([[force, 0, 0, 0, 0, 0] for force in normal_forces], float).reshape(-1, 6),
            "dist": -np.ones(len(pairs))*.001}


def test_support_uses_actual_force_and_fixed_forefoot_patch(model, bindings):
    anchors = np.array([[.2, .1, 0], [.2, -.1, 0]])
    # Left forefoot loaded, right heel loaded but outside the fixed patch.
    contacts = contact_fixture(model, [("floor", "physical/LeftFoot"), ("physical/RightFoot", "floor")],
                               [anchors[0], anchors[1] + [.15, 0, 0]], [10., 12.])
    result = classify_contacts(model, contacts, anchors, bindings)
    np.testing.assert_array_equal(result["support_contact"], [True, True])
    np.testing.assert_array_equal(result["support_patch_contact"], [True, False])
    assert not result["forbidden_collision"]
    contacts["force"][:] = 0
    result = classify_contacts(model, contacts, anchors, bindings)
    assert not result["support_contact"].any()  # Penetration alone is not load.


def test_collision_classification_includes_nonfoot_and_self_contacts(model, bindings):
    anchors = np.zeros((2, 3))
    contacts = contact_fixture(model, [("floor", "physical/Head")], [[0, 0, 0]], [1.])
    assert classify_contacts(model, contacts, anchors, bindings)["forbidden_collision"]
    contacts = contact_fixture(model, [("physical/LeftHand", "physical/Chest")], [[0, 0, 1]], [1.])
    assert classify_contacts(model, contacts, anchors, bindings)["forbidden_collision"]
    allowed = TraceBindings(prefix="", floor_geoms=("floor",), human_geom_prefix="physical/",
                            allowed_self_pairs=(("physical/LeftHand", "physical/Chest"),))
    assert not classify_contacts(model, contacts, anchors, allowed)["forbidden_collision"]


def request():
    g1q = np.zeros(36)
    g1q[3] = 1.
    snapshot = LaunchSnapshot("synthetic", g1q, np.zeros(35), (1, 0, 1.2), (0, 0, 0),
                              (0, 0, 0), (1, 0, 0, 0), "synthetic-group")
    return PunchRequest("jab", .6, snapshot, "test-guard")


def synthetic_recorder(model, n=3):
    recorder = CycleTraceRecorder([request()])
    recorder.model = model
    for tick in range(n):
        recorder.rows.append(dict(time=np.asarray([tick*.005], np.float32),
            qpos=model.qpos0[None].copy(), qvel=np.zeros((1, model.nv)),
            actuator_force=np.zeros((1, model.nu)), qfrc_actuator=np.zeros((1, model.nv)),
            qfrc_applied=np.zeros((1, model.nv)), xfrc_applied=np.zeros((1, model.nbody, 6)),
            overflow=np.zeros(4, int), step_counter=np.asarray(tick), reset_counts=np.asarray([1]),
            external_support=np.asarray([False]), pose_replacement=np.asarray([False])))
        recorder.contacts.append(dict(geom=np.zeros((0, 2), int), pos=np.zeros((0, 3)),
                                      dist=np.zeros(0), force=np.zeros((0, 6)), worldid=np.zeros(0, int)))
    return recorder


def build_arrays(recorder, model, bindings):
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == 3]
    return recorder.episode_arrays(0, reference_glove_position_w=np.zeros((len(recorder.rows), 2, 3)),
        joint_names=names, outward_end_s=.005, bindings=bindings, evidence_kind="synthetic_test")


def test_array_schema_preserves_interval_and_clock_evidence(model, bindings):
    recorder = synthetic_recorder(model)
    result = build_arrays(recorder, model, bindings)
    np.testing.assert_allclose(result["time_s"], [0, .005, .010], atol=0)
    np.testing.assert_allclose(result["force_interval_start_s"], [0, 0, .005], atol=0)
    assert result["effort"].shape == (3, 48)
    assert result["external_wrench"].shape == (3, model.nbody, 6)
    evaluation = evaluate_episode(request(), result, training_seed=0, recipe_id="test",
                                  spatial_cell_id="test", ttc_cell_id="test")
    assert not evaluation.acceptance_eligible
    assert "missing_evidence" not in evaluation.failure_categories
    assert "malformed_evidence" not in evaluation.failure_categories


def test_missing_assistance_dropped_ticks_and_clock_fail_closed(model, bindings):
    recorder = synthetic_recorder(model)
    del recorder.rows[1]["pose_replacement"]
    with pytest.raises(ValueError, match="Missing recorded"):
        build_arrays(recorder, model, bindings)
    recorder = synthetic_recorder(model)
    recorder.rows[1]["step_counter"] += 1
    with pytest.raises(ValueError, match="Dropped"):
        build_arrays(recorder, model, bindings)
    recorder = synthetic_recorder(model)
    recorder.rows[1]["time"] += .001
    with pytest.raises(ValueError, match="Native clock"):
        build_arrays(recorder, model, bindings)


def test_generalized_and_opposing_body_assistance_never_cancel(model, bindings):
    recorder = synthetic_recorder(model)
    recorder.rows[1]["qfrc_applied"][0, -1] = .01
    recorder.rows[2]["xfrc_applied"][0, 1:3, 0] = [1., -1.]
    result = build_arrays(recorder, model, bindings)
    assert result["external_support"][1]
    evaluation = evaluate_episode(request(), result, training_seed=0, recipe_id="test",
                                  spatial_cell_id="test", ttc_cell_id="test")
    assert "external_support" in evaluation.failure_categories
    assert "external_force" in evaluation.failure_categories


def test_raw_contacts_are_lossless_and_follow_preceding_interval(model, bindings):
    recorder = synthetic_recorder(model)
    q0 = model.qpos0.copy()
    anchor = derive_kinematics(model, q0[None], np.zeros((1, model.nv)), bindings)["support_anchor_position_w"][0, 0]
    # The integrated foot moves after the recorded contact was evaluated.
    recorder.rows[1]["qpos"][0, 0] += .2
    actual = contact_fixture(model, [("floor", "physical/LeftFoot")], [anchor], [10.])
    actual["worldid"] = np.zeros(1, int)
    recorder.contacts[1] = actual
    arrays = build_arrays(recorder, model, bindings)
    assert arrays["support_patch_contact"][1, 0]
    assert arrays["support_anchor_position_w"][1, 0, 0] - anchor[0] == pytest.approx(.2)
    raw = recorder.raw_arrays()
    np.testing.assert_array_equal(raw["contact_offsets"], [0, 0, 1, 1])
    np.testing.assert_array_equal(raw["contact_geom"], actual["geom"])
    np.testing.assert_array_equal(raw["contact_force"], actual["force"])
    assert raw["request_sha256"][0] == request().fingerprint
