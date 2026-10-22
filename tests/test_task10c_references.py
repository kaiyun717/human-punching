"""Focused selected-model, source-frame, interpolation and reference checks."""
import mujoco
import numpy as np
import pytest

from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT, GLOVE_SITE_NAMES, assert_selected_model, joint_addresses
from human_punching.task10c.references import StoredReference, arrays_from_qpos


def test_selected_model_and_foot_glove_attachments():
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    assert_selected_model(model)
    qa, va = joint_addresses(model)
    assert len(set(qa)) == len(set(va)) == len(JOINT_NAMES) == 31
    assert qa == list(range(7, 38))
    assert qa[JOINT_NAMES.index("head_pitch_joint")] < qa[JOINT_NAMES.index("left_shoulder_pitch_joint")]
    for side in ("left", "right"):
        sole = model.geom(side + "_foot_sole_collision")
        assert model.body(sole.bodyid[0]).name == side + "_ankle_pitch_link"
        assert sole.condim == 3
        assert sole.contype == sole.conaffinity == 1
        assert sole.friction[0] == pytest.approx(.6)
        glove = model.geom(side + "_glove_collision")
        center = model.site(side + "_glove_center")
        assert glove.size[0] == pytest.approx(.08)
        np.testing.assert_allclose(glove.pos, center.pos)
        assert np.linalg.norm(center.pos) > .13
    assert not any("foot1_collision" in (model.geom(i).name or "") for i in range(model.ngeom))


def test_reference_fractional_time_retimes_all_velocity_and_holds_terminal():
    t = np.array([0., .1, .2])
    arrays = {"time_s": t, "joint_pos": np.tile(t[:, None], (1,31)),
              "joint_vel": np.ones((3,31)), "qpos": np.zeros((3,38)),
              "root_quat_w": np.tile([1.,0,0,0], (3,1)),
              "glove_lin_vel_w": np.ones((3,2,3))}
    arrays["qpos"][:,3] = 1
    arrays["qpos"][:,7:] = arrays["joint_pos"]
    ref = StoredReference(arrays, {})
    sample = ref.sample(np.array([.05, .2, .3]), rate=2.)
    np.testing.assert_allclose(sample["joint_pos"][:,0], [.05,.2,.2])
    np.testing.assert_allclose(sample["joint_vel"][:,0], [2,0,0])
    np.testing.assert_allclose(sample["glove_lin_vel_w"][:,0,0], [2,0,0])
    np.testing.assert_allclose(sample["qpos"][:,7], [.05,.2,.2])


def test_duobox_input_adapter_preserves_metre_z_up_and_composed_world_frames():
    from human_punching.motion import load_crop
    from human_punching.task10c.retargeting import DuoBoxAdapter
    clip = load_crop(ROOT.parent / "DuoBox", "jab", padding_s=.1)
    adapter = DuoBoxAdapter(clip)
    frames = [adapter.frame(i) for i in (0, 17)]
    for i, frame in zip((0, 17), frames, strict=True):
        np.testing.assert_allclose(frame["Hips"][0][2], clip.positions[i,0,2])
        for name, index in (("LeftHand", 8), ("RightHand", 12)):
            np.testing.assert_allclose(np.linalg.norm(frame[name][0] - frame["Hips"][0]),
                                       np.linalg.norm(clip.positions[i,index] - clip.positions[i,0]))
            assert np.linalg.norm(frame[name][1]) == pytest.approx(1)
        assert "LeftFootFrame" in frame and "RightFootFrame" in frame
        assert "Spine1" in frame and "Spine2" not in frame


def test_body_velocity_matches_regular_body_origin_not_inertial_com():
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    t = np.array([0., .00001, .00002])
    q = np.tile(model.qpos0, (3, 1))
    # Pure root yaw: torso's nonzero inertial offset distinguishes COM/body.
    q[:, 3] = np.cos(t / 2)
    q[:, 6] = np.sin(t / 2)
    arrays = arrays_from_qpos(t, q)
    torso = list(arrays["body_names"]).index("torso_link")
    finite_difference = (arrays["body_pos_w"][2, torso] - arrays["body_pos_w"][0, torso]) / (t[2] - t[0])
    np.testing.assert_allclose(arrays["body_lin_vel_w"][1, torso], finite_difference, atol=1e-8)


def test_jab_guard_retargeting_preserves_recorded_forearm_direction():
    """Catch arm-frame objectives that formerly bent H2 wrists to compensate.

    The original rotation-heavy mapping produced a 66-degree forearm direction
    error despite matching the hand's world rotation to less than one degree.
    This is a representative source-geometry regression, not a skill gate.
    """
    from human_punching.motion import load_crop
    from human_punching.task10c.retargeting import DuoBoxAdapter, ExplicitH2Retargeter

    clip = load_crop(ROOT.parent / "DuoBox", "jab", padding_s=.1)
    adapter = DuoBoxAdapter(clip)
    solver = ExplicitH2Retargeter()
    source = adapter.frame(0)
    for _ in range(12):
        solver.retarget(adapter.frame(0))
    data = solver.configuration.data
    for side in ("left", "right"):
        human = side.title()
        recorded = source[human + "Hand"][0] - source[human + "ForeArm"][0]
        fitted = data.body(side + "_wrist_yaw_link").xpos - data.body(side + "_elbow_link").xpos
        cosine = np.dot(recorded, fitted) / (np.linalg.norm(recorded) * np.linalg.norm(fitted))
        assert np.degrees(np.arccos(np.clip(cosine, -1., 1.))) < 10.


@pytest.mark.parametrize("family", ["jab", "left_hook", "cross", "right_hook"])
def test_generated_family_names_kinematics_and_neutral_head(family):
    path = ROOT / f"outputs/task10c/references/{family}.npz"
    if not path.exists():
        pytest.skip("Reference generation not yet run")
    ref = StoredReference.load(path)
    arr = ref.arrays
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    data = mujoco.MjData(model)
    for i in (0, len(ref.time_s)//2, len(ref.time_s)-1):
        data.qpos[:] = arr["qpos"][i]
        mujoco.mj_forward(model, data)
        for hand, name in enumerate(GLOVE_SITE_NAMES):
            np.testing.assert_allclose(data.site(name).xpos, arr["glove_pos_w"][i,hand], atol=1e-10)
    for name in ("head_pitch_joint", "head_yaw_joint"):
        j = JOINT_NAMES.index(name)
        assert np.max(np.abs(arr["joint_pos"][:,j])) == 0
        assert np.max(np.abs(arr["joint_vel"][:,j])) == 0
    assert ref.metadata["source"]["source_rate_hz"] == 120
    assert 0 < ref.metadata["strike_time_s"] < ref.time_s[-1]
    assert ref.metadata["recovery_duration_s"] >= .6
    np.testing.assert_allclose(np.diff(ref.time_s), .02)
