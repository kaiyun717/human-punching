"""Focused fixed-pitch, unchanged-reduction and no-integration launch contracts."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from human_punching.task10d.camera import (
    _glove_observation_counts, add_camera_only, mount_config, scene_ids,
)


def _camera_model(pitch):
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="robot/torso_link" pos="0.2 -0.1 1.0" euler="10 -12 38">
        <freejoint/><geom type="sphere" size="0.1"/>
      </body>
      <body name="left" mocap="true"><geom name="left_glove" type="sphere"
        size="0.08" contype="0" conaffinity="0" group="0"/></body>
      <body name="right" mocap="true"><geom name="right_glove" type="sphere"
        size="0.08" contype="0" conaffinity="0" group="0"/></body>
    </worldbody></mujoco>''')
    config = mount_config(pitch)
    if pitch is None:
        add_camera_only(spec)
    else:
        add_camera_only(spec, camera_config=config)
    model = spec.compile()
    return model, config, scene_ids(model, ("left_glove", "right_glove"),
                                    camera_config=config)


@pytest.mark.parametrize("pitch", range(-15, 36, 5))
def test_actual_mounted_pitch_sign_projection_and_defaults(pitch):
    from boxing.perception.camera import camera_pose, center_in_frustum
    from boxing.perception.rectified_camera import CAMERA

    accepted_before = CAMERA.record()
    model, config, ids = _camera_model(pitch)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    position, optical = camera_pose(data, ids)
    torso_rotation = data.xmat[ids.torso_body_id].reshape(3, 3)
    expected_forward_local = (np.cos(np.radians(pitch)), 0., np.sin(np.radians(pitch)))
    np.testing.assert_allclose(torso_rotation.T @ optical[:, 2],
                               expected_forward_local, atol=1e-14)
    np.testing.assert_allclose(position,
        data.xpos[ids.torso_body_id] + torso_rotation @ np.array(config.mount_position),
        atol=1e-14)
    points_optical = np.array(((0., 0., 1.), (0., 0., -1.),
        (.99 * config.tan_half_horizontal, 0., 1.),
        (1.01 * config.tan_half_horizontal, 0., 1.),
        (0., .99 * config.tan_half_vertical, 1.),
        (0., 1.01 * config.tan_half_vertical, 1.)))
    world_points = position + points_optical @ optical.T
    np.testing.assert_array_equal(center_in_frustum(world_points, position, optical, config),
                                   (True, False, True, False, True, False))
    assert CAMERA.record() == accepted_before
    if pitch == 25:
        np.testing.assert_array_equal(config.mujoco_quaternion_wxyz,
                                       CAMERA.mujoco_quaternion_wxyz)
    with pytest.raises(ValueError, match="only pitch"):
        add_camera_only(mujoco.MjSpec(), camera_config=replace(config, mount_position=(0., 0., 0.)))


def test_omitted_pitch_inherits_canonical_upward_mount_and_old_pitch_requires_explicit_override():
    from boxing.perception.camera import camera_pose
    from boxing.perception.rectified_camera import CAMERA

    model, config, _ = _camera_model(None)
    ids = scene_ids(model, ("left_glove", "right_glove"))
    assert config.mount_pitch_down_deg == CAMERA.mount_pitch_down_deg == -25.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, optical = camera_pose(data, ids)
    local_forward = data.xmat[ids.torso_body_id].reshape(3, 3).T @ optical[:, 2]
    np.testing.assert_allclose(local_forward,
        (np.cos(np.radians(25.)), 0., np.sin(np.radians(25.))), atol=1e-14, rtol=0)
    historical_model, historical_config, _ = _camera_model(-15.)
    assert historical_config.mount_pitch_down_deg == 15.
    with pytest.raises(AssertionError):
        scene_ids(historical_model, ("left_glove", "right_glove"))


def test_valid_source_pixels_and_actual_union_minimum_are_distinct():
    import torch
    from boxing.perception.depth import oracle_foreground, pool_foreground_depth

    depth = torch.full((1, 376, 672), 3.)
    segmentation = torch.full((1, 376, 672, 2), -1, dtype=torch.int32)
    rgb = torch.zeros_like(depth, dtype=torch.bool)
    for row, col, geom, distance, selected in (
        (0, 0, 4, .4, True), (0, 1, 5, .6, True),
        (42, 42, 5, 1., True), (42, 43, 6, .3, True),
        (84, 84, 4, .8, False), (126, 126, 4, 3., True),
        (168, 168, 4, .05, True), (210, 210, 5, float("nan"), True),
    ):
        depth[0, row, col] = distance
        segmentation[0, row, col] = torch.tensor((geom, 5))
        rgb[0, row, col] = selected
    clean = pool_foreground_depth(depth, oracle_foreground(segmentation, (4, 5), 5))
    acquired = pool_foreground_depth(depth, rgb)
    before = tuple(value.clone() for value in (depth, segmentation, rgb, clean.depth, acquired.depth))
    counts = _glove_observation_counts(segmentation, depth, rgb,
        glove_geom_ids=(4, 5), geom_object_type=5,
        clean_depth=clean.depth, clean_masks=clean.masks,
        acquired_depth=acquired.depth, acquired_masks=acquired.masks)
    expected = dict(glove_valid_source_pixels=[[3, 2]],
        glove_rgb_valid_source_pixels=[[2, 2]], glove_clean_cells=[[3, 2]],
        glove_acquired_cells=[[2, 2]], glove_clean_min_cells=[[3, 1]],
        glove_acquired_min_cells=[[2, 0]])
    for name, values in expected.items():
        assert counts[name].tolist() == values
    for original, value in zip(before, (depth, segmentation, rgb, clean.depth, acquired.depth), strict=True):
        torch.testing.assert_close(original, value, rtol=0., atol=0., equal_nan=True)


def test_native_initial_mocap_fk_camera_refresh_never_integrates():
    """Exercise exact native pose operations; CUDA reset assertion is runtime evidence."""
    from mjlab.scene import Scene
    from human_punching.task10d.g1 import make_task08_cfg
    from human_punching.task10d.replay import Placement, PrescribedH2
    from boxing.perception.camera import camera_pose

    reference = PrescribedH2()
    camera = mount_config(15.)
    cfg = make_task08_cfg(extra_entities={"h2": reference.entity_cfg()},
        scene_spec_fn=lambda spec: add_camera_only(spec, camera_config=camera))
    model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(model)
    bound = reference.bind(model)
    ids = scene_ids(model, bound, camera_config=camera)
    data = mujoco.MjData(model)
    data.qvel[:] = np.arange(model.nv) / 19.
    data.ctrl[:] = np.arange(model.nu) / 23.
    data.time = 0.
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    sample = reference.sample(0., Placement((.14, -.06), .2))
    bound.write(sample, data)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    camera_pose(data, ids)
    for original, current in zip(before, (data.qpos, data.qvel, data.ctrl, data.time), strict=True):
        np.testing.assert_array_equal(current, original)
    np.testing.assert_allclose(data.geom_xpos[bound.glove_geom_ids], sample.glove_pos, atol=1e-14)
