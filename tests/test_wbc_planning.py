from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from human_punching.wbc.contracts import LaunchSnapshot, PunchRequest
from human_punching.wbc.planning import (
    common_guard_candidate, launch_features, place_reference, retarget_candidate,
)
from human_punching.wbc.references import PROJECT_ROOT, load_release_reference


def request_for(clip, ttc=.70):
    qpos = np.zeros(36)
    qpos[:3], qpos[3] = [.8, .1, .8], 1
    qvel = np.zeros(35)
    qvel[:6] = [.1, .2, 0, .3, .4, .5]
    origin = clip.qpos[0, :3].copy()
    origin[2] = 0
    target = clip.target_w-origin
    snapshot = LaunchSnapshot("unit-test", qpos, qvel, target, [.2, 0, .1],
                              [0, 0, 0], [1, 0, 0, 0], "command:1")
    return PunchRequest(clip.family, ttc, snapshot, "unqualified-test-guard")


def transform_request(request, yaw, translation):
    r = Rotation.from_euler("z", yaw)
    s = request.snapshot
    qpos, qvel = np.array(s.g1_qpos), np.array(s.g1_qvel)
    qpos[:3] = r.apply(qpos[:3])+translation
    qpos[3:7] = (r*Rotation.from_quat(qpos[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
    qvel[:3] = r.apply(qvel[:3])
    # Free-root angular qvel is body-local and unchanged by a world transform.
    snapshot = replace(s, g1_qpos=qpos, g1_qvel=qvel,
        head_position_w=r.apply(s.head_position_w)+translation,
        head_velocity_w=r.apply(s.head_velocity_w),
        human_origin_w=r.apply(s.human_origin_w)+translation,
        human_quat_wxyz=(r*Rotation.from_quat(np.asarray(s.human_quat_wxyz)[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]])
    return replace(request, snapshot=snapshot)


def test_frozen_conditioning_and_reference_are_yaw_translation_equivariant():
    clip = load_release_reference(PROJECT_ROOT / "outputs/four-punch-milestone", "jab")
    request = request_for(clip)
    shifted = transform_request(request, .8, np.array([3., -2., 0.]))
    assert len(launch_features(request)) == 77
    np.testing.assert_allclose(launch_features(request), launch_features(shifted), atol=1e-12)
    original, transformed = place_reference(clip, request), place_reference(clip, shifted)
    np.testing.assert_allclose(original.qpos[0, :2], request.snapshot.human_origin_w[:2])
    r = Rotation.from_euler("z", .8)
    np.testing.assert_allclose(transformed.qpos[:, :3],
        r.apply(original.qpos[:, :3])+[3, -2, 0], atol=1e-12)
    np.testing.assert_allclose(transformed.qvel[:, :3],
        r.apply(original.qvel[:, :3]), atol=1e-10)


def test_g1_angular_velocity_uses_body_local_to_human_transform():
    clip = load_release_reference(PROJECT_ROOT / "outputs/four-punch-milestone", "jab")
    request = request_for(clip)
    qpos = np.array(request.snapshot.g1_qpos)
    qpos[3:7] = Rotation.from_euler("z", np.pi/2).as_quat()[[3, 0, 1, 2]]
    request = replace(request, snapshot=replace(request.snapshot, g1_qpos=qpos))
    features = np.asarray(launch_features(request))
    np.testing.assert_allclose(features[10:13], [-.4, .3, .5], atol=1e-12)


def test_retargeted_ttc_is_a_measured_unqualified_candidate(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline planning must never execute physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    clip = load_release_reference(PROJECT_ROOT / "outputs/four-punch-milestone", "jab")
    request = request_for(clip)
    target = np.array(request.snapshot.head_position_w)+[.01, 0, 0]
    request = replace(request, snapshot=replace(request.snapshot, head_position_w=target))
    candidate = retarget_candidate(clip, request, ik_iterations=8)
    assert not candidate.qualified
    assert candidate.clip.launch_time_s == 0
    assert candidate.clip.metadata["requested_ttc_s"] == .70
    assert candidate.clip.metadata["snapshot_fingerprint"] == request.snapshot.fingerprint
    np.testing.assert_array_equal(candidate.clip.target_w, request.snapshot.head_position_w)
    np.testing.assert_array_equal(candidate.clip.qpos[0], candidate.clip.qpos[-1])
    assert np.isfinite(candidate.clip.qvel).all()
    assert abs(candidate.diagnostics["timing_error_s"]) < .005
    assert candidate.diagnostics["physics_executed"] is False
    assert candidate.diagnostics["dynamic_feasibility"] == "unqualified"
    plan = candidate.launch_plan("candidate-jab", "a"*64)
    assert plan.target_position_w == request.snapshot.head_position_w


def test_retargeter_rejects_mismatched_family_and_invalid_guard():
    clip = load_release_reference(PROJECT_ROOT / "outputs/four-punch-milestone", "jab")
    request = request_for(clip)
    with pytest.raises(ValueError, match="family"):
        place_reference(clip, replace(request, family="cross"))
    guard = clip.qpos[0].copy()
    guard[7:] = 100
    with pytest.raises(ValueError, match="joint limits"):
        retarget_candidate(clip, request, guard_qpos_w=guard)


def test_common_guard_seed_is_shared_but_not_balance_qualified():
    clip = load_release_reference(PROJECT_ROOT / "outputs/four-punch-milestone", "jab")
    guard = common_guard_candidate(clip, iterations=40)
    assert not guard.qualified_for_launch
    assert len(guard.qpos_w) == 55
    assert np.isfinite(guard.qpos_w).all()
    np.testing.assert_allclose(guard.qpos_w[:2], [0., 0.], atol=1e-12)
    assert guard.maximum_sole_error_m < .01
    assert guard.maximum_selected_self_penetration_m < .01
    with pytest.raises(ValueError, match="search box"):
        common_guard_candidate(clip, lateral_width_m=.8)
