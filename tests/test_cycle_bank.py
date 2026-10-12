"""Reference boundaries are enforceable even when adjacent guards differ."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.contact import GEOMETRY_VERSION
from human_punching.cycle_bank import (
    BoundaryCyclePlayer,
    CycleBank,
    interpolate_pose,
    kinematic_guard_bridge,
)


@pytest.fixture
def bank(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text('''<mujoco><worldbody><body name="root"><freejoint/>
    <geom type="sphere" size=".1"/><body pos="0 0 .2"><joint type="ball"/>
    <geom type="sphere" size=".05"/></body></body></worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(xml))
    manifest = {"schema": "human_punching_contact_reference_v1", "geometry_version": GEOMETRY_VERSION,
                "source_model": str(xml),
                "source_model_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                "references": {}}
    for index, name in enumerate(("jab", "cross")):
        folder = tmp_path / name
        folder.mkdir()
        poses = np.tile(model.qpos0, (3, 1))
        poses[:, 0] = index + np.array([0., .1, .2])
        np.savez(folder / "human_trajectory.npz", time_s=[0., .03, .07], qpos=poses,
                 qvel=np.zeros((3, model.nv)))
        digest = hashlib.sha256((folder / "human_trajectory.npz").read_bytes()).hexdigest()
        manifest["references"][name] = {
            "punch": name, "human_trajectory_sha256": digest,
            "frozen_head_ttc_from_clip_start_s": .04,
            "any_proxy_contact": True, "head_center_w": [.8, .2, 1.2],
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return CycleBank(path)


def start(player, name="jab", time=5.):
    qpos, qvel = player.bank.sample(name, 0.)
    return player.start(name, [.8, .2, 1.2], time, current_qpos=qpos, current_qvel=qvel)


def test_mask_filters_fixed_head_family_duration_and_diagnostic_time(bank):
    assert all(bank.valid_plan_mask([.8, .2, 1.2]).values())
    assert bank.valid_plan_mask([.8, .2, 1.2], family="jab") == {"jab": True, "cross": False}
    assert not any(bank.valid_plan_mask([.9, .2, 1.2]).values())
    assert not any(bank.valid_plan_mask([.8, .2, 1.2], max_duration_s=.05).values())
    assert not any(bank.valid_plan_mask([.8, .2, 1.2], diagnostic_ttc_window_s=(.05, .1)).values())


def test_plan_freezes_snapshot_and_cannot_be_replaced_midcycle(bank):
    player = BoundaryCyclePlayer(bank)
    head = np.array([.8, .2, 1.2])
    qpos, qvel = bank.sample("jab", 0.)
    player.start("jab", head, 5, current_qpos=qpos, current_qvel=qvel)
    head[:] = 42
    assert player.plan.head_center_w == (.8, .2, 1.2)
    with pytest.raises(FrozenInstanceError):
        player.plan.variant = "cross"
    with pytest.raises(RuntimeError, match="Plan changes"):
        start(player, "cross")
    assert not any(player.valid_plan_mask([.8, .2, 1.2]).values())


def test_control_clock_interpolation_and_completion_need_guard(bank):
    player = BoundaryCyclePlayer(bank, .02)
    first = start(player)
    assert first.world_time_s == 5 and not first.cycle_complete
    sample = player.advance()
    assert sample.world_time_s == pytest.approx(5.02)
    assert sample.qpos[0] == pytest.approx(.1 * .02 / .03)
    player.advance()
    assert not player.advance().cycle_complete
    last = player.advance()
    assert last.world_time_s == pytest.approx(5.08)
    assert last.elapsed_s == pytest.approx(.07)
    assert last.cycle_complete and last.requires_guard_controller
    assert player.state == "waiting_guard"
    with pytest.raises(RuntimeError, match="No active cycle"):
        player.advance()
    with pytest.raises(RuntimeError):
        start(player, "cross", 5.08)
    # One cycle ends at x=.2 while the next begins at x=1: no teleport.
    with pytest.raises(ValueError, match="Guard does not match"):
        player.acknowledge_guard("cross", current_qpos=last.qpos, current_qvel=last.qvel)
    qpos, qvel = bank.sample("cross", 0.)
    player.acknowledge_guard("cross", current_qpos=qpos, current_qvel=qvel)
    assert player.state == "boundary"
    with pytest.raises(ValueError, match="cannot precede"):
        start(player, "cross", 5.07)
    second = start(player, "cross", 5.08)
    assert second.qpos[0] == 1


def test_guard_admission_checks_velocity_as_well_as_pose(bank):
    player = BoundaryCyclePlayer(bank)
    qpos, qvel = bank.sample("jab", 0.)
    qvel[0] = 1
    with pytest.raises(ValueError, match="Guard does not match"):
        player.start("jab", [.8, .2, 1.2], 0, current_qpos=qpos, current_qvel=qvel)


def test_quaternion_interpolation_uses_shortest_arc_and_normalization(bank):
    first = bank.model.qpos0.copy()
    second = first.copy()
    # Root yaw of 90 degrees, then negate its quaternion representative.
    second[3:7] = -np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    middle = interpolate_pose(bank.model, first, second, .5)
    np.testing.assert_allclose(middle[3:7], [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)], atol=1e-12)
    bridge = kinematic_guard_bridge(bank.model, first, second, .1, .03)
    assert bridge["time_s"][-1] == .1
    np.testing.assert_allclose(np.linalg.norm(bridge["qpos"][:, 3:7], axis=1), 1.)
    np.testing.assert_allclose(bridge["qvel"][[0, -1]], 0.)
    delta = np.empty(bank.model.nv)
    mujoco.mj_differentiatePos(bank.model, delta, 1., bridge["qpos"][-1], second)
    np.testing.assert_allclose(delta, 0., atol=1e-12)
    assert bridge["requires_physical_guard_controller"]


def test_changed_payload_is_rejected(bank):
    path = bank.path.parent / "jab/human_trajectory.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        CycleBank(bank.path)


def test_actual_generated_bank_is_readable_when_present():
    path = Path(__file__).resolve().parents[1] / "outputs/contact-targeting/manifest.json"
    if not path.is_file():
        pytest.skip("Generated reference bank not installed")
    bank = CycleBank(path)
    assert len(bank.variants) == 12
    item = bank.variants["jab"]
    assert sum(bank.valid_plan_mask(item.head_center_w, family="jab").values()) == 3
    player = BoundaryCyclePlayer(bank)
    qpos, qvel = bank.sample("jab", 0.)
    player.start("jab", item.head_center_w, 0., current_qpos=qpos, current_qvel=qvel)
    while player.state == "active":
        sample = player.advance()
        assert np.isfinite(sample.qpos).all() and np.isfinite(sample.qvel).all()
    assert sample.requires_guard_controller


def test_guard_errors_separate_root_joint_translation_and_rotation(bank):
    player = BoundaryCyclePlayer(bank)
    qpos, qvel = bank.sample("jab", 0.)
    qpos[0] += .3
    qpos[7:11] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
    qvel[0], qvel[6] = .4, 2.
    error = player.guard_errors("jab", qpos, qvel)
    assert error["root_position_m"] == pytest.approx(.3)
    assert error["root_orientation_rad"] == pytest.approx(0.)
    assert error["joint_angle_rad"] == pytest.approx(np.pi / 2)
    assert error["root_linear_velocity_mps"] == pytest.approx(.4)
    assert error["joint_angular_velocity_radps"] == pytest.approx(2.)
