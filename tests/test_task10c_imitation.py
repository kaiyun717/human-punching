"""Focused restored-objective coverage, scaling and commanded-limit checks."""
import json

import mujoco
import numpy as np
import pytest
import torch

from human_punching.task10c.events import guard_weight, scheduled_normalizers
from human_punching.task10c.imitation import (
    H2_TRACKED_BODY_NAMES, IMITATION_WEIGHTS, command_excess_loss,
    command_excess_terms, imitation_rates, named_joint_groups, whole_body_scores,
)
from human_punching.task10c.reward_checks import body_state, sensitivity_checks
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT


@pytest.fixture(scope='module')
def terminal():
    selection = json.loads((ROOT / 'configs/task10c/references.json').read_text())
    with np.load(ROOT / selection['jab'], allow_pickle=False) as saved:
        qpos = saved['qpos'][-1].copy()
    return mujoco.MjModel.from_xml_path(str(H2_XML)), qpos


def test_named_head_knee_ankle_torso_and_final_foot_coverage(terminal):
    result = sensitivity_checks(*terminal)
    assert result['passed'], result
    assert {'head_pitch_link', 'head_yaw_link', 'left_ankle_pitch_link',
            'right_ankle_pitch_link'} <= set(H2_TRACKED_BODY_NAMES)


def test_quaternion_sign_and_horizontal_step_are_pose_invariant(terminal):
    model, qpos = terminal
    base = body_state(model, qpos, np.zeros(model.nv))
    moved = [value.clone() for value in base]
    moved[0][..., :2] += torch.tensor([0.3, -0.2], dtype=torch.float64)
    moved[1] *= -1
    for value in whole_body_scores(*moved, *base).values():
        torch.testing.assert_close(value, torch.ones_like(value), atol=1e-12, rtol=0)


def test_actor_dt_and_full_schedule_integrate_to_seven_plus_six():
    for request in (0.5, 0.707811745538643, 0.9):
        requested = torch.tensor([request], dtype=torch.float64)
        horizon, _, normalizer = scheduled_normalizers(requested, 0.803333333333, 0.02)
        elapsed = torch.arange(1, round(float(horizon[0] / 0.02)) + 1, dtype=torch.float64) * 0.02
        scores = {name: torch.ones_like(elapsed) for name in IMITATION_WEIGHTS}
        ramp = guard_weight(elapsed, requested, 0.803333333333)
        rates = imitation_rates(scores, horizon, ramp, normalizer)
        for name, weight in IMITATION_WEIGHTS.items():
            assert float(rates[name].sum() * 0.02) == pytest.approx(13.0 * weight / 5.0, abs=1e-7)
        # A snippet receives the remaining scheduled reward, not another 13.
        assert float(sum(value[elapsed > request + 0.3].sum() for value in rates.values()) * 0.02) < 13


def test_command_excess_uses_all_six_group_means_and_preserves_arm_gradients():
    groups = named_joint_groups(JOINT_NAMES)
    limits = torch.tensor([[-1.0, 1.0]] * 31, dtype=torch.float64)
    target = torch.zeros((1, 31), dtype=torch.float64, requires_grad=True)
    assert command_excess_loss(target, limits, groups).item() == 0
    with torch.no_grad():
        for ids in groups.values():
            target[0, list(ids)] = 1.3
    terms = command_excess_terms(target, limits, groups)
    assert set(terms) == {'head', 'waist', 'left_leg', 'right_leg', 'left_arm', 'right_arm'}
    assert all(value.item() == pytest.approx(0.5) for value in terms.values())
    loss = command_excess_loss(target, limits, groups)
    assert loss.item() == pytest.approx(3.0)
    loss.backward()
    assert bool(torch.all(target.grad > 0))
    permutation = tuple(reversed(range(31)))
    permuted_names = tuple(JOINT_NAMES[i] for i in permutation)
    torch.testing.assert_close(command_excess_loss(target[:, permutation], limits[list(permutation)],
                                                 named_joint_groups(permuted_names)), loss)
