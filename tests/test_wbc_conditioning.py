from types import SimpleNamespace

import pytest
import torch

from human_punching.wbc.conditioning import Conditioning, apply_conditioning, commands_without_g1
from human_punching.wbc.runtime.mdp import punch_condition


def environment():
    command = SimpleNamespace(
        trajectory_ids=torch.tensor([0]), family_ids=torch.tensor([1]),
        robot_anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        robot_anchor_pos_w=torch.tensor([[0., 0., 1.]]),
        target_w=torch.tensor([[.8, 0., 1.2]]),
        elapsed=torch.tensor([.2]), launch_times=torch.tensor([0.]),
        ttcs=torch.tensor([.6]), g1_context=torch.zeros(1, 77),
    )
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command)), command


def test_no_g1_actor_does_not_leak_frozen_g1_but_keeps_human_feedback():
    env, command = environment()
    combined = punch_condition(env).clone()
    ablation = commands_without_g1(env).clone()
    command.g1_context[:] = 10.
    assert not torch.equal(punch_condition(env), combined)
    torch.testing.assert_close(commands_without_g1(env), ablation)
    command.robot_anchor_pos_w[:, 0] += .1
    assert not torch.equal(commands_without_g1(env), ablation)
    torch.testing.assert_close(command.target_w, torch.tensor([[.8, 0., 1.2]]))


@pytest.mark.parametrize("variant,command_dims,g1_dims", [
    (Conditioning.COMBINED, 10, 77), (Conditioning.COMMANDS, 10, 0),
    (Conditioning.REFERENCE, 0, 0),
])
def test_actor_variants_preserve_the_same_critic_and_reference_terms(variant, command_dims, g1_dims):
    own_state, reference, future = object(), object(), object()
    critic_terms = {"privileged": object()}
    cfg = SimpleNamespace(observations={
        "actor": SimpleNamespace(terms={"proprioception": own_state, "reference": reference,
                                        "future_joint_reference": future, "punch_condition": object()}),
        "critic": SimpleNamespace(terms=critic_terms),
    })
    metadata = apply_conditioning(cfg, variant)
    assert cfg.observations["critic"].terms is critic_terms
    assert cfg.observations["actor"].terms["proprioception"] is own_state
    assert cfg.observations["actor"].terms["reference"] is reference
    assert cfg.observations["actor"].terms["future_joint_reference"] is future
    assert metadata["explicit_actor_command_dimension"] == command_dims
    assert metadata["frozen_actor_g1_dimension"] == g1_dims
    assert ("punch_condition" in cfg.observations["actor"].terms) == (command_dims > 0)
