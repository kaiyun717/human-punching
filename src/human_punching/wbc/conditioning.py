"""Matched actor-conditioning ablations on top of the same WBC task.

The critic retains the combined privileged context in every comparison. These
variants test what the actor needs at inference; they do not change references,
commands, rewards, PPO, or the launch commitment. No ablation result is implied.
"""
from __future__ import annotations

from enum import Enum


class Conditioning(str, Enum):
    COMBINED = "reference_commands_frozen_g1"
    COMMANDS = "reference_commands"
    REFERENCE = "reference_only"


def commands_without_g1(env):
    from .runtime.mdp import punch_condition
    # The named contract is family[4], target_in_current_human_frame[3],
    # requested/elapsed/remaining time[3], then frozen launch-G1 context[77].
    return punch_condition(env)[:, :10]


def apply_conditioning(cfg, variant: Conditioning | str) -> dict:
    """Mutate a freshly built local config; return explicit experiment metadata.

    Call before environment/actor construction. Never switch an existing actor's
    input dimensions or normalization layout during a learned rollout.
    """
    from mjlab.managers.observation_manager import ObservationTermCfg
    from .runtime.mdp import punch_condition

    variant = Conditioning(variant)
    actor = cfg.observations["actor"].terms
    if "future_joint_reference" not in actor:
        raise ValueError("Conditioning variants require the human reference tracker config")
    if variant is Conditioning.REFERENCE:
        actor.pop("punch_condition", None)
    else:
        actor["punch_condition"] = ObservationTermCfg(
            func=punch_condition if variant is Conditioning.COMBINED else commands_without_g1)
    return {
        "schema": "human_wbc_conditioning_v1", "actor_variant": variant.value,
        "actor_observation_terms": list(actor),
        "critic_variant": Conditioning.COMBINED.value,
        "critic_unchanged": True,
        "explicit_actor_command_dimension": 0 if variant is Conditioning.REFERENCE else 10,
        "frozen_actor_g1_dimension": 77 if variant is Conditioning.COMBINED else 0,
        "references_unchanged": True, "live_g1_after_launch": False,
    }
