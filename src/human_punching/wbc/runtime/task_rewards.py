"""Task-specific reward rates from fresh physical telemetry.

These are engineering shaping terms, not acceptance tests. The caller supplies
explicit settling/strike/recovery masks and a nominal guard; this module never
declares a launch or changes a reference. Geometry comes from the final 5 ms
endpoint. Quietness/limits use extrema over all four actual native intervals;
slip uses their accumulated load-bearing patch travel. mjlab integrates rates
over the 20 ms policy period. One-time event bonuses are divided by that period
so they are counted once, rather than scaled by the choice of policy period.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from human_punching.wbc.telemetry.event_state import angle

TensorMap = Mapping[str, torch.Tensor]
POLICY_DT = .020


@dataclass(frozen=True)
class RewardScales:
    """Broader than strict guard tolerances, to provide a learning signal."""

    root_position_m: float = .06
    root_orientation_rad: float = .20
    hand_position_m: float = .06
    hand_orientation_rad: float = math.pi / 6
    joint_posture_rad: float = .24
    footprint_m: float = .06
    root_speed_m_s: float = .15
    root_angular_speed_rad_s: float = .36
    joint_speed_rad_s: float = 1.50
    glove_speed_m_s: float = .45
    loaded_slip_speed_m_s: float = .10
    joint_excursion_rad: float = .02
    effort_excess_nm: float = 1.
    glove_tracking_m: float = .06
    target_position_m: float = .14
    timing_s: float = .04
    event_timing_s: float = .02
    inactive_clearance_m: float = .16
    clearance_deficit_m: float = .02

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Reward scales must be positive and finite")


def _score(error: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-(error / scale).square())


def _maximum_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value, dim=-1).amax(-1)


def validate_guard_joint_posture_metric(value: str) -> str:
    """Keep the historical RMS default; maximum is an explicit recipe choice."""
    if not isinstance(value, str) or value not in ("rms", "maximum"):
        raise ValueError("Guard joint posture metric must be 'rms' or 'maximum'")
    return value


def guard_scores(state: TensorMap, goal: TensorMap, window: TensorMap,
                 scales: RewardScales = RewardScales(), *,
                 guard_joint_posture_metric: str = "rms") -> dict[str, torch.Tensor]:
    """Unmasked endpoint posture and four-tick quietness scores in [0, 1].

    The optional maximum metric uses the strict guard's worst-joint norm, while
    retaining the broader shaping scale. It does not change guard acceptance.
    """
    metric = validate_guard_joint_posture_metric(guard_joint_posture_metric)
    root_error = torch.linalg.vector_norm(state["root_position_w"] - goal["root_position_w"], dim=-1)
    hand_error = _maximum_norm(state["glove_position_guard_frame"] - goal["glove_position_guard_frame"])
    # Compare material forefoot anchors horizontally: heel lift is not a step.
    footprint_error = _maximum_norm((state["support_anchor_position_w"] - goal["support_anchor_position_w"])[..., :2])
    joint_difference = state["joint_position"] - goal["joint_position"]
    joint_error = (joint_difference.square().mean(-1).sqrt() if metric == "rms"
                   else joint_difference.abs().amax(-1))
    quiet_errors = torch.stack((
        window["root_linear_speed_peak_m_s"] / scales.root_speed_m_s,
        window["root_angular_speed_peak_rad_s"] / scales.root_angular_speed_rad_s,
        window["joint_speed_peak_rad_s"] / scales.joint_speed_rad_s,
        window["glove_guard_speed_peak_m_s"] / scales.glove_speed_m_s,
    ), dim=-1)
    return {
        "guard_root_position": _score(root_error, scales.root_position_m),
        "guard_root_orientation": _score(angle(state["root_quat_wxyz"], goal["root_quat_wxyz"]), scales.root_orientation_rad),
        "guard_hands_position": _score(hand_error, scales.hand_position_m),
        "guard_hands_orientation": _score(angle(state["glove_quat_guard_frame"], goal["glove_quat_guard_frame"]).amax(-1), scales.hand_orientation_rad),
        "guard_joint_posture": _score(joint_error, scales.joint_posture_rad),
        "guard_footprint": _score(footprint_error, scales.footprint_m),
        # The worst category matters; fast hands cannot be hidden by quiet legs.
        "guard_quiet": torch.exp(-quiet_errors.square().amax(-1)),
        "guard_double_support": window["double_support_fraction"],
    }


def task_reward_rates(state: TensorMap, goal: TensorMap, window: TensorMap,
                      context: TensorMap, effort_caps: torch.Tensor,
                      scales: RewardScales = RewardScales(), *,
                      guard_joint_posture_metric: str = "rms") -> dict[str, torch.Tensor]:
    """Return unweighted finite rates, without modifying any input tensor.

    Context fields: bool guard_mask/strike_mask/new_contact/new_recovery,
    int64 active_hand, frozen target_w, ttc_s, launch-relative elapsed_s, and
    model-bound reference_glove_w (N,2,3). Uninitialized/terminal rows have both
    phase masks false. Repeated reads do not consume an event; the owning bridge
    clears its event window once per policy interval.
    """
    metric = validate_guard_joint_posture_metric(guard_joint_posture_metric)
    n = state["root_position_w"].shape[0]
    device = state["root_position_w"].device
    dtype = state["root_position_w"].dtype
    for name in ("guard_mask", "strike_mask", "new_contact", "new_recovery"):
        value = context[name]
        if value.shape != (n,) or value.dtype != torch.bool or value.device != device:
            raise ValueError(f"Expected a per-environment boolean {name}")
    hand = context["active_hand"]
    if hand.shape != (n,) or hand.dtype != torch.int64 or hand.device != device:
        raise ValueError("Expected per-environment int64 active_hand")
    for name, shape in (("target_w", (n, 3)), ("reference_glove_w", (n, 2, 3)),
                        ("ttc_s", (n,)), ("elapsed_s", (n,))):
        value = context[name]
        if value.shape != shape or value.dtype != dtype or value.device != device:
            raise ValueError(f"Malformed reward context: {name}")
    if (effort_caps.ndim != 1 or effort_caps.shape[0] != state["effort"].shape[1]
            or effort_caps.device != device or effort_caps.dtype != dtype):
        raise ValueError("Effort caps must match the physical actuator tensor")
    valid_hand = (hand == 0) | (hand == 1)
    row = torch.arange(n, device=device)
    # Prelaunch rows may use -1 metadata. They must not redirect indexing to a
    # wrong hand or invent a strike. Command validation remains the bridge's job.
    index = hand.clamp(0, 1)
    active = state["glove_position_w"][row, index]
    inactive = state["glove_position_w"][row, 1-index]
    reference = context["reference_glove_w"][row, index]
    guard = context["guard_mask"]
    strike = context["strike_mask"] & valid_hand
    enabled = guard | strike
    values = {name: torch.where(guard, value, 0.) for name, value in
              guard_scores(state, goal, window, scales, guard_joint_posture_metric=metric).items()}
    time_error = context["elapsed_s"] - context["ttc_s"]
    target_error = torch.linalg.vector_norm(active-context["target_w"], dim=-1)
    tracking_error = torch.linalg.vector_norm(active-reference, dim=-1)
    clearance = torch.linalg.vector_norm(inactive-context["target_w"], dim=-1)
    values.update({
        "strike_glove_tracking": torch.where(strike, _score(tracking_error, scales.glove_tracking_m), 0.),
        "strike_timed_target": torch.where(strike, _score(target_error, scales.target_position_m)*_score(time_error, scales.timing_s), 0.),
        "inactive_clearance_cost": torch.where(enabled & valid_hand,
            ((scales.inactive_clearance_m-clearance).clamp_min(0)/scales.clearance_deficit_m).square(), 0.),
        "loaded_slip_cost": torch.where(enabled,
            (window["loaded_patch_slip_m"].amax(-1)/(POLICY_DT*scales.loaded_slip_speed_m_s)).square(), 0.),
        "joint_limit_cost": torch.where(enabled,
            (window["joint_limit_excess_rad"]/scales.joint_excursion_rad).square(), 0.),
        "effort_limit_cost": torch.where(enabled,
            (window["effort_excess_nm"]/scales.effort_excess_nm).square(), 0.),
        "effort_cost": torch.where(enabled, (state["effort"]/effort_caps).square().mean(-1), 0.),
        # Contact timing uses its FIRST observed 5 ms timestamp, not the later
        # policy endpoint. The bridge supplies contact_error_s when one occurs.
        "first_contact_bonus": torch.zeros(n, dtype=dtype, device=device),
        "guard_recovered_bonus": torch.where(context["new_recovery"], 1./POLICY_DT, 0.),
    })
    if "contact_error_s" in context:
        value = context["contact_error_s"]
        if value.shape != (n,) or value.dtype != dtype or value.device != device:
            raise ValueError("Malformed first-contact timing error")
        values["first_contact_bonus"] = torch.where(context["new_contact"] & valid_hand,
            _score(value, scales.event_timing_s)/POLICY_DT, 0.)
    else:
        # An absent first-event timestamp cannot earn an endpoint-timed bonus.
        # This also allows a settling-only guard task without fabricated TTCs.
        values["first_contact_bonus"].zero_()
    # Invalid physical data must fail the independent event audit; do not allow
    # a single failing world to inject NaNs into a PPO minibatch. Bad positive
    # scores become zero, bad costs a bounded penalty. This is not a state repair.
    return {name: torch.nan_to_num(value, nan=1. if name.endswith("_cost") else 0.,
                                  posinf=1. if name.endswith("_cost") else 0., neginf=0.)
            for name, value in values.items()}


# Initial engineering weights. Kept separate from acceptance thresholds and
# stock WBC imitation terms. A configuration must explicitly opt into them.
DEFAULT_WEIGHTS = {
    "guard_root_position": .5, "guard_root_orientation": .5,
    "guard_hands_position": 1., "guard_hands_orientation": .25,
    "guard_joint_posture": .25, "guard_footprint": 1.,
    "guard_quiet": 1., "guard_double_support": .25,
    "strike_glove_tracking": 2., "strike_timed_target": 1.,
    "inactive_clearance_cost": -.5, "loaded_slip_cost": -.2,
    "joint_limit_cost": -1., "effort_limit_cost": -1., "effort_cost": -.01,
    "first_contact_bonus": 1., "guard_recovered_bonus": 2.,
}
