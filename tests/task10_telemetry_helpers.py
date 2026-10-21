"""Small finite tensor fixtures; no simulator dependency."""
import torch
from human_punching.wbc.telemetry.event_state import FLAGS


def state(n=2, j=3, dtype=torch.float64):
    z = lambda *s: torch.zeros((n, *s), dtype=dtype)
    q = z(4); q[:, 0] = 1
    gq = z(2, 4); gq[:, :, 0] = 1
    gloves = z(2, 3); gloves[:, :, 2] = 1; gloves[:, 0, 0] = .3; gloves[:, 1, 0] = -.3
    root = z(3); root[:, 2] = 1
    feet = z(2, 3); feet[:, 0, 0] = -.1; feet[:, 1, 0] = .1
    return {"glove_position_w": gloves, "root_position_w": root, "root_quat_wxyz": q,
        "root_linear_velocity_w": z(3), "root_angular_velocity_w": z(3),
        "glove_position_guard_frame": gloves.clone(), "glove_quat_guard_frame": gq,
        "glove_velocity_guard_frame": z(2, 3), "support_anchor_position_w": feet,
        "support_anchor_velocity_w": z(2, 3), "joint_position": z(j), "joint_velocity": z(j),
        "effort": z(j), "support_contact": torch.ones((n, 2), dtype=torch.bool),
        "support_patch_contact": torch.ones((n, 2), dtype=torch.bool),
        **{name: torch.zeros(n, dtype=torch.bool) for name in FLAGS}}

