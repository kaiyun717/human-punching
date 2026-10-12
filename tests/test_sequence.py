"""A failed physical guard gate must prevent the next launch."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.sequence_cli import sequence_rollout

SOURCE = Path(__file__).resolve().parents[1] / "outputs/physical-5890912"
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason="Physical pilot artifact not installed")


def test_guard_timeout_does_not_launch_or_reset_another_cycle():
    model = mujoco.MjModel.from_xml_path(str(SOURCE / "scene/physical_scene.xml"))
    with np.load(SOURCE / "jab/reference.npz", allow_pickle=False) as saved:
        initial = saved["qpos"][0]
    head = np.array(json.loads((SOURCE / "jab/jab_metrics.json").read_text())["head_center_w"])
    # A stationary short reference isolates admission from attack success. The
    # timeout is shorter than the required continuous ready-state dwell.
    trajectory, metrics = sequence_rollout(
        model, np.array([0., .01]), np.stack([initial, initial]), head,
        cycles=2, endpoint_hold_s=0, bridge_s=.01, guard_timeout_s=.002,
        guard_dwell_s=.1,
    )
    assert metrics["outcome"] == "guard_timeout_no_next_launch"
    assert metrics["launched_cycles"] == 1
    assert metrics["completed_punch_cycles"] == 1
    assert not metrics["continuous_sequence_gate"]
    assert metrics["guard_snapshots"][-1]["rejected"]
    assert trajectory["time_s"][-1] > 0
    assert np.isfinite(trajectory["qpos"]).all()
