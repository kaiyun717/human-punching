"""A phase warp must change derivatives and annotations, not just timestamps."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

from human_punching.scalar_body import build_scalar_human
from human_punching.scalar_reference import derivatives

SPEC = importlib.util.spec_from_file_location(
    "retime_punch_reference", Path(__file__).resolve().parents[1]/"scripts/retime_punch_reference.py")
retimer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retimer)


def test_warp_recomputes_velocity_and_preserves_guard_with_honest_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(retimer, "PROJECT", tmp_path)
    body = build_scalar_human(tmp_path/"body")
    model = mujoco.MjModel.from_xml_path(body["xml_path"])
    times = np.linspace(0, 2, 121)
    poses = np.repeat(model.qpos0[None], len(times), axis=0)
    joint = model.joint("RightForeArm_z").qposadr[0]
    moving = (times > .5) & (times < 1.5)
    poses[moving, joint] = .1*np.sin(np.pi*(times[moving]-.5))**2
    source = tmp_path/"reference.npz"
    np.savez(source, time_s=times, qpos=poses, qvel=derivatives(model,times,poses),
             source_time_s=np.clip(times-.5,0,1), launch_time=.5, return_start=1.5,
             guard_hold_start=1.8, active_hand_index=1, active_hand="Right",
             punch_family="right_hook", source_speed=1., head_center=[10.,0.,1.])
    output = tmp_path/"outputs/warped"
    report = retimer.retime(body["xml_path"], source, output,
                            [[0,0],[.5,.25],[1.,1.1],[1.5,1.5],[1.8,1.8],[2,2]])
    with np.load(output/"right_hook.npz", allow_pickle=False) as warped:
        assert "source_speed" not in warped
        assert str(warped["timing_kind"]).startswith("nonuniform")
        assert float(warped["launch_time"]) == .25
        assert float(warped["guard_hold_start"]) == 1.8
        np.testing.assert_array_equal(warped["qpos"][0], warped["qpos"][-1])
        np.testing.assert_array_equal(warped["qvel"][[0,-1]], 0)
        np.testing.assert_allclose(warped["qvel"], derivatives(model,warped["time_s"],warped["qpos"]))
        assert report["local_source_speed_min_max"][0] < 1.
        assert report["local_source_speed_min_max"][1] > 1.
