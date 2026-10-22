"""Offline DuoBox -> existing GMR solver -> selected 31-joint H2 references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from human_punching.motion import load_crop
from human_punching.task10c.retargeting import CONFIG_PATH, DuoBoxAdapter, ExplicitH2Retargeter, create_duobox_config
from human_punching.task10c.references import arrays_from_qpos, quaternion_interpolate
from human_punching.task10c.robot import GLOVE_OFFSETS, H2_XML, ROOT


def convert(family, dataset_root, output, config_path=CONFIG_PATH):
    began = time.monotonic()
    clip = load_crop(dataset_root, family, padding_s=0.10)
    if not config_path.exists():
        create_duobox_config(clip, config_path)
    retarget_config = json.loads(config_path.read_text())
    adapter = DuoBoxAdapter(clip, source_target_offsets_torso_m=retarget_config.get(
        "family_source_target_offsets_torso_m", {}).get(family, {}))
    solver = ExplicitH2Retargeter(config_path=config_path)
    # Warm only the offline solver at the first real source guard; no extra
    # stored attack frames or retrospectively moved launch timestamps.
    for _ in range(12):
        solver.retarget(adapter.frame(0))
    heads = [solver.model.joint(name).qposadr[0] for name in ("head_pitch_joint", "head_yaw_joint")]
    raw = []
    for i in range(len(clip.time_s)):
        q = solver.retarget(adapter.frame(i))
        q[heads] = 0
        solver.configuration.update(q)
        raw.append(q)
    raw = np.asarray(raw)
    # Mild, documented 12.5 ms Gaussian smoothing on scalar coordinates; root
    # orientation is interpolated on SO(3), never averaged as raw quaternion.
    smooth = gaussian_filter1d(raw, sigma=1.5, axis=0, mode="nearest")
    smooth[:, 3:7] = raw[:, 3:7]
    t = np.arange(0, clip.time_s[-1] + 1e-8, .02)
    t = np.minimum(t, clip.time_s[-1])
    qpos = np.stack([np.interp(t, clip.time_s, smooth[:, j]) for j in range(38)], axis=-1)
    qpos[:, 3:7] = quaternion_interpolate(clip.time_s, raw[:, 3:7], t)
    # Put the lowest corrected sole vertex 2 mm above ground per frame. This
    # preserves steps/pivots and prevents nominal penetration; not planted IK.
    data = mujoco.MjData(solver.model)
    shifts = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(solver.model, data)
        lower = []
        for side in ("left", "right"):
            gid = solver.model.geom(side + "_foot_sole_collision").id
            mat = data.geom_xmat[gid].reshape(3, 3)
            lower.append(data.geom_xpos[gid, 2] - np.dot(np.abs(mat[2]), solver.model.geom_size[gid]))
        shift = .002 - min(lower)
        q[2] += shift
        shifts.append(shift)
    arrays = arrays_from_qpos(t, qpos)
    arrays["source_frames"] = clip.source_frames[0] + t * 120
    strike = (clip.metadata["outward_completion_frame"] - clip.source_frames[0]) / 120
    recovery = max(.6, float(t[-1] - strike))
    center = max(.5, strike)
    metadata = {"schema": "task10c_h2_reference_v1", "family": family,
                "active_hand": clip.metadata["anatomical_hand"], "strike_time_s": strike,
                "end_time_s": float(t[-1]), "recovery_duration_s": recovery,
                "ttc_range_s": [max(.5, .9 * center), 1.1 * center],
                "terminal_guard_hold_s": .4, "guard_start_time_s": 0.,
                "terminal_guard_interval_s": [float(max(strike + .1, t[-1] - .1)), float(t[-1])],
                "withdrawal_interval_s": [strike, float(t[-1])],
                "annotation_rationale": "Real 0.10 s source context on both ends; inherited outward-completion frame is nominal strike annotation, never actual encounter ground truth. Terminal interval is recorded candidate guard, no exact joint return.",
                "source": clip.metadata, "adapter": adapter.metadata,
                "source_revisions": {"GMR": "bb1bbe40774794fceb2a7c579a3464a28e68c844", "h2_extension": "3d9a05d951b49e803d9d2c3555aed29a4b1e4b2b"},
                "asset_sha256": hashlib.sha256(H2_XML.read_bytes()).hexdigest(),
                "retarget_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "retarget_config_path": str(config_path.resolve()),
                "retarget_arm_target_revision": retarget_config.get("provenance", {}).get("arm_target_revision", "orientation_fit_v1"),
                "solver": "GMR inherited two-stage Mink proxqp", "ik_dt_s": 1/120,
                "solver_api_correction": "Four local call sites pass the existing limits by keyword; equations, task updates and iteration remain inherited. See _gmr/SOURCE.json.",
                "ik_joint_interior_margin_rad": .001,
                "output_hz": 50., "quaternion_order": "wxyz", "qvel_root_angular_frame": "MuJoCo local root frame",
                "glove_radius_m": .08, "glove_attachment_offsets_m": GLOVE_OFFSETS,
                "smoothing_sigma_source_frames": 1.5,
                "sole_ground_shift_range_m": [float(min(shifts)), float(max(shifts))],
                "head_coordinates": "31-joint model pitch/yaw explicitly neutral and zero velocity",
                "generation_wall_s": time.monotonic() - began}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{family}.npz", **arrays)
    (output / f"{family}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"family": family, "frames": len(t), "strike_s": strike,
                      "generation_wall_s": metadata["generation_wall_s"],
                      "glove_start": arrays["glove_pos_w"][0].tolist(),
                      "glove_strike": arrays["glove_pos_w"][np.argmin(abs(t-strike))].tolist()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["jab", "left_hook"])
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "DuoBox")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task10c/references")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--correct-guards", action="store_true",
                        help="Fit and blend physical guard endpoints; retain raw GMR references in OUTPUT/mapping.")
    parser.add_argument("--source-terminal-torso", action="store_true",
                        help="With --correct-guards, use calibrated recorded torso orientation for hook terminal guards.")
    args = parser.parse_args()
    if args.source_terminal_torso and not args.correct_guards:
        parser.error("--source-terminal-torso requires --correct-guards")
    for family in args.families:
        raw_output = args.output / "mapping" if args.correct_guards else args.output
        convert(family, args.dataset_root, raw_output, args.config)
        if args.correct_guards:
            from human_punching.task10c.guard_correction import correct_reference
            details = correct_reference(raw_output / f"{family}.npz", args.output / f"{family}.npz",
                                        source_terminal_torso=args.source_terminal_torso)
            print(json.dumps({"family": family, "guard_correction": details["method"],
                              "output": str((args.output / f"{family}.npz").resolve())}), flush=True)
