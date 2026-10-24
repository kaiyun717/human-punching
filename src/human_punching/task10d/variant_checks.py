"""Compact native-FK inspection of the four explicitly converted source events."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT
from .source_survey import read_recording


def inspect_reference(path: Path) -> dict:
    reference = StoredReference.load(path)
    a = reference.arrays
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    data = mujoco.MjData(model)
    qa = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    joints = [model.joint(name).id for name in JOINT_NAMES]
    lower, upper = model.jnt_range[joints].T
    recomputed = arrays_from_qpos(reference.time_s, a["qpos"])
    errors = {key: float(np.max(np.abs(a[key]-value))) for key, value in recomputed.items()
              if key in a and value.dtype.kind in "fc"}
    grid = np.arange(0., reference.time_s[-1]+1e-8, .005)
    poses = reference.sample(grid)["qpos"]
    glove_ids = [model.site(side+"_glove_center").id for side in ("left", "right")]
    head = model.geom("head_collision").id
    feet = [model.geom(side+"_foot_sole_collision").id for side in ("left", "right")]
    gloves, own_heads, soles, foot_centers = [], [], [], []
    deepest = {}
    for time_s, q in zip(grid, poses):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        gloves.append(data.site_xpos[glove_ids].copy())
        own_heads.append(data.geom_xpos[head].copy())
        foot_centers.append(data.geom_xpos[feet].copy())
        soles.append([data.geom_xpos[gid, 2]-np.dot(np.abs(data.geom_xmat[gid].reshape(3, 3)[2]), model.geom_size[gid]) for gid in feet])
        for contact in data.contact[:data.ncon]:
            if contact.dist < 0:
                pair = "|".join(sorted((model.geom(contact.geom1).name, model.geom(contact.geom2).name)))
                if pair not in deepest or contact.dist < deepest[pair]["signed_distance_m"]:
                    deepest[pair] = {"signed_distance_m": float(contact.dist), "time_s": float(time_s)}
    gloves = np.asarray(gloves)
    own_heads = np.asarray(own_heads)
    soles = np.asarray(soles)
    foot_centers = np.asarray(foot_centers)
    own_head_gap = np.linalg.norm(gloves-own_heads[:, None], axis=-1)-.08-float(model.geom_size[head, 0])
    speed = np.linalg.norm(np.gradient(gloves, grid, axis=0, edge_order=2), axis=-1)
    dq = np.abs(np.diff(a["joint_pos"], axis=0))
    worst_frame, worst_joint = np.unravel_index(np.argmax(dq), dq.shape)
    accel = np.gradient(a["joint_vel"], reference.time_s, axis=0, edge_order=2)
    source = reference.metadata["source"]
    recorded, _ = read_recording(Path(source["source_path"]), source["source_sha256"])
    positions = recorded["ma_position"]
    hand = source["active_hand_joint_index"]
    proximal = positions[:, hand-2]-positions[:, hand-1]
    distal = positions[:, hand]-positions[:, hand-1]
    flexion = np.pi-np.arccos(np.clip(np.sum(proximal*distal, axis=-1)
        /np.linalg.norm(proximal, axis=-1)/np.linalg.norm(distal, axis=-1), -1., 1.))
    sampled_flexion = np.interp(a["source_frames"], np.arange(len(flexion)), flexion)
    elbow_index = JOINT_NAMES.index(reference.metadata["active_hand"]+"_elbow_joint")
    return {"primitive_id": reference.metadata["primitive_id"], "family": reference.metadata["family"],
            "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": reference.metadata["source"]["source_sha256"],
            "source_frames": reference.metadata["source"]["source_frame_range_inclusive"],
            "source_extension_s": reference.metadata["strike_time_s"], "output_end_s": float(reference.time_s[-1]),
            "all_numeric_arrays_finite": all(np.isfinite(v).all() for v in a.values() if v.dtype.kind in "fc"),
            "recomputed_array_max_errors": errors,
            "initial_and_terminal_pelvis_xyz_m": a["qpos"][[0, -1], :3].tolist(),
            "initial_and_terminal_pelvis_quaternion_wxyz": a["qpos"][[0, -1], 3:7].tolist(),
            "initial_and_terminal_max_abs_joint_speed_rad_s": np.abs(a["joint_vel"][[0, -1]]).max(axis=1).tolist(),
            "initial_and_terminal_root_linear_speed_m_s": np.linalg.norm(a["qvel"][[0, -1], :3], axis=-1).tolist(),
            "initial_and_terminal_glove_speed_m_s_left_right": speed[[0, -1]].tolist(),
            "maximum_joint_speed_rad_s": float(np.abs(a["joint_vel"]).max()),
            "maximum_joint_acceleration_rad_s2": float(np.abs(accel).max()),
            "maximum_20ms_joint_step_rad": float(dq.max()), "maximum_step_joint": JOINT_NAMES[worst_joint],
            "maximum_step_interval_s": a["time_s"][[worst_frame, worst_frame+1]].tolist(),
            "maximum_joint_limit_excess_rad": float(np.maximum(np.maximum(lower-a["qpos"][:, qa], a["qpos"][:, qa]-upper), 0).max()),
            "source_vs_native_active_elbow": {
                "source_maximum_step_rad_at_matching_output_times": float(np.abs(np.diff(sampled_flexion)).max()),
                "native_maximum_step_rad": float(np.abs(np.diff(a["joint_pos"][:, elbow_index])).max()),
                "source_maximum_speed_rad_s_at_matching_output_times": float(np.abs(np.gradient(sampled_flexion, reference.time_s)).max()),
                "native_maximum_speed_rad_s": float(np.abs(a["joint_vel"][:, elbow_index]).max()),
                "note": "Source elbow flexion from recorded shoulder/elbow/wrist positions; native single-hinge coordinate differs morphologically. Matching-time derivative comparison describes source rate versus retargeting amplification, not calibrated biomechanics."},
            "native_filtered_penetrating_pairs": deepest,
            "minimum_glove_to_own_head_surface_gap_m_left_right": own_head_gap.min(axis=0).tolist(),
            "minimum_own_head_gap_times_s_left_right": grid[np.argmin(own_head_gap, axis=0)].tolist(),
            "minimum_glove_to_glove_surface_gap_m": float(np.linalg.norm(gloves[:, 0]-gloves[:, 1], axis=-1).min()-.16),
            "minimum_sole_vertex_m_left_right": soles.min(axis=0).tolist(),
            "foot_center_net_displacement_m_left_right": (foot_centers[-1]-foot_centers[0]).tolist(),
            "diagnostic_scope": "Native FK every5ms; inherited native self-contact exclusions and parent filtering. H2 prescribed replay disables these contacts. Own H2 head radius0.07m is distinct from G1 hazard head radius0.06m. No physical feasibility or acceptance threshold is implied.",
            "endpoint_note": "Raw source-derived endpoints may retain motion; check the adapted complete primitive's smooth recovery/terminal hold before selection."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/task10d/phase2/variants/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task10d/phase2/variants/raw_checks.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / "outputs/task10d/phase2").resolve())
    rows = [inspect_reference(path) for path in sorted(args.root.glob("*/*.npz"))]
    output.write_text(json.dumps({"schema": "task10d_raw_variant_checks_v1", "references": rows}, indent=2)+"\n")
    for row in rows:
        print(json.dumps({k: row[k] for k in ("primitive_id", "maximum_20ms_joint_step_rad", "maximum_step_joint", "initial_and_terminal_max_abs_joint_speed_rad_s", "native_filtered_penetrating_pairs", "minimum_glove_to_own_head_surface_gap_m_left_right")}), flush=True)


if __name__ == "__main__":
    main()
