"""Aim the four human punch references toward a pose-held G1 in native MuJoCo."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .motion import CROP_SPECS, load_crop

PROJECT = Path(__file__).resolve().parents[2]


def compose_trajectory(model, human_trajectory: dict, scene: dict, punch: str) -> tuple[dict, dict]:
    """Scatter human motion into a combined model while holding every G1 coordinate."""
    import mujoco

    result = dict(human_trajectory)
    count = len(result["time_s"])
    human_q = np.asarray(scene["human_qpos_indices"], dtype=int)
    human_v = np.asarray(scene["human_dof_indices"], dtype=int)
    g1_q = np.asarray(scene["g1_qpos_indices"], dtype=int)
    result["qpos"] = np.tile(scene["qpos_reset"], (count, 1))
    result["qvel"] = np.zeros((count, model.nv))
    result["qpos"][:, human_q] = human_trajectory["qpos"]
    result["qvel"][:, human_v] = human_trajectory["qvel"]
    result["human_qpos_indices"] = human_q
    result["human_dof_indices"] = human_v
    result["g1_qpos_indices"] = g1_q
    result["g1_dof_indices"] = np.asarray(scene["g1_dof_indices"], dtype=int)
    result["head_center_w"] = np.asarray(scene["head_center_w"])
    side = "Left" if CROP_SPECS[punch]["active_hand_index"] == 8 else "Right"
    active_geom = model.geom(f"{side}_boxing_glove").id
    head_geom = model.geom(scene["head_geom_name"]).id
    head_mesh = model.mesh("g1/head_link").id
    visible_head_geoms = np.flatnonzero(
        (model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)) & (model.geom_dataid == head_mesh)
    )
    if len(visible_head_geoms) != 1:
        raise ValueError("Expected one visible G1 head mesh")

    def mesh_vertices(geom_id):
        mesh = int(model.geom_dataid[geom_id])
        first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        return model.mesh_vert[first : first + count]

    glove_vertices = mesh_vertices(active_geom)
    head_vertices = mesh_vertices(int(visible_head_geoms[0]))
    data = mujoco.MjData(model)
    gloves, heads, mesh_separation, visible_separation, projection_gap = [], [], [], [], []
    for i in range(count):
        data.qpos[:] = result["qpos"][i]
        data.qvel[:] = result["qvel"][i]
        mujoco.mj_forward(model, data)
        gloves.append(data.geom_xpos[active_geom].copy())
        heads.append(data.geom_xpos[head_geom].copy())
        mesh_separation.append(
            float(mujoco.mj_geomDistance(model, data, active_geom, head_geom, 2.0, None))
        )
        visible_separation.append(
            float(
                mujoco.mj_geomDistance(
                    model, data, active_geom, int(visible_head_geoms[0]), 2.0, None
                )
            )
        )
        visible_head = int(visible_head_geoms[0])
        direction = data.geom_xpos[active_geom] - data.geom_xpos[visible_head]
        direction /= np.linalg.norm(direction)
        glove_world = (
            glove_vertices @ data.geom_xmat[active_geom].reshape(3, 3).T
            + data.geom_xpos[active_geom]
        )
        head_world = (
            head_vertices @ data.geom_xmat[visible_head].reshape(3, 3).T
            + data.geom_xpos[visible_head]
        )
        projection_gap.append(
            float(np.min(glove_world @ direction) - np.max(head_world @ direction))
        )
    gloves, heads = np.asarray(gloves), np.asarray(heads)
    result["active_glove_pos_w"] = gloves
    result["target_distance_m"] = np.linalg.norm(gloves - heads, axis=1)
    result["glove_mesh_head_proxy_distance_m"] = np.asarray(mesh_separation)
    result["glove_mesh_visible_head_distance_m"] = np.asarray(visible_separation)
    result["glove_head_separating_axis_gap_m"] = np.asarray(projection_gap)
    movement = float(np.max(np.linalg.norm(heads - heads[0], axis=1)))
    g1_change = float(np.max(np.abs(result["qpos"][:, g1_q] - result["qpos"][0, g1_q])))
    if movement > 1e-10 or g1_change != 0:
        raise AssertionError("Standing G1 moved during the kinematic preview")
    metrics = {
        "g1_head_motion_max_m": movement,
        "g1_qpos_change_max": g1_change,
        "active_glove_geom_name": f"{side}_boxing_glove",
        "minimum_center_distance_m": float(result["target_distance_m"].min()),
        "minimum_proxy_clearance_m": float(
            result["target_distance_m"].min() - 0.08 - scene["head_radius_m"]
        ),
        "minimum_glove_mesh_to_head_proxy_m": float(np.min(mesh_separation)),
        "minimum_glove_mesh_to_visible_head_m": float(np.min(visible_separation)),
        "mesh_distance_convention": "Native MuJoCo convex mesh distance, diagnostic only; independent vertex-projection bound below",
        "minimum_glove_head_separating_axis_gap_m": float(np.min(projection_gap)),
        "separating_axis_convention": "min glove projection minus max visible-head projection along mesh-center direction; positive proves mesh separation at that frame",
        "closest_frame_index": int(np.argmin(result["target_distance_m"])),
        "combined_qpos_dimension": model.nq,
        "combined_qvel_dimension": model.nv,
        "execution": "Kinematic mj_forward replay; G1 pose held; no balance or impact dynamics",
    }
    return result, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT.parent / "DuoBox")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/g1-targeting")
    parser.add_argument("--height", type=float, default=1.80)
    parser.add_argument(
        "--robot-distance", type=float, default=0.85, help="G1 pelvis world X, metres"
    )
    parser.add_argument(
        "--robot-lateral", type=float, default=0.20, help="G1 pelvis world Y, metres"
    )
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.044,
        help="Extra gap beyond spherical proxies; default fits the actual visible G1 head mesh",
    )
    parser.add_argument(
        "--punches",
        nargs="+",
        choices=list(CROP_SPECS),
        default=["jab", "right_hook", "left_hook", "cross"],
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--backend", choices=["osmesa", "egl"], default=os.environ.get("MUJOCO_GL", "osmesa")
    )
    args = parser.parse_args(argv)
    if not np.isfinite(
        [args.height, args.robot_distance, args.robot_lateral, args.clearance]
    ).all():
        parser.error("Height, placement, and clearance must be finite")
    if not 0 <= args.clearance <= 0.10:
        parser.error("Clearance must be in [0, .10] metres")
    if not 1.2 <= args.height <= 2.3:
        parser.error("Height must be in [1.2, 2.3] metres")
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("Outputs must stay inside human-punching")
    os.environ["MUJOCO_GL"] = args.backend
    import mujoco

    from .g1_scene import create_scene
    from .human import build_human
    from .retarget import retarget_clip, sha256
    from .targeting import aim_clip

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    bind = load_crop(args.dataset, "jab")
    human = build_human(output / "human_model", bind.rest_offsets, args.height)
    human_model = mujoco.MjModel.from_xml_path(human["xml_path"])
    scene = create_scene(
        Path(human["xml_path"]),
        output / "scene",
        robot_position=(args.robot_distance, args.robot_lateral, 0.0),
    )
    scene["human_height_m"] = args.height
    scene["g1_base_position_w"] = scene["robot_pelvis_position_w"]
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    head = np.asarray(scene["head_center_w"])
    report = {
        "schema": "human_punching_standing_g1_ik_v1",
        "execution": "Offline IK; native MuJoCo kinematic replay; no learned policies or impact dynamics",
        "scene": scene,
        "human": human,
        "backend": "none" if args.no_render else args.backend,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ["mujoco", "numpy", "scipy", "imageio"]
        },
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_code_sha256": {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "punches": {},
    }
    stills = []
    print(f"G1 head target: {head.tolist()}", flush=True)
    for punch in args.punches:
        clip_dir = output / punch
        clip_dir.mkdir(exist_ok=True)
        clip = load_crop(args.dataset, punch, args.padding)
        source = retarget_clip(clip, human_model, human["scale"])
        print(f"Adapting {punch}: {len(clip.time_s)} frames", flush=True)
        aimed = aim_clip(
            human_model,
            source["trajectory"],
            punch,
            head,
            head_radius_m=scene["head_radius_m"],
            clearance_m=args.clearance,
        )
        trajectory, checks = compose_trajectory(model, aimed["trajectory"], scene, punch)
        trajectory["source_human_qpos"] = source["trajectory"]["qpos"]
        metrics = dict(aimed["metrics"])
        metrics.update(checks)
        metrics.update(
            punch=punch,
            source=clip.metadata,
            head_center_w=head.tolist(),
            contact_frame_index=checks["closest_frame_index"],
            minimum_clearance_m=checks["minimum_proxy_clearance_m"],
            scene_relative_path="../scene/scene.xml",
        )
        np.savez_compressed(clip_dir / "trajectory.npz", **trajectory)
        metrics["trajectory_sha256"] = sha256(clip_dir / "trajectory.npz")
        if not args.no_render:
            from .target_render import render_target_clip

            print(f"Rendering {punch} with {args.backend}", flush=True)
            metrics["media"] = render_target_clip(model, trajectory, metrics, clip_dir, scene)
            stills.append(Path(metrics["media"]["still"]))
        (clip_dir / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
        report["punches"][punch] = metrics
        print(
            f"{punch}: center distance {checks['minimum_center_distance_m']:.4f} m; "
            f"glove mesh/head proxy gap {checks['minimum_glove_mesh_to_head_proxy_m']:.4f} m",
            flush=True,
        )
    if len(stills) == 4:
        from .render import make_overview

        make_overview(stills, output / "overview.png")
    report["elapsed_s"] = time.monotonic() - started
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed targeting preview: {output} ({report['elapsed_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
