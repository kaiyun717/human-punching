"""Generate and optionally render head-contact references with both-glove labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .contact import (
    CONTACT_RADIUS_M,
    GEOMETRY_VERSION,
    GLOVE_RADIUS_M,
    HEAD_RADIUS_M,
    SAFE_CLEARANCE_RADIUS_M,
    evaluate_clearance,
    first_contact_time,
)
from .motion import CROP_SPECS

PROJECT = Path(__file__).resolve().parents[2]


def reference_contact_metrics(model, trajectory, head_center):
    """Measure actual compiled attachments for both gloves at every saved pose."""
    import mujoco

    ids = [model.geom(f"{side}_boxing_glove").id for side in ("Left", "Right")]
    data = mujoco.MjData(model)
    gloves = np.empty((len(trajectory["time_s"]), 2, 3))
    for i, pose in enumerate(trajectory["qpos"]):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        gloves[i] = data.geom_xpos[ids]
    labels = evaluate_clearance(gloves, head_center)
    first = first_contact_time(trajectory["time_s"], gloves, head_center)
    sampled = np.flatnonzero(labels.any_contact)
    arrays = {
        "glove_centers_w": gloves,
        "glove_head_distance_m": labels.distances_m,
        "glove_contact_clearance_m": labels.contact_clearance_m,
        "glove_safety_clearance_m": labels.safety_clearance_m,
        "any_glove_contact": labels.any_contact,
        "fist_clearance_safe": labels.fist_clearance_safe,
    }
    metrics = {
        "geometry_version": GEOMETRY_VERSION,
        "contact_radius_m": CONTACT_RADIUS_M,
        "safe_clearance_radius_m": SAFE_CLEARANCE_RADIUS_M,
        "head_proxy_radius_m": HEAD_RADIUS_M,
        "glove_proxy_radius_m": GLOVE_RADIUS_M,
        "glove_order": ["left", "right"],
        "minimum_center_distance_each_glove_m": labels.distances_m.min(axis=0).tolist(),
        "minimum_safety_clearance_m": float(labels.min_safety_clearance_m.min()),
        "any_proxy_contact": bool(labels.any_contact.any()),
        "sampled_contact_count": int(labels.any_contact.sum()),
        "sampled_unsafe_clearance_count": int((~labels.fist_clearance_safe).sum()),
        "first_sampled_contact_time_s": (
            float(trajectory["time_s"][sampled[0]]) if len(sampled) else None
        ),
        "first_piecewise_linear_contact_time_s": first if np.isfinite(first) else None,
        "frozen_head_ttc_from_clip_start_s": (
            first - float(trajectory["time_s"][0]) if np.isfinite(first) else None
        ),
        "ttc_clock": "Clip start is diagnostic commitment time; no launch-phase annotation or requested TTC imposed",
        "contact_response": "Analytic sphere overlap only; native physical impact disabled",
        "attachment_local_center_m": model.geom_pos[ids].tolist(),
    }
    return arrays, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "outputs/preview")
    parser.add_argument(
        "--scene", type=Path,
        default=PROJECT / "outputs/g1-targeting/scene/scene_metadata.json",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/contact-targeting")
    parser.add_argument("--punches", nargs="+", choices=list(CROP_SPECS), default=list(CROP_SPECS))
    parser.add_argument("--approach-yaws-deg", nargs="+", type=float, default=[0.0])
    parser.add_argument("--center-target", type=float, default=.135)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--backend", choices=("osmesa", "egl"), default="osmesa")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("Outputs must stay in human-punching")
    if not np.isfinite(args.approach_yaws_deg).all():
        parser.error("Approach yaws must be finite")
    os.environ["MUJOCO_GL"] = args.backend
    import mujoco

    from .retarget import sha256
    from .target_cli import compose_trajectory
    from .targeting import aim_clip

    source = args.source.resolve()
    scene_path = args.scene.resolve()
    scene = json.loads(scene_path.read_text())
    scene["human_height_m"] = 1.80
    scene["g1_base_position_w"] = scene["robot_pelvis_position_w"]
    human_path = source / "model/human.xml"
    human = mujoco.MjModel.from_xml_path(str(human_path))
    model = mujoco.MjModel.from_xml_path(scene["xml_path"])
    head = np.asarray(scene["head_center_w"])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "human_punching_contact_reference_v1",
        "geometry_version": GEOMETRY_VERSION,
        "source_model": str(human_path),
        "source_model_sha256": sha256(human_path),
        "scene_metadata": str(scene_path),
        "scene_xml": scene["xml_path"],
        "center_target_m": args.center_target,
        "scope": "Kinematic source-timing reference bank; geometric contact accepted, dynamic feasibility untested",
        "references": {},
    }
    stills = []
    for punch in args.punches:
        with np.load(source / punch / "trajectory.npz", allow_pickle=False) as saved:
            original = dict(saved)
        for yaw in args.approach_yaws_deg:
            key = punch if yaw == 0 else f"{punch}_yaw_{yaw:+g}deg"
            directory = output / key
            directory.mkdir(exist_ok=True)
            print(f"Solving {key}", flush=True)
            aimed = aim_clip(
                human, original, punch, head,
                target_mode="proxy_contact", contact_center_distance_m=args.center_target,
                approach_yaw_offset_rad=float(np.deg2rad(yaw)),
            )
            arrays, contact = reference_contact_metrics(human, aimed["trajectory"], head)
            human_trajectory = {**aimed["trajectory"], **arrays}
            if not contact["any_proxy_contact"]:
                raise RuntimeError(f"{key} failed to enter the accepted contact envelope")
            combined, scene_metrics = compose_trajectory(model, human_trajectory, scene, punch)
            combined["source_human_qpos"] = original["qpos"]
            metrics = {**aimed["metrics"], **scene_metrics, **contact, "punch": punch,
                       "variant": key, "approach_yaw_offset_deg": yaw,
                       "head_center_w": head.tolist(),
                       "contact_frame_index": scene_metrics["closest_frame_index"],
                       "minimum_clearance_m": scene_metrics["minimum_proxy_clearance_m"]}
            np.savez_compressed(directory / "human_trajectory.npz", **human_trajectory)
            np.savez_compressed(directory / "trajectory.npz", **combined)
            metrics["trajectory_sha256"] = sha256(directory / "trajectory.npz")
            metrics["human_trajectory_sha256"] = sha256(directory / "human_trajectory.npz")
            metrics["human_trajectory_path"] = str(directory / "human_trajectory.npz")
            if args.render:
                from .target_render import render_target_clip

                metrics["media"] = render_target_clip(model, combined, metrics, directory, scene)
                if yaw == 0:
                    stills.append(Path(metrics["media"]["still"]))
            (directory / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
            report["references"][key] = metrics
            print(f"{key}: minimum {scene_metrics['minimum_center_distance_m']:.6f} m; "
                  f"first contact {contact['first_piecewise_linear_contact_time_s']:.6f} s",
                  flush=True)
    if len(stills) == 4:
        from .render import make_overview

        make_overview(stills, output / "overview.png")
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(report['references'])} contact references to {output}", flush=True)


if __name__ == "__main__":
    main()
