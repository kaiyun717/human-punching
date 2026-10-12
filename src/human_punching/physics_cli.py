"""Build, condition, step, audit and render the free-root human physics pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .contact import evaluate_clearance, first_contact_time
from .motion import CROP_SPECS
from .physical import (
    bounded_pd,
    build_physical_human,
    controller_arrays,
    interpolate_pose,
    plant_feet,
    smooth_reference,
)
from .physical_audit import audit_reference
from .targeting import _export_trajectory, aim_clip

PROJECT = Path(__file__).resolve().parents[2]


def rollout(model, times, reference, *, standing=False, hold_s=2., gain_scale=1., speed=1., balance=False):
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    duration = 5. if standing else times[-1] / speed + hold_s
    dt = model.opt.timestep
    ids = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    foot_geoms = [model.geom(f"physical/{side}Foot").id for side in ("Left", "Right")]
    gloves = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    floor = model.geom("floor").id
    poses, velocities, stamps, desired, feet, hands, contact_counts = [], [], [], [], [], [], []
    com, effort_ratios, saturations, joint_errors, floor_depth = [], [], [], [], []
    error = np.zeros(model.nv)
    limits = controller_arrays(model)[2]
    root_up, root_wrench = [], []
    substep_times, substep_gloves = [], []
    for step in range(round(duration / dt) + 1):
        time = step * dt
        mujoco.mj_forward(model, data)
        substep_times.append(float(data.time))
        substep_gloves.append(data.site_xpos[gloves].copy())
        q, v = (reference[0], np.zeros(model.nv)) if standing else interpolate_pose(
            model, times, reference, time * speed)
        v = v * speed
        saturated, _ = bounded_pd(model, data, q, v, gain_scale=gain_scale, balance=balance)
        mujoco.mj_differentiatePos(model, error, 1., data.qpos, q)
        joint_errors.append(float(np.sqrt(np.mean(error[6:] ** 2))))
        saturations.append(saturated)
        effort_ratios.append(float(np.max(np.linalg.norm(data.ctrl.reshape(-1, 3), axis=1) / limits)))
        root_wrench.append(float(np.max(np.abs(data.qfrc_actuator[:6]))))
        if step % max(1, round(1 / (120 * dt))) == 0:
            # Save current physical state before the next step, not reference qpos.
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            desired.append(q.copy())
            stamps.append(data.time)
            feet.append(data.site_xpos[ids].copy())
            hands.append(data.site_xpos[gloves].copy())
            com.append(data.subtree_com[model.body("Hips").id].copy())
            root_up.append(float(data.xmat[model.body("Hips").id].reshape(3, 3)[2, 1]))
            contact_counts.append([sum(
                {int(c.geom1), int(c.geom2)} == {floor, g} for c in data.contact)
                for g in foot_geoms])
            floor_depth.append(min([c.dist for c in data.contact] + [0.]))
        if step < round(duration / dt):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 1000:
                raise RuntimeError(f"Simulation diverged at t={time:.3f}")
    poses, feet, contacts = np.array(poses), np.array(feet), np.array(contact_counts)
    initial_height = poses[0, 2]
    standing_ok = bool(poses[:, 2].min() > .70 * initial_height and min(root_up) > .70)
    metrics = dict(duration_s=duration, standing_only=standing, physical_stepping=True,
                   root_actuator_force_max=float(max(root_wrench)), external_root_assistance=False,
                   initial_pelvis_height_m=float(initial_height), min_pelvis_height_m=float(poses[:, 2].min()),
                   final_pelvis_height_m=float(poses[-1, 2]), root_up_min=float(min(root_up)),
                   remained_upright=standing_ok, joint_error_rms_rad=float(np.sqrt(np.mean(np.array(joint_errors)**2))),
                   max_torque_norm_fraction=float(max(effort_ratios)),
                   mean_joint_saturation_fraction=float(np.mean(saturations)),
                   floor_penetration_max_m=float(-min(floor_depth)),
                   each_foot_contact_fraction=np.mean(contacts > 0, axis=0).tolist(),
                   both_feet_contact_fraction=float(np.mean(np.all(contacts > 0, axis=1))),
                   max_sole_horizontal_drift_m=float(np.linalg.norm(feet[:, :, :2] - feet[0, :, :2], axis=-1).max()),
                   max_sole_height_m=float(feet[:, :, 2].max()), timestep_s=float(dt),
                   gain_scale=gain_scale, playback_speed=speed,
                   controller="bounded quaternion stable-PD; no learned policy", balance_feedforward=balance, impact_response_g1=False)
    return dict(qpos=poses, qvel=np.array(velocities), time_s=np.array(stamps),
                desired_qpos=np.array(desired), foot_centers_w=feet,
                glove_centers_w=np.array(hands), center_of_mass_w=np.array(com),
                foot_contact_counts=contacts, substep_time_s=np.array(substep_times),
                substep_glove_centers_w=np.array(substep_gloves)), metrics


def render(model, trajectory, metrics, output, title):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    from .render import font
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[3] = 0
    cameras = []
    for angle in (215, 100):
        c = mujoco.MjvCamera()
        c.lookat[:] = [.3, 0, .85]
        c.distance, c.azimuth, c.elevation = 3.8, angle, -10
        cameras.append(c)
    times = trajectory["time_s"]
    indices = np.searchsorted(times, np.arange(0, times[-1], 1/30)).clip(0, len(times)-1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    center_distances = None
    if "head_center_w" in metrics and "glove_centers_w" in trajectory:
        center_distances = np.linalg.norm(trajectory["glove_centers_w"] - np.asarray(metrics["head_center_w"]), axis=-1).min(axis=-1)
    still_index = int(indices[len(indices)//2])
    if center_distances is not None:
        contact_indices = indices[center_distances[indices] <= .14]
        still_index = int(contact_indices[0]) if len(contact_indices) else int(indices[np.argmin(center_distances[indices])])
    with imageio.get_writer(output / f"{title}.mp4", fps=30, codec="libx264", quality=8,
                            ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for index in indices:
            data.qpos[:] = trajectory["qpos"][index]
            data.qvel[:] = trajectory["qvel"][index]
            mujoco.mj_forward(model, data)
            canvas = Image.new("RGB", (1280, 720), (15, 22, 32))
            for i, camera in enumerate(cameras):
                renderer.update_scene(data, camera=camera, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render().copy()), (640*i, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 15), f"PHYSICAL HUMAN / {title.upper()}", font=font(27), fill="white")
            draw.text((20, 680), "PHYSICAL HUMAN | FREE ROOT | BOUNDED PD | G1 FIXED / ANALYTIC CONTACT" if model.nbody > 22 else "PHYSICAL HUMAN | FREE ROOT | BOUNDED PD | NO ROOT ASSISTANCE",
                      font=font(17), fill=(180, 200, 215))
            draw.text((735, 22), f"SOURCE SPEED {metrics.get('playback_speed', 1.):.2f}x", font=font(18), fill=(190, 208, 220))
            if center_distances is not None:
                distance = center_distances[index]
                state = "CONTACT" if distance <= .14 else "UNSAFE CLEARANCE" if distance < .16 else "CLEARANCE SAFE"
                draw.rectangle((16, 623, 515, 652), fill=(15, 22, 32))
                draw.text((25, 628), f"Glove / head center: {distance:.5f} m  |  {state}", font=font(16), fill=(255, 215, 135))
            draw.text((1035, 22), f"t={times[index]:.2f} s", font=font(19), fill="white")
            writer.append_data(np.asarray(canvas))
            if index == still_index:
                canvas.save(output / f"{title}.png")
    renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/physics-pilot")
    parser.add_argument("--input", type=Path, default=PROJECT / "outputs/preview")
    parser.add_argument("--punches", nargs="+", choices=list(CROP_SPECS), default=["jab"])
    parser.add_argument("--timestep", type=float, default=.002)
    parser.add_argument("--gain-scale", type=float, default=1.)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--reaim", action="store_true")
    parser.add_argument("--smooth", type=float, default=0.)
    parser.add_argument("--center-target", type=float, default=.135)
    parser.add_argument("--with-g1", action="store_true")
    parser.add_argument("--stance-forward", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = build_physical_human(PROJECT / "outputs/preview/model/human.xml", args.output / "model",
                                    timestep=args.timestep)
    if args.with_g1:
        from .physical_scene import attach_static_g1
        metadata["standing_g1_scene"] = attach_static_g1(Path(metadata["xml_path"]), args.output / "scene")
        metadata["xml_path"] = metadata["standing_g1_scene"]["xml_path"]
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    report = {"model": metadata, "runs": {},
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    for punch in args.punches:
        path = args.input / punch / "human_trajectory.npz"
        if not path.exists():
            path = args.input / punch / "trajectory.npz"
        with np.load(path) as archive:
            source = dict(archive)
        source["qpos"][:, 0] += args.stance_forward
        # Source toes are unnecessary articulation for this featureless pilot.
        # Suppress mocap toe rotations rather than expanding artificial limits.
        for side in ("Left", "Right"):
            address = int(model.joint(f"{side}ToeBase_joint").qposadr[0])
            source["qpos"][:, address:address+4] = [1, 0, 0, 0]
        if args.smooth > 0:
            source["qpos"] = smooth_reference(model, source["qpos"], args.smooth)
        reference, conditioning = plant_feet(model, source["qpos"])
        head = np.array(json.loads((PROJECT / "outputs/g1-targeting/scene/scene_metadata.json").read_text())["head_center_w"])
        if args.reaim:
            for _ in range(2):
                conditioned = _export_trajectory(model, source, reference)
                aimed = aim_clip(model, conditioned, punch, head, target_mode="proxy_contact",
                                  contact_center_distance_m=args.center_target)
                reference = aimed["trajectory"]["qpos"]
                reference, conditioning = plant_feet(model, reference)
            conditioned = _export_trajectory(model, source, reference)
            reference = aim_clip(model, conditioned, punch, head, target_mode="proxy_contact",
                                 contact_center_distance_m=args.center_target)["trajectory"]["qpos"]
        conditioning["stance_forward_offset_m"] = args.stance_forward
        conditioning["smoothing_sigma_frames"] = args.smooth
        conditioning["toe_preprocessing"] = "Both ToeBase rotations set to identity"
        conditioning["reaimed_after_support"] = args.reaim
        out = args.output / punch
        out.mkdir(exist_ok=True)
        audit = audit_reference(model, reference, source["time_s"] / args.speed, head)
        (out / "final-reference-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        np.savez_compressed(out / "reference.npz", qpos=reference, time_s=source["time_s"])
        (out / "conditioning.json").write_text(json.dumps(conditioning, indent=2) + "\n")
        print(punch, "conditioning", conditioning, flush=True)
        for standing in ([True, False] if punch == args.punches[0] else [False]):
            title = "standing" if standing else punch
            trajectory, metrics = rollout(model, source["time_s"], reference,
                standing=standing, gain_scale=args.gain_scale, speed=args.speed, balance=args.balance)
            labels = evaluate_clearance(trajectory["substep_glove_centers_w"], head)
            metrics["minimum_glove_head_distance_m"] = labels.distances_m.min(axis=0).tolist()
            metrics["any_proxy_contact"] = bool(labels.any_contact.any())
            metrics["any_unsafe_clearance"] = bool((~labels.fist_clearance_safe).any())
            contact_time = first_contact_time(trajectory["substep_time_s"], trajectory["substep_glove_centers_w"], head)
            metrics["first_contact_time_s"] = float(contact_time) if np.isfinite(contact_time) else None
            metrics["head_center_w"] = head.tolist()
            metrics["event_sampling_s"] = float(model.opt.timestep)
            metrics["support_gate"] = bool(metrics["remained_upright"] and
                metrics["max_sole_horizontal_drift_m"] <= .03 and metrics["max_sole_height_m"] <= .03 and
                metrics["floor_penetration_max_m"] <= .01 and metrics["both_feet_contact_fraction"] >= .95)
            metrics["isolated_punch_gate"] = bool(metrics["support_gate"] and metrics["any_proxy_contact"] and
                metrics["mean_joint_saturation_fraction"] <= .05)
            metrics["reference_center_target_m"] = args.center_target
            metrics["conditioning"] = conditioning
            metrics["reference_file"] = str(path.resolve())
            np.savez_compressed(out / f"{title}_rollout.npz", **trajectory)
            (out / f"{title}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            report["runs"][title] = metrics
            print(title, json.dumps(metrics), flush=True)
            if args.render:
                render(model, trajectory, metrics, out, title)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
