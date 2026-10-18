"""Constrained full-cycle DuoBox punch transfer to the exact scalar human MJCF.

References are geometric targets, not evidence of physical execution. Source
coordinates are never copied into scalar joints. IK uses compiled MuJoCo FK and
Jacobians, anatomical ranges, planted soles, both glove poses and temporal priors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .contact import first_contact_time
from .motion import CROP_SPECS
from .physical import interpolate_pose

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_RUNS = {"jab": "physical-5890952", "cross": "physical-5890924",
               "left_hook": "physical-5890924", "right_hook": "physical-5891006"}


def cycle_clock(times, source_duration, speed, initial_hold, return_duration):
    """Explicit phase labels; playback speed does not shorten guard recovery.

    Phases are 0=initial guard, 1=source playback, 2=return blend,
    3=final guard. The attack window includes the source crop's own recovery;
    it is an engineering window, not a reviewed capture outward-phase label.
    """
    times = np.asarray(times)
    return_start = initial_hold + source_duration / speed
    guard_start = return_start + return_duration
    phase = np.select([times <= initial_hold, times <= return_start,
                       times < guard_start], [0, 1, 2], default=3).astype(np.int8)
    source_time = np.clip((times-initial_hold)*speed, 0., source_duration)
    return phase, source_time, np.array([initial_hold, return_start])


def rotation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def derivatives(model, times, poses):
    """Central differences in MuJoCo tangent coordinates, including free root."""
    velocity = np.zeros((len(poses), model.nv))
    for i in range(1, len(poses) - 1):
        mujoco.mj_differentiatePos(model, velocity[i], times[i+1] - times[i-1],
                                   poses[i-1], poses[i+1])
    return velocity


class PoseSolver:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.address = model.jnt_qposadr[self.joint_ids]
        self.lower, self.upper = model.jnt_range[self.joint_ids].T
        self.pelvis = model.body("Hips").id

    def solve(self, initial, points, rotations, *, prior=None, iterations=65,
              posture_weight=.025):
        model, data = self.model, self.data
        pose = initial.copy()
        prior = initial if prior is None else prior
        # Support-center regularization moves joints/root only in offline IK.
        soles = [target for kind, name, target, weight in points if kind == "site"
                 and name.endswith("_sole")]
        support = np.mean(soles, axis=0)
        for _ in range(iterations):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            rows, errors = [], []
            for kind, name, target, weight in points:
                if kind == "site":
                    index = model.site(name).id
                    current = data.site_xpos[index]
                    mujoco.mj_jacSite(model, data, self.jp, self.jr, index)
                else:
                    index = model.body(name).id
                    current = data.xpos[index]
                    mujoco.mj_jacBody(model, data, self.jp, self.jr, index)
                rows.append(weight * self.jp.copy())
                errors.append(weight * (target - current))
            for kind, name, target, weight in rotations:
                if kind == "site":
                    index = model.site(name).id
                    current = data.site_xmat[index].reshape(3, 3)
                    mujoco.mj_jacSite(model, data, self.jp, self.jr, index)
                else:
                    index = model.body(name).id
                    current = data.xmat[index].reshape(3, 3)
                    mujoco.mj_jacBody(model, data, self.jp, self.jr, index)
                rows.append(weight * self.jr.copy())
                errors.append(weight * rotation_error(target, current))
            mujoco.mj_jacSubtreeCom(model, data, self.jp, self.pelvis)
            rows.append(.5 * self.jp[:2].copy())
            errors.append(.5 * (support[:2] - data.subtree_com[self.pelvis, :2]))
            # Explicit pairs are selected by the asset. Avoid penetrating any
            # selected self-contact; floor contact is separately fixed by soles.
            for contact in data.contact:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                if not b1 or not b2 or contact.dist >= .004:
                    continue
                normal = contact.frame.reshape(3, 3)[0]
                mujoco.mj_jac(model, data, self.jp, self.jr, contact.pos, b1)
                first = self.jp.copy()
                mujoco.mj_jac(model, data, self.jp, self.jr, contact.pos, b2)
                rows.append(5. * (normal @ (self.jp - first))[None])
                errors.append(np.array([5. * (.004 - contact.dist)]))
            displacement = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, displacement, 1., pose, prior)
            weights = np.r_[np.full(6, posture_weight*.4),
                            np.full(model.nv-6, posture_weight)]
            for joint in self.joint_ids:
                if model.joint(int(joint)).name.startswith("Neck"):
                    weights[model.jnt_dofadr[joint]] = .2
            rows.append(np.diag(weights))
            errors.append(weights * displacement)
            jac = np.vstack(rows)
            error = np.concatenate(errors)
            # Solve within limits, rather than solving for an impossible update
            # then clipping it: the latter stalls all linked tasks at a bound.
            low, high = np.full(model.nv, -.18), np.full(model.nv, .18)
            low[:3], high[:3] = -.04, .04
            low[6:] = np.maximum(low[6:], self.lower-pose[self.address])
            high[6:] = np.minimum(high[6:], self.upper-pose[self.address])
            delta = lsq_linear(np.vstack((jac, .045*np.eye(model.nv))),
                               np.r_[error, np.zeros(model.nv)], bounds=(low, high),
                               method="bvls", tol=1e-7, max_iter=40).x
            mujoco.mj_integratePos(model, pose, delta, 1.)
            pose[self.address] = np.clip(pose[self.address], self.lower, self.upper)
            if np.max(np.abs(delta)) < 2e-5:
                break
        return pose


def build_reference(model_path, output, *, source_dir=None, punch="jab", fps=60., speed=.6,
                    initial_hold=.5, return_duration=1.0, final_hold=.6):
    if punch not in CROP_SPECS:
        raise ValueError(f"Unknown punch family: {punch}")
    if not np.isfinite([fps, speed, initial_hold, return_duration, final_hold]).all():
        raise ValueError("Reference clocks must be finite")
    if fps <= 0 or speed <= 0 or return_duration <= 0 or min(initial_hold, final_hold) < 0:
        raise ValueError("Require positive speed/rate/return and nonnegative guard holds")
    source_dir = Path(source_dir or PROJECT / "outputs" / SOURCE_RUNS[punch])
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError("Reference outputs must remain in human-punching")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    source_path = source_dir / "model/human_physical.xml"
    source_model = mujoco.MjModel.from_xml_path(str(source_path))
    source_archive = source_dir / punch / "reference.npz"
    source = np.load(source_archive, allow_pickle=False)
    source_times, source_poses = source["time_s"], source["qpos"]
    if (source_times[0] != 0. or np.any(np.diff(source_times) <= 0)
            or not np.isfinite(source_times).all() or not np.isfinite(source_poses).all()):
        raise ValueError("Source needs finite increasing timestamps starting at zero and finite poses")
    source_data = mujoco.MjData(source_model)
    target_data = mujoco.MjData(model)
    solver = PoseSolver(model)
    source_bind = mujoco.MjData(source_model)
    source_bind.qpos[3:7] = source_poses[0, 3:7]
    for j in range(1, source_model.njnt):
        a = source_model.jnt_qposadr[j]
        source_bind.qpos[a:a+4] = [1., 0, 0, 0]
    mujoco.mj_forward(source_model, source_bind)
    mujoco.mj_forward(model, target_data)
    bind_map = {}
    for target, original in [("Hips", "Hips"), ("Chest", "Spine1"),
                             ("LeftHand", "LeftHand"), ("RightHand", "RightHand")]:
        bind_map[target] = (source_bind.xmat[source_model.body(original).id].reshape(3, 3).T
                            @ target_data.xmat[model.body(target).id].reshape(3, 3))
    foot_rotations = {side: target_data.site_xmat[model.site(f"{side}Foot_sole").id]
                     .reshape(3, 3).copy() for side in ("Left", "Right")}
    source_data.qpos[:] = source_poses[0]
    mujoco.mj_forward(source_model, source_data)
    foot_targets = {side: source_data.site_xpos[source_model.site(f"{side}Foot_sole").id]
                    .copy() for side in ("Left", "Right")}
    for target in foot_targets.values():
        target[2] = .0004
    cycle_duration = float(source_times[-1] / speed)
    duration = initial_hold + cycle_duration + return_duration + final_hold
    times = np.linspace(0, duration, round(duration * fps) + 1)
    phase, source_clock, attack_window = cycle_clock(
        times, source_times[-1], speed, initial_hold, return_duration)
    pose = model.qpos0.copy()
    pose[:3] = source_poses[0, :3] - [0., 0., .055]
    for side, sign in (("Left", -1), ("Right", 1)):
        for suffix, value in (("Arm_y", -.7 * sign), ("Arm_z", .6 * sign),
                              ("ForeArm_z", 2. * sign), ("Leg_x", -.15), ("Shin_x", .3)):
            pose[model.joint(side + suffix).qposadr[0]] = value
    poses, hand_targets = [], []
    first_pose = None
    for i, time in enumerate(times):
        phase_time = time - initial_hold
        if phase_time <= 0:
            original = source_poses[0]
        elif phase_time <= cycle_duration:
            original, _ = interpolate_pose(source_model, source_times, source_poses,
                                            phase_time * speed)
        else:
            u = np.clip((phase_time-cycle_duration)/return_duration, 0, 1)
            blend = u**3 * (10 - 15*u + 6*u*u)
            original = source_poses[-1].copy()
            dv = np.zeros(source_model.nv)
            mujoco.mj_differentiatePos(source_model, dv, 1., original, source_poses[0])
            mujoco.mj_integratePos(source_model, original, dv, blend)
        source_data.qpos[:] = original
        mujoco.mj_forward(source_model, source_data)
        points = [("body", "Hips", original[:3] - [0., 0., .055], .8)]
        rotations = []
        for target, original_name in (("Hips", "Hips"), ("Chest", "Spine1")):
            rotation = source_data.xmat[source_model.body(original_name).id].reshape(3, 3)
            rotations.append(("body", target, rotation @ bind_map[target], .35))
        target_hands = []
        for side in ("Left", "Right"):
            hand_target = source_data.site_xpos[source_model.site(f"{side}Hand_center").id].copy()
            points.extend([
                ("site", f"{side}Hand_center", hand_target, 5.),
                ("site", f"{side}Foot_sole", foot_targets[side], 12.),
                ("body", f"{side}ForeArm", source_data.xpos[source_model.body(f"{side}ForeArm").id], .45),
                ("body", f"{side}Arm", source_data.xpos[source_model.body(f"{side}Arm").id], .4),
                ("body", f"{side}Shin", source_data.xpos[source_model.body(f"{side}Leg").id], .25),
            ])
            wrist = source_data.xmat[source_model.body(f"{side}Hand").id].reshape(3, 3)
            rotations.append(("body", f"{side}Hand", wrist @ bind_map[f"{side}Hand"], .18))
            rotations.append(("site", f"{side}Foot_sole", foot_rotations[side], 1.5))
            target_hands.append(hand_target)
        if first_pose is not None and (phase_time <= 0 or phase_time >= cycle_duration+return_duration):
            pose = first_pose.copy()
        else:
            pose = solver.solve(pose, points, rotations,
                                prior=first_pose, iterations=400 if i == 0 else 45,
                                posture_weight=0. if i == 0 else .025)
        if first_pose is None:
            first_pose = pose.copy()
        poses.append(pose.copy())
        hand_targets.append(target_hands)
        if i % 60 == 0:
            print(f"scalar IK {punch} {i}/{len(times)}", flush=True)
    poses = np.asarray(poses)
    velocities = derivatives(model, times, poses)
    sites, bodies, quats, min_self = [], [], [], []
    body_ids = np.arange(1, model.nbody)
    for q, v in zip(poses, velocities, strict=True):
        target_data.qpos[:], target_data.qvel[:] = q, v
        mujoco.mj_forward(model, target_data)
        sites.append(target_data.site_xpos.copy())
        bodies.append(target_data.xpos[body_ids].copy())
        quats.append(target_data.xquat[body_ids].copy())
        min_self.append(min([c.dist for c in target_data.contact if
                            model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]] + [0.]))
    sites, bodies, quats = np.asarray(sites), np.asarray(bodies), np.asarray(quats)
    hand_ids = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    foot_ids = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    scene = json.loads((PROJECT / "outputs/g1-targeting/scene/scene_metadata.json").read_text())
    head = np.asarray(scene["head_center_w"])
    gloves = sites[:, hand_ids]
    contact_time = first_contact_time(times, gloves, head)
    active_index = 0 if CROP_SPECS[punch]["anatomical_hand"] == "left" else 1
    active_contact = first_contact_time(times, gloves[:, [active_index, active_index]], head)
    # Keep recovery-only contacts from masquerading as successful attacks.
    # This geometric annotation is deliberately marked as derived/unreviewed;
    # it is not a human-labelled outward/follow-through phase from capture.
    active_distance = np.linalg.norm(gloves[:, active_index]-head, axis=-1)
    source_mask = (times >= initial_hold) & (times <= attack_window[1])
    source_indices = np.flatnonzero(source_mask)
    closest_index = int(source_indices[np.argmin(active_distance[source_mask])])
    precise_attack_window = np.array([initial_hold, min(attack_window[1], times[closest_index]+.15)])
    window_semantics = "derived: inserted launch through closest active-glove approach plus 150ms follow-through; not capture-reviewed"
    conditioning_path = source_dir / punch / "conditioning.json"
    conditioning = json.loads(conditioning_path.read_text()) if conditioning_path.exists() else None
    acceleration = np.gradient(velocities, times, axis=0)
    glove_velocity = np.gradient(gloves, times, axis=0)
    boundary_steps = {}
    for name, boundary in (("launch", initial_hold), ("return", attack_window[1]),
                           ("guard", initial_hold+cycle_duration+return_duration)):
        index = np.searchsorted(times, boundary)
        nearby = poses[max(0,index-2):min(len(poses), index+3), solver.address]
        boundary_steps[name] = float(np.abs(np.diff(nearby, axis=0)).max())
    # Annotation convention: source motion begins after explicit quiet hold;
    # this is not an inferred capture label or a requested-TTC certification.
    metadata = dict(schema="scalar_punch_reference_v2", punch=punch,
        active_hand=CROP_SPECS[punch]["anatomical_hand"], active_hand_index=active_index,
        model=str(model_path),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        source_model=str(source_path), source_archive=str(source_archive),
        source_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        source_kind=f"conditioned, head-directed DuoBox {punch}; not raw untouched capture",
        source_conditioning=conditioning,
        source_capture_candidate=CROP_SPECS[punch],
        method="bounded compiled-model IK, both glove poses, stationary soles, return blend",
        solver="bounded variable least squares in MuJoCo tangent coordinates",
        reference_pelvis_lowering_m=.055,
        joint_limit_violation_max_rad=float(max(0., np.max(solver.lower-poses[:,solver.address]),
                                                np.max(poses[:,solver.address]-solver.upper))),
        endpoint_configuration_max_difference=float(np.abs(poses[-1]-poses[0]).max()),
        joint_step_max_rad=float(np.abs(np.diff(poses[:,solver.address],axis=0)).max()),
        boundary_joint_step_max_rad=boundary_steps,
        frames=len(times), fps=float(1/np.diff(times).mean()), source_speed=speed,
        duration_s=duration, initial_hold_s=initial_hold, launch_time_s=initial_hold,
        launch_annotation="explicit source-playback start after inserted quiet guard; no prescribed TTC",
        return_start_s=initial_hold+cycle_duration,
        attack_window_s=attack_window.tolist(),
        attack_window_annotation="source playback including source recovery; excludes inserted final return and guard",
        phase_labels={"0": "initial_guard", "1": "source_playback", "2": "return_blend", "3": "final_guard"},
        attack_window_reference_s=precise_attack_window.tolist(),
        attack_window_semantics=window_semantics, attack_window_reviewed=False,
        closest_active_glove_time_s=float(times[closest_index]),
        guard_hold_start_s=initial_hold+cycle_duration+return_duration,
        head_center=head.tolist(), minimum_glove_head_distance_m=np.linalg.norm(gloves-head,axis=-1).min(axis=0).tolist(),
        first_contact_time_s=float(contact_time) if np.isfinite(contact_time) else None,
        active_first_contact_time_s=float(active_contact) if np.isfinite(active_contact) else None,
        contact_delay_from_playback_launch_s=float(contact_time-initial_hold) if np.isfinite(contact_time) else None,
        glove_target_max_error_m=float(np.linalg.norm(gloves-np.asarray(hand_targets),axis=-1).max()),
        sole_height_range_m=[float(sites[:,foot_ids,2].min()),float(sites[:,foot_ids,2].max())],
        sole_drift_max_m=float(np.linalg.norm(sites[:,foot_ids]-sites[0,foot_ids],axis=-1).max()),
        selected_self_penetration_max_m=float(-min(min_self)),
        joint_velocity_max_rad_s=float(np.abs(velocities[:,6:]).max()),
        joint_acceleration_max_rad_s2=float(np.abs(acceleration[:,6:]).max()),
        glove_velocity_max_m_s=np.linalg.norm(glove_velocity, axis=-1).max(axis=0).tolist(),
        root_external_assistance="not applicable: offline kinematic reference",
        physical_execution_validated=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/f"{punch}.npz", t=times, time=times, time_s=times, qpos=poses,
        qvel=velocities, head_center=head, body_pos=bodies, body_quat=quats,
        body_names=np.array([model.body(i).name for i in body_ids]),
        joint_names=np.array([model.joint(i).name for i in solver.joint_ids]),
        site_pos=sites, glove_centers=gloves, launch_time=initial_hold,
        source_time_s=source_clock, phase=phase, attack_window=attack_window,
        punch=np.array(punch), punch_family=np.array(punch),
        active_hand=np.array(CROP_SPECS[punch]["anatomical_hand"].title()),
        source_speed=speed, return_start=initial_hold+cycle_duration,
        attack_window_reference_s=precise_attack_window,
        attack_window_semantics=np.array(window_semantics), attack_window_reviewed=False,
        active_hand_index=active_index,
        first_contact_time=contact_time, active_first_contact_time=active_contact,
        guard_hold_start=metadata["guard_hold_start_s"])
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT/"outputs/scalar-body/human.xml")
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/scalar-reference")
    parser.add_argument("--source-dir", type=Path, help="Override conditioned source run")
    parser.add_argument("--punch", choices=list(CROP_SPECS), default="jab")
    parser.add_argument("--speed", type=float, default=.6)
    parser.add_argument("--fps", type=float, default=60.)
    parser.add_argument("--initial-hold", type=float, default=.5)
    parser.add_argument("--return-duration", type=float, default=1.)
    parser.add_argument("--final-hold", type=float, default=.6)
    args = parser.parse_args()
    print(json.dumps(build_reference(args.model, args.output, source_dir=args.source_dir,
                                    punch=args.punch, speed=args.speed, fps=args.fps,
                                    initial_hold=args.initial_hold,
                                    return_duration=args.return_duration,
                                    final_hold=args.final_hold), indent=2))


if __name__ == "__main__":
    main()
