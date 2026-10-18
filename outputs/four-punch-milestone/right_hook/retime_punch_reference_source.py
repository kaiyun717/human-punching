#!/usr/bin/env python3
"""Apply a declared monotone phase warp to an existing scalar punch reference.

This preserves the configuration-space path by interpolation, not source-local
speed. Numeric derivatives are recomputed on the new clock, including warp
acceleration. It does not certify torque feasibility or physical execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from human_punching.contact import first_contact_time
from human_punching.physical import interpolate_pose
from human_punching.scalar_reference import derivatives

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retime(model_path, reference_path, output, knots):
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT / "outputs"):
        raise ValueError("Output must remain in human-punching/outputs")
    if output.exists():
        raise ValueError("Refuse to overwrite an existing reference")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(reference_path, allow_pickle=False) as archive:
        source = dict(archive)
    old_time = source["time_s"]
    old_launch, old_return, old_guard = (float(source[key]) for key in
                                          ("launch_time", "return_start", "guard_hold_start"))
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or knots.shape[1] != 2 or not np.isfinite(knots).all():
        raise ValueError("Knots must be finite old_time:new_time pairs")
    if np.any(np.diff(knots, axis=0) <= 0) or not np.allclose(knots[0], [0, 0]):
        raise ValueError("Both knot clocks must increase strictly from zero")
    if not np.isclose(knots[-1, 0], old_time[-1]):
        raise ValueError("Last source knot must equal the complete original duration")
    for boundary in (old_launch, old_return, old_guard):
        if not np.any(np.isclose(knots[:, 0], boundary, atol=1e-10, rtol=0)):
            raise ValueError("Launch, return and guard boundaries must be explicit knots")
    inverse = PchipInterpolator(knots[:, 1], knots[:, 0])
    duration = float(knots[-1, 1])
    times = np.linspace(0, duration, round(duration / np.diff(old_time).mean())+1)
    query = inverse(times)
    if np.any(np.diff(query) <= 0):
        raise ValueError("Interpolated phase warp is not strictly monotone")
    def boundary_new(boundary):
        return float(knots[np.flatnonzero(np.isclose(knots[:, 0], boundary, atol=1e-10, rtol=0))[0], 1])
    launch, return_start, guard = (boundary_new(t) for t in (old_launch, old_return, old_guard))
    poses = np.array([interpolate_pose(model, old_time, source["qpos"], t)[0] for t in query])
    poses[times <= launch] = source["qpos"][0]
    poses[times >= guard] = source["qpos"][0]
    velocity = derivatives(model, times, poses)
    acceleration = np.gradient(velocity, times, axis=0)
    data = mujoco.MjData(model)
    sites, bodies, quaternions, penetration = [], [], [], []
    for pose, speed in zip(poses, velocity, strict=True):
        data.qpos[:], data.qvel[:] = pose, speed
        mujoco.mj_forward(model, data)
        sites.append(data.site_xpos.copy())
        bodies.append(data.xpos[1:].copy())
        quaternions.append(data.xquat[1:].copy())
        penetration.append(min([c.dist for c in data.contact
                                if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]+[0.]))
    sites = np.array(sites)
    hands = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    soles = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    addresses = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    gloves, head = sites[:, hands], source["head_center"]
    active = int(source["active_hand_index"])
    distance = np.linalg.norm(gloves[:, active]-head, axis=-1)
    mask = (times >= launch) & (times <= return_start)
    closest = int(np.flatnonzero(mask)[np.argmin(distance[mask])])
    window = np.array([launch, min(return_start, times[closest]+.15)])
    contact = first_contact_time(times, gloves[:, [active, active]], head)
    phase = np.select([times <= launch, times <= return_start, times < guard], [0, 1, 2], default=3)
    nominal_source_speed = float(source.pop("source_speed", 1.))
    local_speed = inverse.derivative()(times) * nominal_source_speed
    semantics = "Nonuniform phase warp; 250ms borrowed from initial guard to lengthen the high-load swing; no uniform source-playback speed"
    source.update(t=times, time=times, time_s=times, qpos=poses, qvel=velocity,
        site_pos=sites, body_pos=np.array(bodies), body_quat=np.array(quaternions),
        glove_centers=gloves, launch_time=launch, return_start=return_start,
        guard_hold_start=guard, phase=phase, source_time_s=np.interp(query, old_time, source["source_time_s"]),
        attack_window=np.array([launch, return_start]), attack_window_reference_s=window,
        first_contact_time=first_contact_time(times, gloves, head), active_first_contact_time=contact,
        source_timing_modified=True, source_speed_semantics=np.array(semantics),
        timing_kind=np.array("nonuniform_pchip_phase_warp"), timing_description=np.array(semantics),
        source_speed_kind=np.array("nonuniform_local_rates"), nominal_source_speed=nominal_source_speed,
        local_source_speed_min=float(local_speed[mask].min()), local_source_speed_max=float(local_speed[mask].max()),
        retiming_duration_ratio=duration/old_time[-1],
        source_playback_duration_ratio=(return_start-launch)/(old_return-old_launch),
        phase_warp_knots=knots,
        local_source_speed=local_speed,
        unwarped_reference_time_s=query)
    family = str(source["punch_family"].item())
    report = dict(schema="scalar_punch_retimed_reference_v1", punch=family,
        base_reference=str(Path(reference_path).resolve()), base_reference_sha256=digest(reference_path),
        model=str(Path(model_path).resolve()), model_sha256=digest(model_path),
        nominal_source_speed=nominal_source_speed, source_speed_semantics=semantics,
        source_timing_modified=True, timing_kind="nonuniform_pchip_phase_warp",
        phase_warp_knots_old_new_s=knots.tolist(), duration_s=duration,
        launch_time_s=launch, return_start_s=return_start, guard_hold_start_s=guard,
        local_source_speed_min_max=[float(local_speed[mask].min()), float(local_speed[mask].max())],
        active_first_contact_time_s=float(contact) if np.isfinite(contact) else None,
        contact_delay_from_playback_launch_s=float(contact-launch) if np.isfinite(contact) else None,
        attack_window_reference_s=window.tolist(), attack_window_reviewed=False,
        minimum_glove_head_distance_m=np.linalg.norm(gloves-head, axis=-1).min(axis=0).tolist(),
        active_hand=str(source["active_hand"].item()), active_hand_index=active,
        joint_limit_violation_max_rad=float(max(0., np.max(lower-poses[:, addresses]), np.max(poses[:, addresses]-upper))),
        joint_velocity_max_rad_s=float(np.abs(velocity[:, 6:]).max()),
        joint_acceleration_max_rad_s2=float(np.abs(acceleration[:, 6:]).max()),
        endpoint_configuration_max_difference=float(np.abs(poses[0]-poses[-1]).max()),
        sole_drift_max_m=float(np.linalg.norm(sites[:, soles]-sites[0, soles], axis=-1).max()),
        sole_height_range_m=[float(sites[:, soles, 2].min()), float(sites[:, soles, 2].max())],
        selected_self_penetration_max_m=float(-min(penetration)), physical_execution_validated=False,
        derivative_method="MuJoCo tangent-space central differences on new clock, then numerical derivative; includes warp velocity and acceleration",
        path_limitation="Configuration-space interpolation preserves the sampled geometric path approximately; FK/support/contact are re-audited")
    output.mkdir(parents=True)
    np.savez_compressed(output / f"{family}.npz", **source)
    (output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "retime_punch_reference_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knot", action="append", required=True, help="old_time:new_time in seconds")
    args = parser.parse_args()
    retime(args.model, args.reference, args.output, [[float(x) for x in value.split(":")] for value in args.knot])


if __name__ == "__main__":
    main()
