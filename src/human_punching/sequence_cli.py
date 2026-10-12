"""Continuous classical two-jab physics proof with measured guard admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from .contact import evaluate_clearance, first_contact_time
from .cycle_bank import GuardTolerances
from .physical import bounded_pd, controller_arrays, interpolate_pose

PROJECT = Path(__file__).resolve().parents[2]


def guard_errors(model, qpos, qvel, target):
    """Physical pilot has one free root followed by 20 ball joints."""
    if model.nv != 66 or model.nq != 87:
        raise ValueError("Sequence proof expects the finalized 87/66 human model")
    delta = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1., qpos, target)
    return dict(root_position_m=float(np.linalg.norm(delta[:3])),
                root_orientation_rad=float(np.linalg.norm(delta[3:6])),
                joint_position_m=0., joint_angle_rad=float(np.linalg.norm(delta[6:].reshape(-1, 3), axis=1).max()),
                root_linear_velocity_mps=float(np.linalg.norm(qvel[:3])),
                root_angular_velocity_radps=float(np.linalg.norm(qvel[3:6])),
                joint_linear_velocity_mps=0., joint_angular_velocity_radps=float(np.linalg.norm(qvel[6:].reshape(-1, 3), axis=1).max()))


def sequence_rollout(model, times, reference, head_center, *, cycles=2, speed=.6,
                     gain_scale=1.5, endpoint_hold_s=.4, bridge_s=1.5,
                     guard_timeout_s=3., guard_dwell_s=.1):
    if cycles < 1 or not np.isfinite([speed, gain_scale, endpoint_hold_s, bridge_s,
                                    guard_timeout_s, guard_dwell_s]).all():
        raise ValueError("Invalid sequence settings")
    if min(speed, gain_scale, bridge_s, guard_timeout_s, guard_dwell_s) <= 0 or endpoint_hold_s < 0:
        raise ValueError("Sequence durations, gain and speed must be positive")
    data = mujoco.MjData(model)
    # The only state initialization. Every subsequent state comes from mj_step.
    data.qpos[:] = reference[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    if abs(dt - .002) > 1e-12:
        raise ValueError("The finalized sequence proof uses 2 ms physics steps")
    floor = model.geom("floor").id
    feet = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    foot_geoms = [model.geom(f"physical/{side}Foot").id for side in ("Left", "Right")]
    gloves = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    hips = model.body("Hips").id
    head_id = model.geom("g1/head_collision").id
    if not np.allclose(data.geom_xpos[head_id], head_center, atol=1e-9, rtol=0):
        raise ValueError("Fixed scene head differs from the committed reference target")
    initial_feet = data.site_xpos[feet].copy()
    initial_height = float(data.qpos[2])
    limits = controller_arrays(model)[2]
    tolerances = asdict(GuardTolerances())
    bridge_delta = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, bridge_delta, 1., reference[-1], reference[0])
    punch_duration = float((times[-1] - times[0]) / speed)
    phase, phase_start, cycle = "punch", 0., 1
    launch_times = [0.]
    completed_punches = 0
    guard_dwell_start = None
    transitions = [dict(time_s=0., phase="punch", cycle=1)]
    guard_snapshots = []
    launches = [dict(cycle=1, time_s=0., initial_reset=True, head_center_w=list(head_center))]
    all_times, all_gloves, all_phases, all_cycles = [], [], [], []
    qpos, qvel, saved_times, desired, sole_positions, contact_counts = [], [], [], [], [], []
    saturation, effort_fraction, root_wrench, root_external = [], [], [], []
    up_values, heights, penetrations, foot_drift, sole_heights = [], [], [], [], []
    outcome = "administrative_timeout"
    maximum_duration = cycles * (punch_duration + endpoint_hold_s + bridge_s + guard_timeout_s + dt) + .1
    for step in range(int(np.ceil(maximum_duration / dt)) + 1):
        mujoco.mj_forward(model, data)
        now = float(data.time)
        phase_elapsed = now - phase_start
        current_gloves = data.site_xpos[gloves].copy()
        labels = evaluate_clearance(current_gloves, head_center)
        feet_now = data.site_xpos[feet].copy()
        contacts = [sum({int(c.geom1), int(c.geom2)} == {floor, geom} for c in data.contact)
                    for geom in foot_geoms]
        up = float(data.xmat[hips].reshape(3, 3)[2, 1])
        penetration = -min([float(c.dist) for c in data.contact] + [0.])
        drift = float(np.linalg.norm(feet_now[:, :2] - initial_feet[:, :2], axis=1).max())
        support_ready = (all(contacts) and data.qpos[2] > .70 * initial_height and up > .70
                         and penetration <= .01 and drift <= .03 and feet_now[:, 2].max() <= .03)
        residual = guard_errors(model, data.qpos, data.qvel, reference[0])
        state_ready = all(residual[name] <= value for name, value in tolerances.items())
        guard_ready = bool(state_ready and support_ready and labels.fist_clearance_safe)
        finish = False
        if phase == "punch" and phase_elapsed >= punch_duration:
            completed_punches += 1
            phase, phase_start = "endpoint_hold", now
        elif phase == "endpoint_hold" and phase_elapsed >= endpoint_hold_s:
            phase, phase_start = "return_bridge", now
        elif phase == "return_bridge" and phase_elapsed >= bridge_s:
            phase, phase_start = "guard_wait", now
            guard_dwell_start = None
        elif phase == "guard_wait":
            guard_dwell_start = now if guard_ready and guard_dwell_start is None else guard_dwell_start
            if not guard_ready:
                guard_dwell_start = None
            if guard_dwell_start is not None and now - guard_dwell_start >= guard_dwell_s:
                snapshot = dict(cycle=cycle, time_s=now, residuals=residual, tolerances=tolerances,
                                foot_contacts=contacts, support_ready=bool(support_ready),
                                glove_clearance_safe=bool(labels.fist_clearance_safe),
                                guard_dwell_s=now-guard_dwell_start,
                                actual_qpos=data.qpos.tolist(), actual_qvel=data.qvel.tolist())
                guard_snapshots.append(snapshot)
                if cycle >= cycles:
                    outcome, finish = "completed_with_guard_recovery", True
                else:
                    cycle += 1
                    launch_times.append(now)
                    launches.append(dict(cycle=cycle, time_s=now, initial_reset=False,
                                         head_center_w=list(head_center), guard_snapshot=snapshot))
                    phase, phase_start = "punch", now
                    guard_dwell_start = None
            elif phase_elapsed >= guard_timeout_s:
                outcome, finish = "guard_timeout_no_next_launch", True
                guard_snapshots.append(dict(cycle=cycle, time_s=now, residuals=residual,
                    tolerances=tolerances, support_ready=bool(support_ready),
                    glove_clearance_safe=bool(labels.fist_clearance_safe), rejected=True))
        if phase != transitions[-1]["phase"] or cycle != transitions[-1]["cycle"]:
            transitions.append(dict(time_s=now, phase=phase, cycle=cycle))
        phase_elapsed = now - phase_start
        if phase == "punch":
            target, velocity = interpolate_pose(model, times, reference, times[0] + phase_elapsed * speed)
            velocity *= speed
        elif phase == "endpoint_hold":
            target, velocity = reference[-1], np.zeros(model.nv)
        elif phase == "return_bridge":
            fraction = float(np.clip(phase_elapsed / bridge_s, 0, 1))
            blend = fraction**3 * (10 - 15*fraction + 6*fraction**2)
            derivative = 30*fraction**2*(1-fraction)**2 / bridge_s
            target = reference[-1].copy()
            mujoco.mj_integratePos(model, target, bridge_delta, blend)
            velocity = bridge_delta * derivative
        else:
            target, velocity = reference[0], np.zeros(model.nv)
        clipped, _ = bounded_pd(model, data, target, velocity, gain_scale=gain_scale, balance=True)
        saturation.append(clipped)
        effort_fraction.append(float(np.max(np.linalg.norm(data.ctrl.reshape(-1, 3), axis=1) / limits)))
        root_wrench.append(float(np.max(np.abs(data.qfrc_actuator[:6]))))
        root_external.append(float(max(np.max(np.abs(data.qfrc_applied[:6])), np.max(np.abs(data.xfrc_applied)))))
        all_times.append(now)
        all_gloves.append(current_gloves)
        all_phases.append(phase)
        all_cycles.append(cycle)
        up_values.append(up)
        heights.append(float(data.qpos[2]))
        penetrations.append(penetration)
        foot_drift.append(drift)
        sole_heights.append(float(feet_now[:, 2].max()))
        contact_counts.append(contacts)
        if step % 4 == 0 or finish:
            saved_times.append(now)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            desired.append(target.copy())
            sole_positions.append(feet_now)
        if finish:
            break
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 1000:
            outcome = "diverged"
            break
    sub_times, sub_gloves = np.asarray(all_times), np.asarray(all_gloves)
    labels = evaluate_clearance(sub_gloves, head_center)
    per_cycle = []
    phases, cycle_ids = np.asarray(all_phases), np.asarray(all_cycles)
    for number, launch in enumerate(launch_times, 1):
        mask = (cycle_ids == number) & (phases == "punch")
        first = first_contact_time(sub_times[mask], sub_gloves[mask], head_center)
        distances = labels.distances_m[mask]
        per_cycle.append(dict(cycle=number, launch_time_s=launch,
                              any_proxy_contact=bool(labels.any_contact[mask].any()),
                              first_contact_time_s=float(first) if np.isfinite(first) else None,
                              measured_ttc_s=float(first-launch) if np.isfinite(first) else None,
                              minimum_center_distance_each_glove_m=distances.min(axis=0).tolist()))
    contacts = np.asarray(contact_counts)
    upright = min(heights) > .70*initial_height and min(up_values) > .70
    support_gate = bool(upright and max(foot_drift) <= .03 and max(sole_heights) <= .03
                        and max(penetrations) <= .01 and np.mean(np.all(contacts > 0, axis=1)) >= .95)
    metrics = dict(schema="human_punching_physical_sequence_v1", outcome=outcome,
        requested_cycles=cycles, launched_cycles=len(launches), completed_punch_cycles=completed_punches,
        physical_stepping=True, timestep_s=dt, duration_s=float(sub_times[-1]),
        controller="bounded quaternion stable-PD plus contact-aware balance feedforward",
        gain_scale=gain_scale, playback_speed=speed, root_state_overwrites_after_reset=0,
        root_actuator_force_max=max(root_wrench), external_root_assistance=False,
        externally_applied_force_max=max(root_external), fixed_head_center_w=list(head_center),
        head_commitment="Same stationary G1 snapshot; no homing or target recomputation",
        impact_response_g1=False, event_sampling_s=dt,
        endpoint_hold_s=endpoint_hold_s, return_bridge_s=bridge_s,
        guard_timeout_s=guard_timeout_s, guard_dwell_required_s=guard_dwell_s,
        launch_records=launches, guard_snapshots=guard_snapshots, transitions=transitions,
        per_cycle=per_cycle, support_gate=support_gate, remained_upright=bool(upright),
        min_pelvis_height_m=min(heights), root_up_min=min(up_values),
        max_sole_horizontal_drift_m=max(foot_drift), max_sole_height_m=max(sole_heights),
        floor_penetration_max_m=max(penetrations), both_feet_contact_fraction=float(np.mean(np.all(contacts > 0, axis=1))),
        mean_joint_saturation_fraction=float(np.mean(saturation)), max_torque_norm_fraction=max(effort_fraction),
        minimum_center_distance_each_glove_m=labels.distances_m.min(axis=0).tolist(),
        contact_frames_by_phase={phase: int(labels.any_contact[phases == phase].sum()) for phase in np.unique(phases)})
    metrics["continuous_sequence_gate"] = bool(outcome == "completed_with_guard_recovery" and
        len(launches) == cycles and support_gate and np.mean(saturation) <= .05 and
        all(item["any_proxy_contact"] for item in per_cycle) and max(root_wrench) == 0 and max(root_external) == 0)
    trajectory = dict(time_s=np.asarray(saved_times), qpos=np.asarray(qpos), qvel=np.asarray(qvel),
        desired_qpos=np.asarray(desired), foot_centers_w=np.asarray(sole_positions),
        substep_time_s=sub_times, substep_glove_centers_w=sub_gloves,
        substep_phase=phases, substep_cycle=cycle_ids, substep_foot_contact_counts=contacts)
    return trajectory, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "outputs/physical-5890912")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/physical-sequence")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("Outputs must stay within human-punching")
    source = args.input.resolve()
    model_path = source / "scene/physical_scene.xml"
    reference_path = source / "jab/reference.npz"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(reference_path, allow_pickle=False) as arrays:
        times, reference = arrays["time_s"], arrays["qpos"]
    head = np.asarray(json.loads((source / "jab/jab_metrics.json").read_text())["head_center_w"])
    output.mkdir(parents=True, exist_ok=True)
    trajectory, metrics = sequence_rollout(model, times, reference, head, cycles=args.cycles)
    metrics["source_scene_xml"] = str(model_path)
    metrics["source_reference"] = str(reference_path)
    metrics["source_reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    np.savez_compressed(output / "two_jabs_rollout.npz", **trajectory)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if args.render:
        from .physics_cli import render
        render(model, trajectory, metrics, output, "two_jabs")


if __name__ == "__main__":
    main()
