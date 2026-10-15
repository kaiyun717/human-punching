#!/usr/bin/env python3
"""Independent native MuJoCo PD/feedforward baseline, never learned control.

Run on an allocated CPU worker. Hold the initial guard, then play the complete
reference and final hold without a state reset. Only bounded motor torque is
applied. Every physics state and force/support evidence is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from human_punching.jab_evaluation import evaluate_rollout
from human_punching.jab_learning import (
    actuator_layout,
    load_reference,
    support_balance_map,
    support_feedforward,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate(times, values, query):
    upper = min(max(int(np.searchsorted(times, query, side="right")), 1), len(times) - 1)
    lower = upper - 1
    alpha = np.clip((query - times[lower]) / (times[upper] - times[lower]), 0, 1)
    return values[lower] * (1-alpha) + values[upper] * alpha


def run(args, dt):
    destination = args.output / f"dt_{round(dt * 1000):03d}ms"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination / "scalar_native_baseline_source.py")
    helper_source = Path(support_feedforward.__code__.co_filename)
    shutil.copy2(helper_source, destination / "jab_learning_helper_source.py")
    tree = ET.parse(args.model)
    # Copy the standalone asset; the source model and reference stay immutable.
    # Match the current GPU trainer's solver settings for the native comparison.
    tree.getroot().find("option").attrib.update(
        timestep=str(dt), iterations="30", ls_iterations="15", tolerance="1e-7")
    model_path = destination / "human.xml"
    tree.write(model_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    ids, qadr, vadr = actuator_layout(model)
    reference = load_reference(model, args.reference)
    metadata = json.loads(args.body_metadata.read_text())
    if metadata["joint_names"] != [model.joint(int(j)).name for j in ids]:
        raise ValueError("Controller metadata is not in compiled actuator order")
    kp = np.asarray(metadata["kp"]) * args.gain_scale
    kd = np.asarray(metadata["kd"]) * np.sqrt(args.gain_scale)
    torque = support_feedforward(model, reference, vadr)
    balance_map = support_balance_map(model, reference, vadr) if args.balance_feedback else None
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    feet = [model.geom(f"physical/{side}Foot").id for side in ("Left", "Right")]
    soles = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    hands = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    final_time = args.quiet_guard + reference["t"][-1] + args.hold
    steps = int(np.ceil(final_time / dt))
    keys = ("time_s", "qpos", "qvel", "ctrl", "qfrc_actuator", "qfrc_applied",
            "xfrc_applied", "foot_normal_force_n", "support_contact", "sole_pos_w",
            "glove_pos_w", "qfrc_constraint", "self_contact_count",
            "balance_torque_nm", "balance_root_error", "balance_desired_acceleration")
    history = {key: [] for key in keys}
    force = np.zeros(6)
    start = time.monotonic()
    for step in range(steps + 1):
        clock = step * dt
        if abs(data.time - clock) > 1e-8:
            raise RuntimeError("Unexpected simulator time discontinuity / automatic state reset")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite state; reject this rollout")
        reference_clock = max(clock-args.quiet_guard, 0)
        desired = interpolate(reference["t"], reference["qpos"], reference_clock)
        desired[3:7] /= np.linalg.norm(desired[3:7])
        desired_v = interpolate(reference["t"], reference["qvel"], reference_clock)
        feedforward = interpolate(reference["t"], torque, reference_clock)
        if clock < args.quiet_guard:
            desired_v = np.zeros_like(desired_v)
        target = np.clip(desired[qadr], model.jnt_range[ids, 0], model.jnt_range[ids, 1])
        control = kp * (target - data.qpos[qadr]) + kd * (desired_v[vadr] - data.qvel[vadr])
        root_error = np.zeros(model.nv)
        acceleration, balance_torque = np.zeros(6), np.zeros(model.nu)
        if args.balance_feedback:
            # mj_differentiatePos gives the shortest current-to-desired root
            # rotation vector in the same local frame as free-joint qvel.
            mujoco.mj_differentiatePos(model, root_error, 1., data.qpos, desired)
            acceleration = np.clip(12 * root_error[:6] - 5 * (
                data.qvel[:6] - desired_v[:6]), -3., 3.)
            balance_torque = interpolate(reference["t"], balance_map, reference_clock) @ acceleration
        data.ctrl[:] = np.clip(control + feedforward + balance_torque,
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        # Record quantities at the same current state, before integrating the
        # recorded bounded motor command over the next interval.
        mujoco.mj_forward(model, data)
        foot_forces = np.zeros(2)
        self_count = 0
        for contact_id, contact in enumerate(data.contact):
            mujoco.mj_contactForce(model, data, contact_id, force)
            if floor not in contact.geom:
                self_count += int(force[0] > 1e-6)
                continue
            for side, foot in enumerate(feet):
                if foot in contact.geom:
                    foot_forces[side] += max(float(force[0]), 0)
        history["time_s"].append(clock)
        for key in ("qpos", "qvel", "ctrl", "qfrc_actuator", "qfrc_applied",
                    "xfrc_applied", "qfrc_constraint"):
            history[key].append(getattr(data, key).copy())
        history["foot_normal_force_n"].append(foot_forces)
        history["support_contact"].append(foot_forces > 1e-6)
        history["sole_pos_w"].append(data.site_xpos[soles].copy())
        history["glove_pos_w"].append(data.site_xpos[hands].copy())
        history["self_contact_count"].append(self_count)
        history["balance_torque_nm"].append(balance_torque)
        history["balance_root_error"].append(root_error[:6])
        history["balance_desired_acceleration"].append(acceleration)
        if step < steps:
            mujoco.mj_step(model, data)
    arrays = {key: np.asarray(value) for key, value in history.items()}
    arrays.update(state_reset_count=np.array(1), physics_step_count=np.array(steps),
                  evaluation_kind=np.array("native_zero_residual_pd_inverse_statics"),
                  head_center_w=np.asarray(reference["head_center"]),
                  model_sha256=np.array(digest(model_path)),
                  reference_sha256=np.array(digest(args.reference)),
                  quiet_guard_duration_s=np.array(args.quiet_guard))
    rollout = destination / "rollout.npz"
    np.savez_compressed(rollout, **arrays)
    aligned_reference = destination / "reference_with_quiet_guard.npz"
    np.savez_compressed(aligned_reference,
                        t=np.r_[0., reference["t"] + args.quiet_guard],
                        qpos=np.r_[reference["qpos"][:1], reference["qpos"]],
                        qvel=np.r_[np.zeros((1, model.nv)), reference["qvel"]],
                        head_center=reference["head_center"],
                        guard_hold_start=reference["guard_hold_start"]+args.quiet_guard)
    evaluation = evaluate_rollout(model_path, rollout, aligned_reference,
                                  destination / "evaluation.json")
    quiet = arrays["time_s"] <= args.quiet_guard
    quiet_report = dict(
        duration_s=args.quiet_guard,
        root_height_min_m=float(arrays["qpos"][quiet, 2].min()),
        root_position_drift_max_m=float(np.linalg.norm(
            arrays["qpos"][quiet, :3] - arrays["qpos"][0, :3], axis=1).max()),
        root_speed_max_m_s=float(np.linalg.norm(arrays["qvel"][quiet, :3], axis=1).max()),
        sole_drift_max_m=float(np.linalg.norm(arrays["sole_pos_w"][quiet] -
                                             arrays["sole_pos_w"][0], axis=2).max()),
        double_support_fraction=float(np.all(arrays["support_contact"][quiet], axis=1).mean()),
        joint_error_max_rad=float(abs(arrays["qpos"][quiet][:, qadr] -
                                      reference["qpos"][0, qadr]).max()),
        final_joint_speed_max_rad_s=float(abs(arrays["qvel"][np.flatnonzero(quiet)[-1], vadr]).max()),
    )
    (destination / "quiet_guard.json").write_text(json.dumps(quiet_report, indent=2) + "\n")
    provenance = dict(
        native_cpu_mujoco=True, learning_used=False, offline_inverse_statics=True,
        feedforward_helper="human_punching.jab_learning.support_feedforward",
        helper_source_sha256=digest(helper_source),
        balance_feedback=args.balance_feedback, gain_scale=args.gain_scale,
        balance_feedback_gains=dict(kp=12., kd=5., acceleration_clip=3.),
        balance_map_helper="human_punching.jab_learning.support_balance_map",
        mujoco_version=mujoco.__version__,
        model_source=str(args.model.resolve()), model_source_sha256=digest(args.model),
        model_runtime=str(model_path.resolve()), model_runtime_sha256=digest(model_path),
        reference_source=str(args.reference.resolve()), reference_sha256=digest(args.reference),
        controller_metadata_sha256=digest(args.body_metadata), script_sha256=digest(__file__),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), timestep_s=dt,
        initial_quiet_guard_s=args.quiet_guard, cycle_duration_s=float(reference["t"][-1]),
        final_hold_s=args.hold, state_reset_count=1, physics_step_count=steps,
        state_writes="single initial reset; no subsequent writes to qpos/qvel",
        external_forces="none; support forces used only to compute bounded joint feedforward",
        wall_seconds=time.monotonic()-start,
        limitations=["Native CPU integrator, not evidence of GPU equivalence",
                     "Fixed head target and one deterministic reset"],
    )
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return dict(timestep_s=dt, directory=str(destination.resolve()),
                pilot_pass=evaluation["pilot_pass"], failed_checks=evaluation["failed_checks"],
                quiet_guard=quiet_report, contact=evaluation["glove_head_min_distance_m"],
                minimum_root_height_m=evaluation["root_height_min_m"],
                maximum_root_tilt_rad=evaluation["root_tilt_max_rad"],
                final_guard=evaluation["final_guard"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--body-metadata", type=Path,
                        default=Path("outputs/scalar-body/scalar_model.json"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/scalar-reference-v2/jab.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.002, .005])
    parser.add_argument("--quiet-guard", type=float, default=3.)
    parser.add_argument("--hold", type=float, default=1.)
    parser.add_argument("--balance-feedback", action="store_true")
    parser.add_argument("--gain-scale", type=float, default=1.)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside a CPU Slurm allocation")
    results = [run(args, dt) for dt in args.timesteps]
    (args.output / "summary.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
