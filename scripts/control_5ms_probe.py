#!/usr/bin/env python3
"""Native CPU numerical diagnostic; not a learned-policy acceptance result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np

from human_punching.jab_learning import load_reference, support_balance_map, support_feedforward
from human_punching.scalar_control import (
    actuator_torque_ranges,
    export_implicit_pd_model,
    native_pd_drive,
    scalar_actuator_layout,
)


def interpolate(times, values, query):
    i = np.clip(np.searchsorted(times, query, side="right"), 1, len(times)-1)
    a = np.clip((query-times[i-1])/(times[i]-times[i-1]), 0, 1)
    return (1-a)*values[i-1]+a*values[i]


def run(args, mode, dt, balance):
    output = args.output / f"{mode}_{dt*1000:g}ms_balance{int(balance)}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.body_metadata.read_text())
    kp, kd = np.asarray(metadata["kp"]), np.asarray(metadata["kd"])
    model_path = output / "human.xml"
    if mode == "native":
        export_implicit_pd_model(args.model, model_path, kp, kd, timestep=dt)
    else:
        shutil.copy2(args.model, model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep, model.opt.iterations = dt, 30
    model.opt.ls_iterations, model.opt.tolerance = 15, 1e-7
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ids, qadr, vadr = scalar_actuator_layout(model)
    # Compute inverse statics with the unchanged source motor XML: helper
    # clipping intentionally reads the original motor ctrlrange.
    source_model = mujoco.MjModel.from_xml_path(str(args.model))
    ref = load_reference(source_model, args.reference, args.speed)
    ff = support_feedforward(source_model, ref, vadr)
    balance_map = support_balance_map(source_model, ref, vadr) if balance else None
    data = mujoco.MjData(model)
    data.qpos[:] = ref["qpos"][0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    limits = actuator_torque_ranges(model)
    hands = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    soles = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    history = {key: [] for key in ("t", "qpos", "qvel", "ctrl", "actuator_force",
                                  "qfrc_actuator", "qfrc_applied", "xfrc_applied",
                                  "glove_pos", "sole_pos")}
    duration = args.quiet_guard + ref["t"][-1] + args.hold
    steps = int(np.ceil(duration/dt))
    for step in range(steps+1):
        clock = step*dt
        if abs(data.time-clock) > 1e-8:
            raise RuntimeError("Physics clock reset or hidden timestep")
        ref_time = max(clock-args.quiet_guard, 0)
        target = interpolate(ref["t"], ref["qpos"], ref_time)
        velocity = interpolate(ref["t"], ref["qvel"], ref_time)
        if clock < args.quiet_guard:
            velocity[:] = 0
        feedforward = interpolate(ref["t"], ff, ref_time)
        if balance:
            error = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, error, 1., data.qpos, target)
            acceleration = np.clip(12*error[:6]-5*(data.qvel[:6]-velocity[:6]), -3., 3.)
            feedforward += interpolate(ref["t"], balance_map, ref_time)@acceleration
        qtarget = np.clip(target[qadr], model.jnt_range[ids, 0], model.jnt_range[ids, 1])
        drive = native_pd_drive(qtarget, velocity[vadr], kp, kd, feedforward)
        data.ctrl[:] = drive if mode == "native" else np.clip(
            drive-kp*data.qpos[qadr]-kd*data.qvel[vadr], limits[:, 0], limits[:, 1])
        mujoco.mj_forward(model, data)
        history["t"].append(clock)
        for key in history:
            if key not in ("t", "glove_pos", "sole_pos"):
                history[key].append(getattr(data, key).copy())
        history["glove_pos"].append(data.site_xpos[hands].copy())
        history["sole_pos"].append(data.site_xpos[soles].copy())
        if step < steps:
            mujoco.mj_step(model, data)
    arrays = {key: np.asarray(value) for key, value in history.items()}
    np.savez_compressed(output/"rollout.npz", **arrays, physics_timestep_s=dt,
                        state_reset_count=1, physics_step_count=steps)
    height, q = arrays["qpos"][:, 2], arrays["qpos"][:, 3:7]
    tiltcos = 1-2*(q[:, 1]**2+q[:, 2]**2)
    fall = (height < ref["qpos"][0, 2]*.65) | (tiltcos < .3)
    quiet = arrays["t"] <= args.quiet_guard
    attack = (arrays["t"] >= args.quiet_guard+.5/args.speed) & (
        arrays["t"] <= args.quiet_guard+3.388888888888889/args.speed)
    distance = np.linalg.norm(arrays["glove_pos"]-ref["head_center"], axis=-1)
    result = dict(controller=mode, timestep_s=dt, balance_feedback=balance,
                  source_speed=args.speed, quiet_guard_s=args.quiet_guard,
                  first_fall_s=float(arrays["t"][fall][0]) if fall.any() else None,
                  quiet_max_joint_velocity_rad_s=float(abs(arrays["qvel"][quiet][:, vadr]).max()),
                  quiet_max_root_drift_m=float(np.linalg.norm(
                      arrays["qpos"][quiet, :3]-arrays["qpos"][0, :3], axis=-1).max()),
                  quiet_max_sole_drift_m=float(np.linalg.norm(
                      arrays["sole_pos"][quiet]-arrays["sole_pos"][0], axis=-1).max()),
                  peak_joint_velocity_rad_s=float(abs(arrays["qvel"][:, vadr]).max()),
                  attack_minimum_glove_head_m=float(distance[attack].min()),
                  all_cycle_minimum_glove_head_m=float(distance.min()),
                  final_root_height_m=float(height[-1]),
                  final_joint_tracking_max_rad=float(abs(arrays["qpos"][-1, qadr]-
                                                        ref["qpos"][-1, qadr]).max()),
                  maximum_actual_torque_nm=float(abs(arrays["qfrc_actuator"][:, vadr]).max()),
                  maximum_torque_violation_nm=float(np.maximum(
                      np.maximum(limits[:, 0]-arrays["actuator_force"],
                                 arrays["actuator_force"]-limits[:, 1]), 0).max()),
                  external_force_max=float(max(abs(arrays["xfrc_applied"]).max(),
                                               abs(arrays["qfrc_applied"]).max())),
                  root_actuation_max=float(abs(arrays["qfrc_actuator"][:, :6]).max()),
                  integrator="implicitfast", physics_step_count=steps,
                  state_reset_count=1, native_cpu_only=True, learned=False,
                  model_source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--body-metadata", type=Path,
                        default=Path("outputs/scalar-body/scalar_model.json"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/scalar-reference-v3/jab.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet-guard", type=float, default=3.)
    parser.add_argument("--hold", type=float, default=1.)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.002, .005])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this batch diagnostic in a Slurm CPU allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/"probe_source.py")
    results = [run(args, mode, dt, balance) for dt in args.timesteps
               for mode in ("explicit", "native") for balance in (False, True)]
    (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
