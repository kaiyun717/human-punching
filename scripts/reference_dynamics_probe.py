#!/usr/bin/env python3
"""Idealized reference inverse-dynamics/support diagnostic, not feasibility proof.

For each sampled pose, allocate nonnegative forces at eight sole corners to
match the desired free-root wrench while minimizing the largest fraction of
the unchanged joint torque limits. Four planar friction inequalities form an
inscribed diamond (|fx|+|fy| <= mu*fz). No force is applied to a simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from human_punching.scalar_control import actuator_torque_ranges, scalar_actuator_layout


def allocate_support(required, jacobian, vadr, limits, friction):
    """LP variable is [eight xyz ground forces, peak normalized joint torque]."""
    contacts = jacobian.shape[0] // 3
    n = 3*contacts+1
    root = jacobian[:, :6].T
    joint = jacobian[:, vadr].T
    upper = np.c_[-joint, -limits[:, 1]]
    lower = np.c_[joint, limits[:, 0]]
    friction_rows = []
    for contact in range(contacts):
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            row = np.zeros(n)
            row[3*contact:3*contact+3] = [sx, sy, -friction]
            friction_rows.append(row)
    a = np.r_[upper, lower, friction_rows]
    b = np.r_[-required[vadr], required[vadr], np.zeros(4*contacts)]
    bounds = [(None, None), (None, None), (0, None)]*contacts+[(0, None)]
    objective = np.zeros(n)
    objective[-1] = 1
    result = linprog(objective, A_ub=a, b_ub=b,
                     A_eq=np.c_[root, np.zeros(6)], b_eq=required[:6],
                     bounds=bounds, method="highs")
    if not result.success:
        return dict(success=False, status=int(result.status), message=result.message)
    forces, alpha = result.x[:-1].reshape(contacts, 3), float(result.x[-1])
    tau = required[vadr]-joint@forces.ravel()
    root_residual = root@forces.ravel()-required[:6]
    return dict(success=True, peak_normalized_torque=alpha,
                torque=tau.tolist(), forces=forces.tolist(),
                root_wrench_residual_max=float(abs(root_residual).max()))


def analyze(model, reference_path, speed, stride, friction):
    reference = np.load(reference_path, allow_pickle=False)
    times, poses, velocity = reference["t"], reference["qpos"], reference["qvel"]
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    _, _, vadr = scalar_actuator_layout(model)
    limits = actuator_torque_ranges(model)
    sites = [model.site(f"{side}Foot_sole_{corner}").id
             for side in ("Left", "Right") for corner in range(4)]
    root_body = model.body("Hips").id
    data = mujoco.MjData(model)
    mass = np.zeros((model.nv, model.nv))
    jac = np.zeros((3*len(sites), model.nv))
    results = []
    attack = reference["attack_window_reference_s"]
    sample_indices = np.unique(np.r_[np.arange(0, len(times), stride), len(times)-1])
    for frame in sample_indices:
        data.qpos[:], data.qvel[:] = poses[frame], velocity[frame]*speed
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, mass)
        required = mass@(acceleration[frame]*speed**2)+data.qfrc_bias
        for i, site in enumerate(sites):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, site)
        solved = allocate_support(required, jac, vadr, limits, friction)
        # Upper-body joints have zero sole Jacobian columns: ground-force
        # allocation cannot reduce their exact-reference torque demand. Report
        # this lower bound even when the root-wrench allocation is infeasible.
        independent = np.linalg.norm(jac[:, vadr], axis=0) < 1e-12
        fraction = np.maximum(required[vadr]/limits[:, 1], required[vadr]/limits[:, 0])
        independent_fraction = np.where(independent, fraction, 0.)
        worst_independent = int(independent_fraction.argmax())
        hull = ConvexHull(data.site_xpos[sites, :2])
        com = data.subtree_com[root_body].copy()
        margins = -(hull.equations[:, :2]@com[:2]+hull.equations[:, 2])
        margin = float(margins.min())
        if solved["success"]:
            tau = np.asarray(solved["torque"])
            normalized = abs(tau)/np.maximum(abs(limits[:, 0]), abs(limits[:, 1]))
            i = int(normalized.argmax())
            solved["most_loaded_joint"] = model.actuator(i).name
        results.append(dict(frame=int(frame), reference_time_s=float(times[frame]),
                            source_motion_window=bool(float(reference["launch_time"]) <= times[frame]
                                                      <= float(reference["return_start"])),
                            attack_window=bool(attack[0] <= times[frame] <= attack[1]),
                            com_support_margin_m=margin, com_w=com.tolist(),
                            ground_independent_peak_strength_fraction=float(
                                independent_fraction[worst_independent]),
                            ground_independent_peak_joint=model.actuator(worst_independent).name,
                            ground_independent_peak_torque_nm=float(required[vadr[worst_independent]]),
                            desired_root_generalized_wrench=required[:6].tolist(), **solved))
    groups = {}
    for name, rows in (("all", results), ("attack", [r for r in results if r["attack_window"]]),
                       ("source_motion", [r for r in results if r["source_motion_window"]])):
        solved = [row for row in rows if row["success"]]
        utilization = [row["peak_normalized_torque"] for row in solved]
        groups[name] = dict(samples=len(rows), support_wrench_infeasible=sum(not r["success"] for r in rows),
                            minimum_com_support_margin_m=min(r["com_support_margin_m"] for r in rows),
                            ground_independent_peak_strength_fraction=max(
                                r["ground_independent_peak_strength_fraction"] for r in rows),
                            ground_independent_strength_violations=sum(
                                r["ground_independent_peak_strength_fraction"] > 1+1e-6 for r in rows),
                            peak_minimum_torque_fraction=max(utilization) if utilization else None,
                            p95_minimum_torque_fraction=float(np.percentile(utilization, 95))
                            if utilization else None,
                            feasible_with_original_torque_limits=sum(u <= 1+1e-6 for u in utilization))
    return dict(reference=str(reference_path),
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                reference_clock_multiplier=speed, stride=stride,
                friction_mu=friction, summaries=groups, frames=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--bank", type=Path, default=Path("outputs/scalar-four-reference-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--speeds", type=float, nargs="+", default=[1., .6])
    parser.add_argument("--families", nargs="+", default=["jab", "cross", "left_hook", "right_hook"],
                        choices=("jab", "cross", "left_hook", "right_hook"))
    parser.add_argument("--friction", type=float, default=.9)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this diagnostic inside a Slurm CPU allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"probe_source.py").write_text(Path(__file__).read_text())
    model = mujoco.MjModel.from_xml_path(str(args.model))
    summaries = []
    for family in args.families:
        reference = args.bank/family/f"{family}.npz"
        for speed in args.speeds:
            result = analyze(model, reference, speed, args.stride, args.friction)
            (args.output/f"{family}_{speed:g}.json").write_text(json.dumps(result, indent=2)+"\n")
            summary = dict(family=family, speed=speed, **result["summaries"])
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    (args.output/"limitations.json").write_text(json.dumps(dict(
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        interpretation="Idealized instantaneous demand for exactly tracking the reference; not a proof of dynamic feasibility or impossibility.",
        assumptions=["All eight sole corners may supply nonnegative ground normal force.",
                     "Inscribed diamond friction approximation; no contact compliance or force-rate bound.",
                     "Reference qacc is finite-differenced from saved qvel.",
                     "Passive joint-stop and self-contact reaction forces are omitted.",
                     "Contact acceleration consistency and tracking feedback are not enforced.",
                     "Clock multiplier scales all velocities and accelerations, including inserted return; attack summaries isolate the punch.",
                     "COM support margin is a static indicator, not a dynamic balance certificate.",
                     "A policy may deviate from the reference to achieve feasible behavior."],
    ), indent=2)+"\n")


if __name__ == "__main__":
    main()
