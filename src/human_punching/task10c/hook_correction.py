"""One bounded hook recovery correction on the frozen Phase-1 references.

This is an explicit hand bypass with independent, bounded arm pose IK. It is
not a trajectory optimizer, controller, or policy. The outward punch, clock,
model and collision geometry stay fixed. Only RH needs the bypass; LH is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .guard_correction import (
    _blend, _name, fit_guard, pose_diagnostics, source_terminal_torso_seed,
    training_model,
)
from .references import StoredReference, arrays_from_qpos
from .robot import JOINT_NAMES, ROOT

OUTPUT_ROOT = ROOT / "outputs/task10c/reference_correction/hooks_final"


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x ** 3 * (10. - 15. * x + 6. * x ** 2)


def recovery_weight(time_s, start, end):
    """Smooth rise, brief plateau, and return to the exact endpoint."""
    return float(smoothstep((time_s - start) / (.90 - start))
                 * (1. - smoothstep((time_s - 1.20) / (end - 1.20))))


def recovery_endpoint(reference, model):
    """Select the recorded nearby RH return pose, keeping the original clock."""
    index = int(np.argmin(np.abs(reference.time_s - 1.32)))
    seed_reference = StoredReference(
        {"qpos": reference.arrays["qpos"][[index]],
         "source_frames": reference.arrays["source_frames"][[index]]},
        reference.metadata,
    )
    seed, source_details = source_terminal_torso_seed(seed_reference, model)
    # The selected return seed still overbends the active wrist. Relax its two
    # bending coordinates halfway toward neutral, preserving forearm pronation
    # (wrist roll), then run the same bounded physical guard fit once.
    wrist_names = ("right_wrist_pitch_joint", "right_wrist_yaw_joint")
    wrist_qa = [model.jnt_qposadr[_name(model, name, "joint")] for name in wrist_names]
    wrist_before = seed[wrist_qa].copy()
    seed[wrist_qa] *= .5
    endpoint, fit_details = fit_guard(model, seed)
    return endpoint, {
        "selected_input_index": index,
        "selected_input_time_s": float(reference.time_s[index]),
        "selected_source_frame": float(reference.arrays["source_frames"][index]),
        "selection_reason": "Nearby family-specific return pose precedes the original final high-cover/slip pose; no universal torso-tilt constraint.",
        "source_torso": source_details,
        "active_wrist_relaxation": {
            "method": "One bounded endpoint seed adjustment: halve wrist pitch/yaw, preserve wrist roll, then existing guard fit",
            "joint_names": list(wrist_names), "before_rad": wrist_before.tolist(),
            "seed_rad": seed[wrist_qa].tolist(), "after_rad": endpoint[wrist_qa].tolist(),
            "maximum_seed_adjustment_rad": float(np.abs(wrist_before - seed[wrist_qa]).max()),
            "source_claim": "Intentional reasonable-guard adjustment; not exact recorded wrist tracking",
        },
        "guard_fit": fit_details,
        "pose_diagnostics": pose_diagnostics(model, endpoint),
    }


def arm_bypass(model, time_s, base_qpos, strike_s):
    """Fit one smooth, small forward/lateral bypass; all other joints fixed."""
    data = mujoco.MjData(model)
    names = [name for name in JOINT_NAMES
             if any(part in name for part in ("shoulder_", "elbow_", "wrist_"))]
    joints = [_name(model, name, "joint") for name in names]
    addresses = model.jnt_qposadr[joints]
    sites = [_name(model, side + "_glove_center", "site") for side in ("left", "right")]
    elbows = [_name(model, side + "_elbow_link", "body") for side in ("left", "right")]
    torso = _name(model, "torso_link", "body")
    # The right glove passes forward of the inactive hand during withdrawal;
    # the inactive elbow also gets a small forward clearance from the torso.
    hand_offsets = np.array([[.04, .03, -.01], [.09, -.025, .01]])
    elbow_offsets = np.array([[.035, .015, 0.], [.03, -.015, 0.]])
    pairs = [("left_glove_collision", "right_glove_collision"),
             ("left_wrist_collision", "right_wrist_collision")]
    for side, other in (("left", "right"), ("right", "left")):
        pairs.extend((side + "_glove_collision", name) for name in
                     ("head_collision", "torso_collision", other + "_wrist_collision",
                      other + "_elbow_collision", other + "_shoulder_yaw_collision"))
        pairs.extend((side + part, "torso_collision") for part in
                     ("_shoulder_yaw_collision", "_elbow_collision", "_wrist_collision"))
    pairs = [(_name(model, a, "geom"), _name(model, b, "geom")) for a, b in pairs]
    # Preserve both interpolation brackets of strike and the following sample:
    # central-difference strike velocity is unchanged too.
    strike_hi = int(np.searchsorted(time_s, strike_s, side="left"))
    start = float(time_s[min(strike_hi + 1, len(time_s) - 1)])
    end = float(time_s[-1] - .04)
    output = base_qpos.copy()
    previous_delta = np.zeros(len(addresses))
    evaluations, position_changes, joint_changes = [], [], []
    for index, t in enumerate(time_s):
        if t <= start or t >= end:
            previous_delta[:] = 0.
            continue
        base = base_qpos[index]
        initial = base[addresses].copy()
        data.qpos[:] = base
        mujoco.mj_kinematics(model, data)
        original_hands = data.site_xpos[sites].copy()
        rotation = data.xmat[torso].reshape(3, 3).copy()
        weight = recovery_weight(float(t), start, end)
        target_hands = original_hands + weight * hand_offsets @ rotation.T
        target_elbows = data.xpos[elbows].copy() + weight * elbow_offsets @ rotation.T
        target_rotations = [Rotation.from_matrix(data.site_xmat[s].reshape(3, 3)) for s in sites]
        lower = np.maximum(initial - .50, model.jnt_range[joints, 0] + .001)
        upper = np.minimum(initial + .50, model.jnt_range[joints, 1] - .001)

        def residual(values):
            pose = base.copy()
            pose[addresses] = values
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            parts = [(data.site_xpos[sites] - target_hands).ravel() / .025,
                     (data.xpos[elbows] - target_elbows).ravel() / .06,
                     .25 * (values - initial) / .50,
                     .15 * (values - initial - previous_delta) / .15]
            for site, desired in zip(sites, target_rotations):
                actual = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
                parts.append((desired.inv() * actual).as_rotvec() / .8)
            distance = np.array([mujoco.mj_geomDistance(model, data, a, b, .15, None) for a, b in pairs])
            parts.append(np.maximum(.010 - distance, 0.) / .004)
            return np.concatenate(parts)

        fit = least_squares(residual, np.clip(initial + previous_delta, lower + 1e-8, upper - 1e-8),
                            bounds=(lower, upper), max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        output[index, addresses] = fit.x
        previous_delta = fit.x - initial
        data.qpos[:] = output[index]
        mujoco.mj_kinematics(model, data)
        position_changes.append(np.linalg.norm(data.site_xpos[sites] - original_hands, axis=-1))
        joint_changes.append(float(np.max(np.abs(previous_delta))))
        evaluations.append(int(fit.nfev))
    return output, {
        "method": "One explicit smooth torso-frame hand bypass with bounded independent arm pose IK and selected physical clearance residuals",
        "start_time_s": start, "end_time_s": end,
        "plateau_interval_s": [.90, 1.20],
        "hand_target_offsets_torso_m_left_right": hand_offsets.tolist(),
        "elbow_target_offsets_torso_m_left_right": elbow_offsets.tolist(),
        "arm_joint_names": names, "per_joint_adjustment_bound_rad": .50,
        "maximum_observed_joint_adjustment_rad": max(joint_changes, default=0.),
        "maximum_glove_displacement_from_endpoint_blended_input_m": np.max(position_changes, axis=0).tolist(),
        "maximum_pose_fit_evaluations": max(evaluations, default=0),
        "clearance_shaping_margin_m": .010,
        "fixed_coordinates": "pelvis position/orientation, legs, waist and head; only named arm joints move",
        "strike_preservation": "both interpolation brackets and their central-difference neighbor remain unchanged",
        "limitation": "Pose IK with FK clearance shaping; no dynamic tracking or balance guarantee",
    }


def correct_hook(input_npz, output_npz):
    input_npz, output_npz = Path(input_npz).resolve(), Path(output_npz).resolve()
    output_npz.relative_to(OUTPUT_ROOT.resolve())
    reference = StoredReference.load(input_npz)
    family = reference.metadata["family"]
    if family not in ("left_hook", "right_hook"):
        raise ValueError("This focused correction accepts hooks only")
    parent_hash = hashlib.sha256(input_npz.read_bytes()).hexdigest()
    details = {"revision": "hook_recovery_bypass_v1", "input_npz": str(input_npz),
               "input_sha256": parent_hash, "family": family,
               "clock": "Unchanged time_s, launch at0, nominal strike and all postlaunch wind-up; no retiming"}
    if family == "left_hook":
        arrays = reference.arrays
        details["changes"] = "None; existing family-specific LH guard and complete trajectory retained"
    else:
        model = training_model()
        endpoint, endpoint_details = recovery_endpoint(reference, model)
        qpos = reference.arrays["qpos"].copy()
        end = float(reference.time_s[-1])
        for i, t in enumerate(reference.time_s):
            if t > end - .30:
                weight = smoothstep((t - end + .30) / (.30 - .04))
                qpos[i] = _blend(qpos[i], endpoint, weight)
        qpos, bypass_details = arm_bypass(model, reference.time_s, qpos, reference.metadata["strike_time_s"])
        arrays = arrays_from_qpos(reference.time_s, qpos)
        arrays["source_frames"] = reference.arrays["source_frames"].copy()
        details["endpoint"] = endpoint_details
        details["endpoint_blend_interval_s"] = [end - .30, end]
        details["bypass"] = bypass_details
        details["source_frames_role"] = "Original conversion lineage only for modified recovery samples; final held guard comes from selected_source_frame in endpoint metadata, not the inherited final source frame. Clock is unchanged."
    metadata = {**reference.metadata, "hook_final_correction": details}
    if family == "right_hook":
        metadata["terminal_guard_interval_s"] = [float(reference.time_s[-1] - .04), float(reference.time_s[-1])]
        metadata["terminal_guard_source_frame"] = details["endpoint"]["selected_source_frame"]
        metadata["annotation_rationale"] = "Original outward annotation and launch clock retained. RH withdrawal hand bypass and nearby recorded return guard are explicit corrections; terminal source identity is terminal_guard_source_frame, not source_frames[-1]. Prior guard_correction records the parent operation only and is superseded for the RH terminal guard by hook_final_correction."
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_npz.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(correct_hook(args.input, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
