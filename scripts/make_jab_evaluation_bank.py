#!/usr/bin/env python3
"""Create reproducible paired guard disturbances for one scalar-body reference."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from human_punching.jab_evaluation import _reference_metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bank(model_path, reference_path, output, *, count=32, seed=20261012,
              joint_noise_std=0.005, velocity_noise_std=0.02):
    if count < 2 or joint_noise_std < 0 or velocity_noise_std < 0:
        raise ValueError("count must be at least two and noise scales nonnegative")
    model_path, reference_path, output = map(Path, (model_path, reference_path, output))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(reference_path, allow_pickle=False) as reference:
        guard = np.asarray(reference["qpos"][0], dtype=float).copy()
        reference_metadata = _reference_metadata(reference, reference_path)
    if guard.shape != (model.nq,) or not np.isfinite(guard).all():
        raise ValueError("Reference first pose does not match the body")
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    root = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(root) != 1 or len(joints) + 1 != model.njnt:
        raise ValueError("Expected one free root and scalar hinges")
    addresses = model.jnt_qposadr[joints]
    limited = model.jnt_limited[joints].astype(bool)
    qpos = np.repeat(guard[None, :], count, axis=0)
    qvel = np.zeros((count, model.nv), dtype=float)
    rng = np.random.default_rng(seed)
    for trial in range(1, count):
        # Per-trial interleaving makes the first N disturbances independent of
        # how many additional trials are generated or how the GPU batches them.
        qpos[trial, addresses] += rng.normal(0, joint_noise_std, len(joints))
        qpos[trial, addresses[limited]] = np.clip(qpos[trial, addresses[limited]],
                                                model.jnt_range[joints[limited], 0],
                                                model.jnt_range[joints[limited], 1])
        qvel[trial] = rng.normal(0, velocity_noise_std, model.nv)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "guard_trials.npz"
    model_hash, reference_hash = digest(model_path), digest(reference_path)
    np.savez_compressed(path, qpos=qpos, qvel=qvel, trial_ids=np.arange(count, dtype=np.int64),
                        seed=np.int64(seed), model_sha256=model_hash,
                        reference_sha256=reference_hash,
                        joint_noise_std_rad=joint_noise_std,
                        generalized_velocity_noise_std=velocity_noise_std)
    metadata = {
        "bank": str(path), "bank_sha256": digest(path), "trials": count, "seed": seed,
        "model": str(model_path), "model_sha256": model_hash,
        "reference": str(reference_path), "reference_sha256": reference_hash,
        "generator": "numpy.random.default_rng / PCG64",
        "random_draw_order": "For each trial 1..N-1: hinge position Gaussian vector, then full qvel Gaussian vector",
        "trial_zero": "Exact first reference guard pose and zero generalized velocity",
        "joint_position_noise_std_rad": joint_noise_std,
        "joint_position_noise_clipped_to_compiled_limits": True,
        "root_position_and_orientation_perturbed": False,
        "root_linear_velocity_noise_std_m_s": velocity_noise_std,
        "root_angular_and_joint_velocity_noise_std_rad_s": velocity_noise_std,
        "joint_names": [model.joint(int(joint)).name for joint in joints],
        "punch_family": reference_metadata.get("punch_family", reference_metadata.get("family", "unspecified")),
        "active_hand": reference_metadata.get("active_hand", "unspecified"),
        "source_speed": reference_metadata.get("source_speed", reference_metadata.get("source_speed_scale")),
        "reference_timing": {key: reference_metadata[key] for key in ("timing_kind", "timing_description", "source_speed_kind", "local_source_speed_min", "local_source_speed_max", "retiming_duration_ratio", "nominal_source_speed", "source_playback_duration_ratio") if key in reference_metadata},
        "scope": "Paired small guard perturbations on the same punch reference; not held-out motion generalization",
        "usage": "Use these exact trial_ids and initial arrays for both classical baseline and learned controller; save every trial at every physics substep",
        "evaluation_protocol": "Keep these exact initial states out of gradient-update rollouts. If used for checkpoint selection, call this a validation bank, not an untouched final test set.",
    }
    if "attack_window_reference_s" in reference_metadata:
        metadata["attack_window_reference_s"] = reference_metadata["attack_window_reference_s"]
    elif "launch_time_s" in reference_metadata and "return_start_s" in reference_metadata:
        metadata["attack_window_reference_s"] = [reference_metadata["launch_time_s"], reference_metadata["return_start_s"]]
    elif "launch_time" in reference_metadata and "return_start" in reference_metadata:
        metadata["attack_window_reference_s"] = [reference_metadata["launch_time"], reference_metadata["return_start"]]
    if "attack_window_reference_s" in metadata:
        metadata["attack_window_semantics"] = reference_metadata.get("attack_window_semantics", "Coarse playback launch through explicit return-blend start; includes source recovery, not reviewed outward/follow-through segmentation")
        metadata["attack_window_reviewed"] = bool(reference_metadata.get("attack_window_reviewed", False))
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/scalar-reference-v3/jab.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/scalar-evaluation-bank"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    print(json.dumps(make_bank(args.model, args.reference, args.output,
                              count=args.count, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
